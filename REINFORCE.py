import os
import torch
from pantograph.server import Server

from lean_env import OracleTactic, get_theorems, theorem_to_goal, get_moves, step
from tokenizer import tokenize, Vocab, encode_text
from policy import PolicyNetwork
from plotting import plot_training_curves

# --- Config ---
# NOTE: Replace REPO_URL with the path to your local clone of
# https://github.com/nelsmartin/lean-stuff
REPO_URL = "/Users/nelsmartin/Lean/lean-stuff"
COMMIT = "2e162026d5b4bc822f8d6a961369c56832df252d"
FILE_PATH = "LeanStuff/Curriculum.lean"
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", 20))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
GAMMA = 0.9
LR = 1e-3
MAX_STEPS = 15

# --- Setup ---
server = Server(
    project_path=REPO_URL,
    imports=["Init", "LeanStuff.OracleTactic", "LeanStuff.SimpleTheorems",
             "LeanStuff.CurriculumTest", "LeanStuff.Curriculum"],
)

oracle_tactic = OracleTactic(
    close=["constructor"],
    hyp=["induct_rename", "apply", "cases"],
    var=["intro"],
    func=["Nat.succ"],
    rw=[],  # enable forward `rw [..]` moves (lemmas filled in per-episode)
)

policy = PolicyNetwork()
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

theorems = get_theorems(REPO_URL, COMMIT, FILE_PATH)
# Keep theorem identity (full_name) alongside its goal so we can record which
# theorems the agent has actually proved during the run.
problems = [(theorem.full_name, theorem_to_goal(server, theorem)) for theorem in theorems]

# --- Build vocabulary ---
vocab = Vocab()
for _, goal in problems:
    for token in tokenize(goal):
        vocab.add_token(token)
for tactic in oracle_tactic.get_all_tactics():
    for token in tokenize(tactic):
        vocab.add_token(token)
# Seed theorem names so dynamic `apply <name>` moves aren't all <UNK>.
for name, _ in problems:
    for token in tokenize(name):
        vocab.add_token(token)
# Seed literal tokens used by dynamic apply/exact/rw moves.
for token in ["apply", "exact", "rw", "[", "]"]:
    vocab.add_token(token)

# --- Training loop ---
baseline = 0.0
proved = set()  # full_names of theorems the agent has closed so far (curriculum)
episode_idx = 0  # walks `problems` in order, wrapping across epochs
epoch_solve_rates = []
epoch_avg_steps = []
epoch_avg_returns = []
epoch_losses = []

for epoch in range(NUM_EPOCHS):
    batch_loss = torch.tensor(0.0)
    epoch_solves = 0
    epoch_steps = []
    epoch_returns = []

    for _ in range(BATCH_SIZE):
        name, goal = problems[episode_idx % len(problems)]
        episode_idx += 1
        state = server.goal_start(goal)
        rewards = []
        log_probs = []
        num_steps = 0

        # Offer every proved theorem as an `apply`/`rw` lemma, except the current one
        # (prevents trivially closing a goal with `exact <itself>`).
        available = sorted(proved - {name})
        oracle_tactic.apply = available
        oracle_tactic.rw = available
        lemma_set = set(available)
        used_lemmas = []  # lemma moves the agent actually took this episode

        done = False
        while not done:
            moves = get_moves(server, state, oracle_tactic)

            if not moves or num_steps > MAX_STEPS:
                rewards.append(-1.0)
                break

            state_enc = encode_text(str(state.goals[0]), vocab)
            action_encs = [encode_text(move, vocab) for move in moves]

            probs = policy(state_enc, action_encs)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_probs.append(dist.log_prob(action_idx))

            action = moves[int(action_idx.item())]

            # Detect use of a previously-proved theorem as a lemma, via either
            # `apply/exact <thm>` or `rw [<thm>]`. (`rw [<hyp>]`, e.g. the induction
            # hypothesis, is not counted as cross-theorem reuse.)
            parts = action.split()
            used = None
            if len(parts) == 2 and parts[0] in ("apply", "exact") and parts[1] in lemma_set:
                used = parts[1]
            elif action.startswith("rw [") and action.endswith("]") and action[4:-1] in lemma_set:
                used = action[4:-1]
            if used is not None:
                used_lemmas.append(action)
                print(f"    [{name}] step {num_steps}: used lemma `{action}`")

            state, reward, done = step(server, state, action)
            rewards.append(reward)
            num_steps += 1

            if done and reward == 1:
                epoch_solves += 1
                epoch_steps.append(num_steps)
                is_new = name not in proved
                proved.add(name)
                if is_new:
                    print(f"  + first proof of {name} "
                          f"(proved set now: {len(proved)})")
                if used_lemmas:
                    print(f"  >> SOLVED {name} using lemma(s): {used_lemmas}")

        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = torch.tensor(returns)

        # REINFORCE loss with baseline
        episode_loss = torch.tensor(0.0)
        for t in range(len(log_probs)):
            episode_loss += -log_probs[t] * (returns[t] - baseline)

        baseline = 0.9 * baseline + 0.1 * returns[0].item()
        epoch_returns.append(returns[0].item())
        batch_loss = batch_loss + episode_loss

    optimizer.zero_grad()
    (batch_loss / BATCH_SIZE).backward()
    optimizer.step()

    solve_rate = epoch_solves / BATCH_SIZE
    avg_steps = sum(epoch_steps) / len(epoch_steps) if epoch_steps else float("nan")
    avg_return = sum(epoch_returns) / len(epoch_returns)
    loss_val = batch_loss.item() / BATCH_SIZE

    epoch_solve_rates.append(solve_rate)
    epoch_avg_steps.append(avg_steps)
    epoch_avg_returns.append(avg_return)
    epoch_losses.append(loss_val)

    print(f"Epoch {epoch:3d} | solve rate: {solve_rate:.2f} | avg steps: {avg_steps:.1f} "
          f"| avg return: {avg_return:.2f} | loss: {loss_val:.3f}")

# --- Plot results ---
plot_training_curves(epoch_solve_rates, epoch_avg_steps, epoch_avg_returns, epoch_losses)
