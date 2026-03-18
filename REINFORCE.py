import random
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
COMMIT = "2f0e298bc4891070b6f02acbd040a1b82936b521"
FILE_PATH = "LeanStuff/SimpleTheorems.lean"
NUM_EPOCHS = 200
BATCH_SIZE = 32
GAMMA = 0.9
LR = 1e-3
MAX_STEPS = 10

# --- Setup ---
server = Server(
    project_path=REPO_URL,
    imports=["Init", "LeanStuff.OracleTactic", "LeanStuff.SimpleTheorems", "LeanStuff.CurriculumTest"],
)

oracle_tactic = OracleTactic(
    close=["constructor"],
    hyp=["induct_rename", "apply", "cases"],
    var=["intro"],
    func=["Nat.succ"],
)

policy = PolicyNetwork()
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

theorems = get_theorems(REPO_URL, COMMIT, FILE_PATH)
goals = [theorem_to_goal(server, theorem) for theorem in theorems]

# --- Build vocabulary ---
vocab = Vocab()
for goal in goals:
    for token in tokenize(goal):
        vocab.add_token(token)
for tactic in oracle_tactic.get_all_tactics():
    for token in tokenize(tactic):
        vocab.add_token(token)

# --- Training loop ---
baseline = 0.0
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
        goal = random.choice(goals)
        state = server.goal_start(goal)
        rewards = []
        log_probs = []
        num_steps = 0

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
            state, reward, done = step(server, state, action)
            rewards.append(reward)
            num_steps += 1

            if done and reward == 1:
                epoch_solves += 1
                epoch_steps.append(num_steps)

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
