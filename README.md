# RL-Lean

Reinforcement learning for automated Lean 4 theorem proving. A policy network learns to select proof tactics by interacting with the Lean prover through [PyPantograph](https://github.com/stanford-centaur/PyPantograph), using the REINFORCE algorithm.

## How it works

An **oracle tactic** (implemented on the Lean side) enumerates candidate proof steps for a given goal state. The RL agent learns to pick the right tactic at each step to close the proof. The training loop:

1. Samples a theorem from a Lean repository
2. Queries the oracle for available moves at each proof state
3. Uses a policy network to select a move
4. Receives reward (+1 for closing the proof, -0.1 per step, -1 for dead ends)
5. Updates the policy with REINFORCE + a running baseline

![Training curves](training_curves.png)

## Project structure

| File | Description |
|---|---|
| `REINFORCE.py` | Training script — config, training loop, entry point |
| `lean_env.py` | Lean prover interface — `OracleTactic`, `get_theorems`, `get_moves`, `step` |
| `policy.py` | Neural network — `SimpleEncoder` (bag-of-embeddings) and `PolicyNetwork` (state-action scorer) |
| `tokenizer.py` | Text processing — `tokenize`, `Vocab`, `encode_text` |
| `plotting.py` | Training curve visualization |

### Key definitions

**`OracleTactic`** (`lean_env.py`) — Wraps the Lean-side `so` tactic. Configured with four tactic groups:
- `close`: tactics that close a goal (e.g. `constructor`)
- `hyp`: tactics that operate on hypotheses (e.g. `induct_rename`, `apply`, `cases`)
- `var`: tactics that introduce variables (e.g. `intro`)
- `func`: function/constructor names to try (e.g. `Nat.succ`)

**`get_moves`** (`lean_env.py`) — Sends the oracle tactic to the Lean server and parses the suggested next moves from the response messages.

**`step`** (`lean_env.py`) — Applies a tactic string to the current proof state via Pantograph and returns `(new_state, reward, done)`.

**`PolicyNetwork`** (`policy.py`) — Scores each candidate action given the current goal state. Encodes both the goal and each action as bag-of-embeddings vectors, concatenates them, and passes through an MLP to produce a scalar score. Scores are softmaxed to get a distribution over actions.

**`Vocab`** (`tokenizer.py`) — Builds a token-to-id mapping on the fly. Special tokens: `<PAD>`, `<UNK>`, `<HYP>` (hypothesis names are normalized to `<HYP>`).

## Setup

### 1. Clone the Lean repository

```bash
git clone https://github.com/nelsmartin/lean-stuff
cd lean-stuff
lake build
```

### 2. Install Python dependencies

Requires Python 3.13. Using [uv](https://docs.astral.sh/uv/):

```bash
cd /path/to/RL-Lean
uv sync
```

### 3. Configure and run

Edit `REPO_URL` in `REINFORCE.py` to point to your local clone of `lean-stuff`:

```python
REPO_URL = "/path/to/your/lean-stuff"
```

Then run:

```bash
uv run python REINFORCE.py
```

Training curves are saved to `training_curves.png` on completion.

## Next steps

- **Curriculum learning** — Allow the agent to use previously proved theorems as lemmas when proving new ones, enabling it to build up a library of results and tackle increasingly difficult goals.
- **Alternative RL algorithms** — Support other policy gradient methods such as PPO or A2C, which may offer more stable training and better sample efficiency compared to vanilla REINFORCE.

## References

- Williams, R.J. (1992). [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://link.springer.com/article/10.1007/BF00992696). *Machine Learning*, 8(3-4), 229-256.
