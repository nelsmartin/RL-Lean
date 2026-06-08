import ast
import json
from pantograph.server import Server
from lean_dojo_v2.database import DynamicDatabase


class OracleTactic:
    def __init__(self, close=None, hyp=None, var=None, func=None, apply=None, rw=None):
        self.close = close
        self.hyp = hyp
        self.var = var
        self.func = func
        # Names of previously-proved theorems to offer as `apply`/`exact` moves.
        # Updated dynamically during training as the agent closes goals.
        self.apply = apply or []
        # Names of previously-proved theorems to offer as forward `rw [..]` moves.
        # `None` disables the rw feature entirely; `[]` enables it (so the induction
        # hypothesis can still be rewritten) with no lemmas yet.
        self.rw = rw

    def to_string(self):
        config = {}
        if self.close: config["close"] = self.close
        if self.hyp:   config["hyp"]   = self.hyp
        if self.var:   config["var"]   = self.var
        if self.func:  config["func"]  = self.func
        if self.apply: config["apply"] = self.apply
        if self.rw is not None: config["rw"] = self.rw
        inner = json.dumps(config).replace('"', '\\"')
        return f'so "{inner}"'

    def get_all_tactics(self):
        tactics = []
        for group in [self.close, self.hyp, self.var, self.func, self.apply, self.rw]:
            if group:
                tactics.extend(group)
        return tactics


def get_theorems(repo_url, commit, file_path=None):
    database = DynamicDatabase()
    repo = database.trace_repository(
        url=repo_url,
        commit=commit,
        build_deps=False,
    )
    if repo is None:
        raise ValueError("Repository not found")
    theorems = list(repo.get_all_theorems)
    if file_path is not None:
        theorems = [t for t in theorems if str(t.file_path) == file_path]
    # Sort by source position so an in-order curriculum is walked simple -> complex.
    def source_pos(theorem):
        try:
            return ast.literal_eval(str(theorem.start))
        except (ValueError, SyntaxError):
            return (0, 0)
    theorems.sort(key=source_pos)
    return theorems


def theorem_to_goal(server, theorem):
    return server.expr_type(f"@{theorem.full_name}")


def get_moves(server, state, oracle_tactic):
    moves = []
    suggestion_state = server.goal_tactic(state, oracle_tactic.to_string())
    for msg in suggestion_state.messages:
        data = json.loads(msg.data)
        for move in data.get("nextMoves", []):
            moves.append(move["tactic"])
    return moves


def step(server: Server, state, action):
    new_state = server.goal_tactic(state, action)
    if len(new_state.goals) == 0:
        reward = 1
        done = True
    else:
        reward = -0.1
        done = False
    return new_state, reward, done
