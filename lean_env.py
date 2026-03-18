import json
from pantograph.server import Server
from lean_dojo_v2.database import DynamicDatabase


class OracleTactic:
    def __init__(self, close=None, hyp=None, var=None, func=None):
        self.close = close
        self.hyp = hyp
        self.var = var
        self.func = func

    def to_string(self):
        config = {}
        if self.close: config["close"] = self.close
        if self.hyp:   config["hyp"]   = self.hyp
        if self.var:   config["var"]   = self.var
        if self.func:  config["func"]  = self.func
        inner = json.dumps(config).replace('"', '\\"')
        return f'so "{inner}"'

    def get_all_tactics(self):
        tactics = []
        for group in [self.close, self.hyp, self.var, self.func]:
            if group:
                tactics.extend(group)
        return tactics


def get_theorems(repo_url, commit, file_path):
    database = DynamicDatabase()
    repo = database.trace_repository(
        url=repo_url,
        commit=commit,
        build_deps=False,
    )
    if repo is None:
        raise ValueError("Repository not found")
    return repo.get_all_theorems


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
