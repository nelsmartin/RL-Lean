import json
from pantograph.server import Server
from pathlib import Path
from lean_dojo_v2.database import DynamicDatabase

def build_so_tactic(
        close: list[str] | None = None,
        hyp: list[str] | None = None,
        var: list[str] | None = None,
        func: list[str] | None = None
    ) -> str:
        config = {}
        if close: config["close"] = close
        if hyp:   config["hyp"]   = hyp
        if var:   config["var"]   = var
        if func:  config["func"]  = func
        inner = json.dumps(config).replace('"', '\\"')
        return f'so "{inner}"'


oracle_tactic = build_so_tactic(
                      close=["constructor"],
                      hyp=["induct_rename", "apply", "cases"],
                      var=["intro"],
                      func=["Nat.succ"])

server = Server(
    project_path=str(Path(__file__).parent.parent / "lean-stuff"),
    imports=["Init", "LeanStuff.OracleTactic"]
)

goal_expr = "∀ n : Nat, 0 + n = n"
initial_state = server.goal_start(goal_expr)
initial_state = server.goal_tactic(initial_state, "intro h")


suggestion_state = server.goal_tactic(initial_state, oracle_tactic)
for msg in suggestion_state.messages:
        data = json.loads(msg.data)
        for move in data.get("nextMoves", []):
            print(move)


path = "/lean-stuff"