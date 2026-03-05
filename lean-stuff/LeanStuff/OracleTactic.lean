import Lean
import LeanStuff.InductRename
open Lean Meta Elab Tactic

private def tryTacticStr (env : Environment) (s : String) : TacticM Bool := do
    match Parser.runParserCategory env `tactic s "<so>" with
    | .error _ => return false
    | .ok stx  =>
      let saved ← saveState
      try
        evalTactic stx
        restoreState saved
        return true
      catch _ =>
        restoreState saved
        return false

  syntax "so" str : tactic
  elab_rules : tactic
    | `(tactic| so $cfg:str) => do
      let json ← match Lean.Json.parse cfg.getString with
        | .error e => throwError s!"so: bad config: {e}"
        | .ok j    => pure j
      let closeNames := (json.getObjValAs? (Array String) "close").toOption.getD #[]
      let hypNames   := (json.getObjValAs? (Array String) "hyp").toOption.getD #[]
      let varNames   := (json.getObjValAs? (Array String) "var").toOption.getD #[]
      let funcNames  := (json.getObjValAs? (Array String) "func").toOption.getD #[]
      let env  ← getEnv
      let lctx := (← (← getMainGoal).getDecl).lctx
      let mut moves : Array String := #[]

      let fresh : Name := lctx.getUnusedName `h

      -- Category 1: no-arg tactics (try directly)
      for t in closeNames do
        if ← tryTacticStr env t then moves := moves.push t

      -- Category 2: local-decl tactics (try against each hypothesis)
      for t in hypNames do
        for decl in lctx do
          if decl.isImplementationDetail then continue
          let ts := s!"{t} {decl.userName}"
          if ← tryTacticStr env ts then moves := moves.push ts

      -- Category 3: fresh-var tactics (try with a generated name)
      for t in varNames do
        let ts := s!"{t} {fresh}"
        if ← tryTacticStr env ts then moves := moves.push ts

      -- Category 4: func tactics (apply congrArg <f> for each named function)
      for f in funcNames do
        let ts := s!"apply congrArg {f}"
        if ← tryTacticStr env ts then moves := moves.push ts

      -- Emit result as JSON message (read by Python via suggestion_state.messages)
      let parts := moves.toList.map fun t => "{\"tactic\":\"" ++ t ++ "\"}"
      logInfo s!"\{\"nextMoves\":[{",".intercalate parts}]}"
