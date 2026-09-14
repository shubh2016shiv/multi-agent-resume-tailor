"""Resume enhancement pipeline -- a LangGraph multi-agent orchestration.

START HERE IF YOU ARE NEW TO THIS PACKAGE.

THE ACTUAL PROCESS ENTRY POINT IS NOT IN THIS PACKAGE
    Running the CLI (`python -m src.main` / the packaged entry point) calls
    src/main.py:main(), which calls one of the two functions this package
    exports below. web_app/server.py calls the same two functions. Neither
    caller talks to the graph directly -- this package is the front door.

THE TWO FUNCTIONS (both in runner.py)
    tailor_resume(resume_path, jd_path)  -- start a fresh run
    resume_paused_run(paused_run_path)   -- continue a run that paused for
                                             candidate answers

WHAT EACH ONE DOES, THE SHORT VERSION
    build the starting state (a dict)  ->  build_resume_enhancement_graph()
    (see state.py)                         (see graph.py)
                                                |
                                                v
                                        compiled_graph.invoke(state)
                                                |
                        +-----------------------+-----------------------+
                        |                                               |
                a node called interrupt()                    the graph ran to the end
                (see nodes/experience/node.py)                          |
                        |                                               v
                        v                                   the final state dict, every
            state was saved to the checkpoint              field every node along the
            (see checkpointing.py) -- process               way filled in
            can exit; resume_paused_run() picks
            this same thread_id back up later

READ THE MODULES IN THIS ORDER
    state.py   -- the shared dict every node reads and writes (30 seconds)
    graph.py   -- the whole topology: which node runs after which, and why.
                  This file IS the architecture diagram; the other files are
                  its footnotes.
    nodes/     -- one node is one function: state in, a partial dict out. Read
                  any single small one (strategy.py is a good first pick).
    runner.py  -- the two public functions above, and the pause/resume plumbing.

DEBUGGING A GRAPH IS NOT DEBUGGING A LINEAR PROGRAM
    There is no single call stack to read: Stage 1 and Stage 3 run their nodes
    in separate threads (see graph.py's fan-out edges), and a paused run's
    "stack" is a state dict sitting in a SQLite file, not a live Python frame.
    Three things replace stepping through a debugger:

    1. Structured logs, filtered by run_id. Every run's run_id ties its whole
       story together across threads and across process restarts. The events
       to grep for, in the order a run produces them:
         pipeline_run_started / pipeline_run_completed / pipeline_run_failed
                                                          (runner.py, once per run)
         pipeline_stage_started / pipeline_stage_completed
                                                          (one pair per node, see
                                                           nodes/_stage.py)
         graph_routing_decision                          (one per conditional
                                                           edge; names which
                                                           branch was taken and
                                                           why -- see graph.py)
         agent_task_started / agent_task_completed        (one pair per LLM call,
                                                           see crew_task_execution.py)

    2. DEBUG_CHECKPOINTS=1 (env var). Writes every agent's exact prompt and raw
       response to checkpoints/<timestamp>_<run_id>/, in call order. This is how
       you see what an LLM actually saw and actually said, not what the code
       assumed it would say. See src/checkpointing.py.

    3. Call one node directly. A node is a plain function -- state in, dict
       out (see nodes/ above) -- so you do not need to run the whole graph to
       inspect one stage. In a REPL:

           from src.orchestration.nodes import strategy
           fake_state = {"run_id": "debug", "resume": ..., "job_description": ...}
           strategy.run_gap_analysis(fake_state)   # runs just this one stage

       This works for every node, decorated or not: @pipeline_stage only wraps
       a node to add the two log events above, it does not change the
       state-in/dict-out contract.
"""

from src.orchestration.runner import resume_paused_run, tailor_resume

__all__ = ["resume_paused_run", "tailor_resume"]
