# Hand-tuned motions

`MotionLibrary.get(name)` loads `<name>.json` from this folder when it exists, otherwise it generates
the motion from `config/stack/motion.yaml`. Files appear here when you save in the studio
(`python -m robot.motion.studio <name>`) or run `python -m robot.motion.library --write`.

Walk pieces are `walk_forward_start.json`, `walk_forward_cycle.json`, `walk_forward_stop.json`
(same for `walk_backward`). Delete a file to go back to the generated version.
