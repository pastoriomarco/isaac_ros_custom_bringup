# Trocar pipeline validation scripts

- `gt.py OUT.json` (laptop, simulator domain): reads the simulator's true poses of the six trocars and the camera through `simulation_interfaces` `/get_entity_state`.
- `live_run.sh TAG SECONDS "LAUNCH ARGS"` (container): launches on domain 0 and runs `live_monitor.py`, which subscribes read-only and saves `live.json` plus RGB/mask pairs.
- `bag_run.sh TAG` (container): replays the recording on domain 78 (localhost only), with reliable QoS from `replay_qos.yaml`.
- `analyze.py RUN_DIR GT.json`: pose error against the nearest true trocar, long-axis angle, jitter, latency.
- `overlay.py RUN_DIR GT.json trocar_short.obj OUT.png`: draws the mask, YOLO boxes, true meshes and estimated mesh. Needs numpy, scipy and Pillow.

The scripts expect the build under `worktrees/isaac_ros5_trocar-build`, with the scripts copied into its `test/` folder.
