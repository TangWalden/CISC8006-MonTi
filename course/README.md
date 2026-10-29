# Remote Execution

~~~bash
cd /home/tjy/Project/MonTi-course
bash course/run.sh -m course.test_protocol
bash course/run.sh -m course.runner --config best_models/configs/GCN-GAGA-gossipcop_0.05-best_config.json --phase pilot --mode learned --seed 0 --gpu 0 --train-limit 16 --valid-limit 8 --epochs 2
~~~

Freeze the protocol before formal runs. Feasibility-pilot results are not paper-reproduction scores; pilots do not evaluate the test split.

For each run, evidence/runs/<id>/registered.json is written before training. The resolved configuration is in config.json, target_groups.json records the selected groups, *.jsonl contains per-target predictions, and result.json records completion or failure. Logs and models are stored in a separate runs/<id>/ directory. Previous runs are retained.

The base dependencies come from the existing /home/tjy/miniconda3/envs/monti environment. Exact versions are recorded in environment/pip-freeze.txt and environment/runtime.json. The wrapper sets the CUDA library path and thread limits only for the current process; it does not change other environments.

The source code uses the upstream CC BY-NC 4.0 license; the original license and author attribution are retained. If a separate license for a data archive is not provided, record it as unspecified. Do not automatically apply the code license to the data.
