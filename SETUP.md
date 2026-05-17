# OpenVLA — local setup

## Venv

```bash
python3 -m venv ~/envs/openvla
source ~/envs/openvla/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e /mnt/d/workspace/openvla
pip install uvicorn fastapi draccus
pip install -e /mnt/d/workspace/vla-bench
pip install -e /mnt/d/workspace/LIBERO
```

## Start server

```bash
source ~/envs/openvla/bin/activate
cd /mnt/d/workspace/openvla

python vla-scripts/deploy.py \
    --openvla_path openvla/openvla-7b \
    --attn_implementation sdpa \
    --host 0.0.0.0 --port 8000
```

> `--attn_implementation sdpa` is required on RTX 2080 Ti (sm75).
> FlashAttention2 requires sm80+ (Ampere) and will fail here.

## Sanity test

```bash
python - <<'EOF'
import numpy as np
from vla_bench.client import VLAClient

vla = VLAClient(
    server_url="http://localhost:8000",
    unnorm_key="nyu_franka_play_dataset_converted_externally_to_rlds",
)
action = vla.query(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8), "pick up the red cube")
print("action:", action)
EOF
```

## LIBERO eval

```bash
python /mnt/d/workspace/vla-bench/scripts/libero_eval.py \
    --suite libero_10 --task_id 0 \
    --server http://localhost:8000 \
    --unnorm_key nyu_franka_play_dataset_converted_externally_to_rlds \
    --num_episodes 5 \
    --out_dir /mnt/d/workspace/results/openvla
```

## Isaac Sim eval

```bash
cd /mnt/d/workspace/IsaacSim
./_build/linux-x86_64/release/python.sh \
    source/standalone_examples/vla/openvla_franka.py \
    --server http://localhost:8000 \
    --unnorm_key nyu_franka_play_dataset_converted_externally_to_rlds
```

## unnorm_key reference

OpenVLA was trained on 25 OXE datasets. Common keys:

| Robot | Key |
|---|---|
| Franka (Isaac Sim / generic) | `nyu_franka_play_dataset_converted_externally_to_rlds` |
| WidowX (BridgeData) | `bridge_orig` |
| Kuka iiwa | `kuka` |

Full list: `python -c "from transformers import AutoModelForVision2Seq; m = AutoModelForVision2Seq.from_pretrained('openvla/openvla-7b', trust_remote_code=True); print(list(m.norm_stats.keys()))"`

## Known bugs fixed in deploy.py

- `json_numpy.patch()` moved after all imports (was before — caused `TypeError` in numpy internals)
- `image = np.array(payload["image"], dtype=np.uint8)` added (image arrives as JSON list)
- `attn_implementation` field added to `DeployConfig` with default `"sdpa"`
