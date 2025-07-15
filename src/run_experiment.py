"""
self‑contained pipeline to reproduce the main experiments:
(1) pre‑train on Caravan
(2) generate predictions on test period for pre-trained model
(3) fine‑tune on Uba local data
(4) generate predictions on test period for fine-tuned model

Edit CONFIG_PRETRAIN, CONFIG_FINETUNE, RUN_DIR_PRETRAIN, RUN_DIR_FINETUNE to match paths correctly.
"""

from pathlib import Path
import torch
from neuralhydrology.nh_run import start_run, eval_run, finetune

# ------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------
CONFIG_PRETRAIN  = Path("path/to/configs/file.yml")
CONFIG_FINETUNE  = Path("path/to/configs/file.yml")

RUN_DIR_PRETRAIN = Path("path/to/run/directory")
RUN_DIR_FINETUNE = Path("path/to/run/directory")
# ------------------------------------------------------------------

# 2. Choose device automatically
device_kwargs = {}                     # default: use first CUDA GPU
if not torch.cuda.is_available() and not torch.backends.mps.is_available():
    print("No GPU/MPS found – using CPU")
    device_kwargs["gpu"] = -1

# 3. Pre‑train
print(f"Pre‑training with {CONFIG_PRETRAIN}")
start_run(config_file=CONFIG_PRETRAIN, **device_kwargs)

# 4. Generate predictions for pre‑trained model
print("Generating predictions for pre‑trained model (test period)")
eval_run(run_dir=RUN_DIR_PRETRAIN, period="test")

# 5. Fine‑tune
print(f"Fine‑tuning with {CONFIG_FINETUNE}")
finetune(config_file=CONFIG_FINETUNE, **device_kwargs)

# 6. Generate predictions for fine‑tuned model
print("Generating predictions for fine‑tuned model (test period)")
eval_run(run_dir=RUN_DIR_FINETUNE, period="test")
