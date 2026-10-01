"""Training helpers shared by pre-training and fine-tuning."""
import shutil
from pathlib import Path

import torch
from transformers import TrainerCallback, TrainingArguments

from common import REPORT_TO


class PermanentDeleteCallback(TrainerCallback):
    """Keep the two newest checkpoints plus the best one and delete the rest for good.

    Deleting through the Trainer sends folders to the Google Drive trash, where they
    still count against quota. shutil.rmtree bypasses that.
    """

    def on_save(self, args, state, control, **kwargs):
        checkpoints = sorted(Path(args.output_dir).glob('checkpoint-*'),
                             key=lambda p: int(p.name.split('-')[1]))
        keep = {p.name for p in checkpoints[-2:]}
        if state.best_model_checkpoint:
            keep.add(Path(state.best_model_checkpoint).name)
        for path in checkpoints:
            if path.name not in keep:
                shutil.rmtree(path)
                print(f'Deleted {path.name}')


def make_args(output_dir, run_name, **kwargs):
    return TrainingArguments(
        output_dir=str(output_dir), run_name=run_name, weight_decay=0.01,
        fp16=torch.cuda.is_available(), report_to=REPORT_TO, **kwargs,
    )
