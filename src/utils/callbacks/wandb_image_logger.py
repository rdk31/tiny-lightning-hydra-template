from collections import defaultdict
from typing import Any, Mapping

import lightning as L
import torch
import wandb
from torch import Tensor


class WandbImageLogger(L.Callback):
    """Log array of images to W&B.

    To use, return dictionary in validation_step or test_step named `wandb_image_logger` specifying wandb key, images (tensor of size B, C, H, W) and optional captions (list[str] of size B).

    Args:
        num_samples: (int) The total number of images to log across all devices.

    Examples:
        ### Log images with captions
        ```python
            ...
            preds = torch.argmax(logits, dim=1)
            captions = []
            for pred, label in zip(preds, y):
                captions.append(f"pred: {pred} true: {label}")

            return {
                "wandb_image_logger": {"val/samples": {"images": x, "captions": captions}}
            }
        ```

        ### Log multiple images without captions
        ```python
            ...
            return {
                "wandb_image_logger": {
                    "test/clean": {"images": clean},
                    "test/noisy": {"images": noisy},
                }
            }
        ```
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        self.outputs = defaultdict(list)

    def update(self, trainer: L.Trainer, outputs: Tensor | Mapping[str, Any] | None):
        if isinstance(outputs, dict) and "wandb_image_logger" in outputs:
            for k, v in outputs["wandb_image_logger"].items():
                if len(self.outputs[k]) < self.num_samples // trainer.world_size:
                    images = v["images"].cpu().numpy()
                    captions = (
                        v["captions"] if "captions" in v else [None] * images.shape[0]
                    )

                    o = list(zip(list(images), captions))
                    self.outputs[k].extend(
                        o[
                            : min(
                                images.shape[0], self.num_samples // trainer.world_size
                            )
                        ]
                    )

    def log_outputs(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if trainer.world_size > 1:
            outputs = [None for _ in range(trainer.world_size)]
            torch.distributed.all_gather_object(outputs, self.outputs)

            self.outputs.clear()

            if not trainer.is_global_zero:
                return

            merged_dict = defaultdict(list)
            for d in outputs:
                for k, v in d.items():
                    merged_dict[k].extend(v)
        else:
            merged_dict = self.outputs

        for k, v in merged_dict.items():
            wandb_images = [
                wandb.Image(image.transpose(1, 2, 0), caption=caption)
                for image, caption in v[: self.num_samples]
            ]
            pl_module.logger.log_image(key=k, images=wandb_images)

        self.outputs.clear()

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.update(trainer, outputs)

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self.log_outputs(trainer, pl_module)

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.update(trainer, outputs)

    def on_test_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self.log_outputs(trainer, pl_module)
