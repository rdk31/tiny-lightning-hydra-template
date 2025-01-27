from typing import Any, Mapping

import lightning as L
import torch
import wandb
from torch import Tensor


class WandbImageLogger(L.Callback):
    """Log array of images to W&B.

    To use, return dictionary in validation_step or test_step named `wandb_image_logger` specifying wandb key, images (tensor of size B, C, H, W), optional captions (list[str] of size B),
    optional denormalize_from (str or dict) to denormalize images before logging and optional num_samples (int) to override the specified number of samples to log.

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
                "wandb_image_logger": {"val/samples":
                    {
                        "images": x,
                        "captions": captions,
                        "denormalize_from": "imagenet",
                        "num_samples": 10,
                    }
                }
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

        self.standard_mean = torch.tensor([0.5, 0.5, 0.5])
        self.standard_std = torch.tensor([0.5, 0.5, 0.5])

        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225])

        self.outputs = {}

    def update(self, trainer: L.Trainer, outputs: Tensor | Mapping[str, Any] | None):
        if isinstance(outputs, dict) and "wandb_image_logger" in outputs:
            for k, v in outputs["wandb_image_logger"].items():
                if "samples" in v:
                    num_samples = v["samples"]
                else:
                    num_samples = self.num_samples

                if k not in self.outputs:
                    self.outputs[k] = {
                        "num_samples": num_samples,
                        "images": [],
                        "captions": [],
                    }

                if len(self.outputs[k]["images"]) < num_samples // trainer.world_size:
                    images = v["images"].detach().cpu()
                    captions = (
                        v["captions"] if "captions" in v else [None] * images.shape[0]
                    )

                    if "denormalize_from" in v:
                        denormalize_from = v["denormalize_from"]
                        if isinstance(denormalize_from, str):
                            if denormalize_from == "standard":
                                mean = self.standard_mean
                                std = self.standard_std
                            elif denormalize_from == "imagenet":
                                mean = self.imagenet_mean
                                std = self.imagenet_std
                            else:
                                raise ValueError(
                                    f"denormalize_from must be one of 'standard', 'imagenet' or a dict with mean and std, got {denormalize_from}"
                                )
                        elif isinstance(denormalize_from, dict):
                            mean = denormalize_from["mean"]
                            std = denormalize_from["std"]
                        else:
                            raise ValueError(
                                f"denormalize_from must be one of '[-1,1]', 'imagenet' or a dict with mean and std, got {denormalize_from}"
                            )

                        images = images * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)

                    self.outputs[k]["images"].extend(
                        images[
                            : min(images.shape[0], num_samples // trainer.world_size)
                        ]
                    )
                    self.outputs[k]["captions"].extend(captions)

    def log_outputs(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if trainer.world_size > 1:
            outputs = [None for _ in range(trainer.world_size)]
            torch.distributed.all_gather_object(outputs, self.outputs)

            self.outputs.clear()

            if not trainer.is_global_zero:
                return

            merged_dict = {}
            for d in outputs:
                for k, v in d.items():
                    if k not in merged_dict:
                        merged_dict[k] = {
                            "num_samples": v["num_samples"],
                            "images": [],
                            "captions": [],
                        }

                    merged_dict[k]["images"].extend(v["images"])
                    merged_dict[k]["captions"].extend(v["captions"])
        else:
            merged_dict = self.outputs

        for k, v in merged_dict.items():
            images_with_captions = list(zip(v["images"], v["captions"]))
            wandb_images = [
                wandb.Image(image.numpy().transpose(1, 2, 0), caption=caption)
                for image, caption in images_with_captions[: v["num_samples"]]
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
