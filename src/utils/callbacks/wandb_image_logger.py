from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor

Stats = tuple[tuple[float, ...], tuple[float, ...]]


@dataclass
class _Buffer:
    limit: int
    images: list[Tensor] = field(default_factory=list)
    captions: list[str | None] = field(default_factory=list)


class WandbImageLogger(L.Callback):
    STATS: ClassVar[Mapping[str, Stats]] = {
        "standard": ((0.5,), (0.5,)),
        "imagenet": (
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    }

    def __init__(self, num_samples: int = 8) -> None:
        super().__init__()
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        self.num_samples = num_samples
        self.outputs: dict[str, _Buffer] = {}

    @staticmethod
    def _logger(trainer: L.Trainer) -> WandbLogger | None:
        return next(
            (x for x in trainer.loggers if isinstance(x, WandbLogger)),
            None,
        )

    @classmethod
    def _denormalize(
        cls,
        images: Tensor,
        spec: str | Mapping[str, Any],
    ) -> Tensor:
        if isinstance(spec, str):
            mean, std = cls.STATS[spec]
        else:
            mean, std = spec["mean"], spec["std"]

        channels = images.shape[1]
        mean = torch.as_tensor(mean, dtype=images.dtype).flatten()
        std = torch.as_tensor(std, dtype=images.dtype).flatten()

        if mean.numel() not in (1, channels):
            raise ValueError(f"Expected 1 or {channels} mean values")
        if std.numel() not in (1, channels):
            raise ValueError(f"Expected 1 or {channels} std values")

        return images * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

    @staticmethod
    def _to_uint8(
        images: Tensor,
        value_range: tuple[float, float] | None,
    ) -> Tensor:
        if images.dtype == torch.uint8 and value_range is None:
            return images

        low, high = value_range or (0.0, 1.0)

        if high <= low:
            raise ValueError(f"Invalid value_range: {value_range}")

        images = torch.nan_to_num(
            images.float(),
            nan=low,
            posinf=high,
            neginf=low,
        )

        return (
            ((images - low) / (high - low)).clamp(0, 1).mul(255).round().to(torch.uint8)
        )

    def update(
        self,
        trainer: L.Trainer,
        outputs: Tensor | Mapping[str, Any] | None,
    ) -> None:
        if not isinstance(outputs, Mapping):
            return

        items = outputs.get("wandb_image_logger")
        if not isinstance(items, Mapping):
            return

        for key, cfg in items.items():
            total = int(cfg.get("num_samples", self.num_samples))
            if total <= 0:
                raise ValueError("num_samples must be positive")

            local_limit = total // trainer.world_size
            local_limit += int(trainer.global_rank < total % trainer.world_size)

            buffer = self.outputs.setdefault(key, _Buffer(total))
            if buffer.limit != total:
                raise ValueError(f"num_samples changed for image key {key!r}")

            remaining = local_limit - len(buffer.images)
            if remaining <= 0:
                continue

            images: Tensor = cfg["images"]

            if images.ndim != 4 or images.shape[1] not in (1, 3, 4):
                raise ValueError(f"Expected Bx(1|3|4)xHxW, got {tuple(images.shape)}")

            images = images.detach().cpu()[:remaining]

            if "denormalize_from" in cfg:
                images = self._denormalize(
                    images.float(),
                    cfg["denormalize_from"],
                )

            images = self._to_uint8(
                images,
                cfg.get("value_range"),
            )

            captions = cfg.get("captions")

            if captions is None:
                captions = [None] * len(images)
            else:
                captions = list(captions[: len(images)])

            if len(captions) != len(images):
                raise ValueError("Number of captions must match images")

            buffer.images.extend(images)
            buffer.captions.extend(captions)

    def log_outputs(self, trainer: L.Trainer) -> None:
        logger = self._logger(trainer)
        local, self.outputs = self.outputs, {}

        if logger is None:
            return

        if trainer.world_size > 1:
            gathered: list[dict[str, _Buffer] | None] = [None] * trainer.world_size

            torch.distributed.all_gather_object(gathered, local)

            if not trainer.is_global_zero:
                return

            merged: dict[str, _Buffer] = {}

            for rank_outputs in gathered:
                for key, value in (rank_outputs or {}).items():
                    buffer = merged.setdefault(
                        key,
                        _Buffer(value.limit),
                    )
                    buffer.images.extend(value.images)
                    buffer.captions.extend(value.captions)
        else:
            merged = local

        for key, buffer in merged.items():
            images = buffer.images[: buffer.limit]

            if images:
                logger.log_image(
                    key=key,
                    images=images,
                    caption=buffer.captions[: len(images)],
                    step=trainer.global_step,
                )

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not trainer.sanity_checking and self._logger(trainer) is not None:
            self.update(trainer, outputs)

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if trainer.sanity_checking:
            self.outputs.clear()
        else:
            self.log_outputs(trainer)

    on_test_batch_end = on_validation_batch_end
    on_test_epoch_end = on_validation_epoch_end
