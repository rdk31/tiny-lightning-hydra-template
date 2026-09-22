import os

from omegaconf import OmegaConf


def register_resolvers() -> None:
    OmegaConf.register_new_resolver(
        "slurm_prefix",
        lambda: (
            f"{os.environ['SLURM_JOB_ID']}_" if "SLURM_JOB_ID" in os.environ else ""
        ),
        replace=True,
    )
