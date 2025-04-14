# tiny-lightning-hydra-template

## Setting up

### uv

- `uv sync`
- `source .venv/bin/activate`

### venv

- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

## Configuring

- fill in `config/core/default.yaml`
- create `config/default.yaml` based on prepared examples (`classifier.yaml`, `diffusion_image_enhancement.yaml`, `diffusion.yaml`)
- run: `pre-commit install`

## Running

- inside venv
- `./train.py experiment=???`
