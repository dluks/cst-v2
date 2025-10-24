"""Parses a DVC params.yaml file to return a config object that can be used in other scripts."""

import os
from pathlib import Path

import yaml
from box import ConfigBox
from dotenv import find_dotenv, load_dotenv


def parse_params(params_path: str | Path | None = None) -> dict:
    """
    Parse a DVC params.yaml file to return a config object that can be used in other scripts.
    Note that this requires a PROJECT_ROOT variable to be set as an environment variable
    in a local .env file.
    """
    if params_path is None:
        load_dotenv(find_dotenv())
        params_path = Path(os.environ["PROJECT_ROOT"]) / "params.yaml"

    if not Path(params_path).exists():
        raise FileNotFoundError(f"Params file not found at {params_path}")

    with open(params_path, encoding="utf-8") as f:
        return yaml.safe_load(f.read())


def get_config(
    params_path: str | Path | None = None, subset: str | None = None
) -> ConfigBox:
    """
    Get a ConfigBox object from a dictionary.
    """
    if subset is not None:
        return ConfigBox(parse_params(params_path))[subset]

    return ConfigBox(parse_params(params_path))


if __name__ == "__main__":
    print(get_config())
