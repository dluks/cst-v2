from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def _eprint(message: str) -> None:
    """Write a message to stderr."""

    print(message, file=sys.stderr)


def _validate_file(path: Path) -> None:
    """Validate that a file exists at the given path."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")


def _load_transformer(pkl_path: Path):
    """Load a scikit-learn PowerTransformer from pickle.

    Returns the deserialized transformer object. This requires scikit-learn to
    be installed in the current environment.
    """

    try:
        import pickle

        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)
    except ModuleNotFoundError as exc:  # likely sklearn missing when unpickling
        raise ModuleNotFoundError(
            "Failed to import a required module while unpickling the transformer. "
            "Ensure scikit-learn is installed in this environment."
        ) from exc


def _extract_params(transformer: Any, traits: list[str]) -> Mapping[str, Any]:
    """Extract minimal Yeo-Johnson params for the requested traits.

    Parameters
    ----------
    transformer: Any
        The fitted sklearn PowerTransformer instance.
    traits: list[str]
        Trait column names to extract (must match `feature_names_in_`).

    Returns
    -------
    Mapping[str, Any]
        A dictionary with keys: method, standardize, traits -> per-trait params.
    """

    method: str = getattr(transformer, "method", "yeo-johnson")
    standardize: bool = bool(getattr(transformer, "standardize", True))

    feature_names_in: np.ndarray | None = getattr(
        transformer, "feature_names_in_", None
    )
    if feature_names_in is None:
        raise AttributeError(
            "Transformer is missing 'feature_names_in_'; cannot map traits."
        )

    lambdas: np.ndarray | None = getattr(transformer, "lambdas_", None)
    if lambdas is None:
        raise AttributeError("Transformer is missing 'lambdas_' values.")

    means = None
    scales = None
    if standardize:
        scaler = getattr(transformer, "_scaler", None)
        if scaler is not None:
            means = getattr(scaler, "mean_", None)
            scales = getattr(scaler, "scale_", None)

    feature_names: list[str] = [str(x) for x in list(feature_names_in)]
    name_to_index: dict[str, int] = {
        name: idx for idx, name in enumerate(feature_names)
    }

    missing = [t for t in traits if t not in name_to_index]
    if missing:
        raise KeyError(
            "Requested trait(s) not found in transformer columns: "
            f"{missing}. Available columns include examples like: "
            f"{feature_names[:8]}{' ...' if len(feature_names) > 8 else ''}"
        )

    result: dict[str, Any] = {
        "method": method,
        "standardize": standardize,
        "traits": {},
    }

    for trait in traits:
        i = name_to_index[trait]
        entry: dict[str, Any] = {"lambda": float(lambdas[i])}
        if means is not None and scales is not None:
            entry["mean"] = float(means[i])
            entry["scale"] = float(scales[i])
        result["traits"][trait] = entry

    return result


def _write_json(obj: Mapping[str, Any], out_path: Path) -> None:
    """Write the object as pretty-printed JSON to the path."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(obj, fh, indent=2)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for exporting power-transform parameters."""

    default_pkl = Path.cwd() / "data" / "interim" / "try" / "power_transformer.pkl"
    default_out = (
        Path.cwd() / "data" / "interim" / "try" / "power_params_for_share.json"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Export minimal Yeo-Johnson PowerTransformer parameters for selected "
            "traits into a JSON file (method, standardize, lambda, mean, scale)."
        )
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=default_pkl,
        help=f"Path to pickled transformer (default: {default_pkl})",
    )
    parser.add_argument(
        "--traits",
        nargs="+",
        required=True,
        help=(
            "Space-separated list of trait column names to export, e.g. "
            "X144_mean X3117_mean X3112_mean X50_mean"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"Output JSON path (default: {default_out})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI to export Yeo-Johnson PowerTransformer params for selected traits."""

    args = _parse_args(argv)

    try:
        _validate_file(args.pkl)
        transformer = _load_transformer(args.pkl)
        params = _extract_params(transformer, args.traits)
        _write_json(params, args.out)
        print(json.dumps(params, indent=2))
        print(f"\nWROTE: {args.out}")
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level CLI handler
        _eprint(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
