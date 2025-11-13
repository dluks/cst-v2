# Python Module Structure Convention

When structuring Python modules in this codebase, follow this standardized order:

## Module Organization Pattern

1. **Module docstring** (at the very top)
2. **Imports** (standard library, third-party, local)
3. **Module-level constants** (if any)
4. **`cli()` function** (if module has CLI argument parsing)
5. **`main()` function** (if module has a main entry point)
6. **Helper functions** in order of how they are called by `main()`
   - Functions should be ordered by their call sequence in the execution flow
   - Functions called first appear first
   - Functions called later appear later
7. **`if __name__ == "__main__":` clause** (at the very end)

## Example Structure

```python
"""Module docstring explaining purpose."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.conf.conf import get_config


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--input", type=str, help="...")
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = cli()
    
    # Call helper functions in order
    data = load_data(args.input)
    processed = process_data(data)
    save_results(processed)


def load_data(input_path: str) -> pd.DataFrame:
    """Load data from file (called first by main)."""
    return pd.read_parquet(input_path)


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process the data (called second by main)."""
    return df.dropna()


def save_results(df: pd.DataFrame) -> None:
    """Save results to file (called third by main)."""
    df.to_parquet("output.parquet")


if __name__ == "__main__":
    main()
```

## Rationale

This structure provides:
- **Clear entry point**: Readers immediately see CLI and main logic at the top
- **Logical flow**: Functions appear in the order they are executed
- **Easy navigation**: Can trace execution path by reading top-to-bottom
- **Consistency**: All modules follow the same pattern

## Files Following This Pattern

Examples in the codebase:
- `src/features/build_cv_splits.py` (lines 57-145: `get_trait_range()` helper before main's call site)
- `stages/build_y.py`
- `stages/train_models.py`

## Note on Module-Level Functions

For utility modules without a `main()` function (e.g., `src/utils/training_utils.py`), organize functions:
1. By logical grouping (related functions together)
2. By dependency order (functions that call others appear after)
3. Public API functions before private helpers
