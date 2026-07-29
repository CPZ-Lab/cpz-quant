"""DataFrame-native inputs: accept Polars and pandas anywhere returns go.

Every allocator, covariance estimator, and pre-selection transformer in
cpz-quant accepts its per-asset return series as any of:

- ``polars.DataFrame`` — numeric columns become assets (recommended)
- ``pandas.DataFrame`` — numeric columns become assets
- ``{asset_id: [returns]}`` mapping — the internal wire format

Temporal, string, and boolean columns (date indexes, labels) are treated
as metadata and excluded. A frame with **no** numeric columns raises —
never silently produces an empty universe.

Neither polars nor pandas is imported unless you actually pass one, so
the core dependency set stays lean.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def as_returns(data: Any) -> Any:
    """Coerce a Polars/pandas DataFrame to ``{column: [values]}``.

    Non-frame inputs (mappings, numpy arrays) pass through unchanged, so
    this is safe to call unconditionally at any API boundary.
    """
    cls = type(data)
    root = cls.__module__.split(".")[0]
    if root == "polars" and cls.__name__ == "DataFrame":
        out = {
            name: data[name].to_list()
            for name, dtype in zip(data.columns, data.dtypes)
            if dtype.is_numeric()
        }
        if not out:
            raise ValueError(
                "Polars DataFrame has no numeric columns to use as return "
                f"series (columns: {list(data.columns)}). Temporal/string "
                "columns are treated as labels and excluded."
            )
        return out
    if root == "pandas" and cls.__name__ == "DataFrame":
        numeric = data.select_dtypes(include="number")
        if numeric.shape[1] == 0:
            raise ValueError(
                "pandas DataFrame has no numeric columns to use as return "
                f"series (columns: {list(data.columns)}). Non-numeric "
                "columns are treated as labels and excluded."
            )
        return {str(c): numeric[c].tolist() for c in numeric.columns}
    return data


def frame_friendly(fn: F) -> F:
    """Decorator: run :func:`as_returns` on the function's first argument."""
    first_param = next(iter(inspect.signature(fn).parameters))

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if args:
            args = (as_returns(args[0]),) + args[1:]
        elif first_param in kwargs:
            kwargs[first_param] = as_returns(kwargs[first_param])
        return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
