"""Record and compare the library versions a model was pickled with.

Estimators are pickled by reference to their class, not by value, so loading a
model under a different library version silently rebuilds it from whatever the
installed code now does. scikit-learn warns about this and keeps going. For a
fraud model that is the worst failure mode available: predictions that are
wrong but plausible, with a 200 response and no error anywhere.

Serving pins should make a mismatch impossible, but pins drift and base images
get rebuilt, so the model carries the versions it was trained with and the API
checks them on load.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# Only libraries whose objects end up inside the pickled artifacts. Adding
# anything else here produces noise on every deploy.
PICKLE_BEARING = ("scikit-learn", "numpy", "scipy", "lightgbm", "xgboost", "joblib")


def collect() -> dict[str, str]:
    """Installed versions of the libraries that affect unpickling."""
    found = {}
    for name in PICKLE_BEARING:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def _minor(v: str) -> str:
    return ".".join(v.split(".")[:2])


def compare(trained_with: dict[str, str]) -> list[str]:
    """Human-readable mismatches between training and the current environment.

    Compared at minor-version granularity: patch releases do not change pickle
    layout, while minor releases regularly do.
    """
    if not trained_with:
        return []

    installed = collect()
    mismatches = []
    for name, trained in sorted(trained_with.items()):
        current = installed.get(name)
        if current is None:
            mismatches.append(f"{name}: trained with {trained}, not installed")
        elif _minor(current) != _minor(trained):
            mismatches.append(f"{name}: trained with {trained}, running {current}")
    return mismatches
