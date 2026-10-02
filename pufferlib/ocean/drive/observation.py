"""Shared Drive observation sizes; offsets are in the root README.md."""

# Shared buffer sizes; keep in sync with observation.h.
HISTORY_CAPACITY = 60


def ego_features(dynamics_model="classic"):
    if dynamics_model == "classic":
        return 95
    if dynamics_model == "jerk":
        return 98
    raise ValueError(f"Unknown dynamics model: {dynamics_model!r}")
