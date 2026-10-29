#ifndef DRIVE_OBSERVATION_H
#define DRIVE_OBSERVATION_H

// Shared buffer sizes; keep in sync with observation.py.
// Individual field offsets are documented in the root README.md.
#define EGO_HISTORY_CAPACITY 60

static inline int drive_ego_features(int dynamics_model) {
    return dynamics_model == 1 ? 100 : 97;
}

#endif
