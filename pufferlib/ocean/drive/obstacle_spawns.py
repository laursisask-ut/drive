"""Persistent, uniformly sampled obstacle positions near road lanelet surfaces."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import warnings
from zipfile import BadZipFile

import numpy as np


CACHE_VERSION = 1


def precompute_obstacle_positions(lanelets, *, count, max_distance, seed=0):
    """Reject town-bounds samples until count distinct points are near a road."""
    if count < 0 or not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("Invalid obstacle spawn settings")
    positions = np.empty((count, 2), dtype=np.float32)
    if count == 0:
        return positions

    # Geometry is needed only for preprocessing, never during map generation.
    import shapely
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree

    roads = []
    for lanelet in lanelets:
        if "subtype" not in lanelet.attributes or lanelet.attributes["subtype"] != "road":
            continue
        ring = [(p.x, p.y) for p in lanelet.leftBound]
        ring += [(p.x, p.y) for p in reversed(lanelet.rightBound)]
        if len(ring) < 3:
            continue
        polygon = shapely.make_valid(Polygon(ring))
        if not polygon.is_empty and polygon.area > 0:
            roads.append(polygon)
    if not roads:
        raise ValueError("Obstacle spawns require at least one road lanelet")

    bounds = np.asarray([road.bounds for road in roads])
    low = bounds[:, :2].min(axis=0) - max_distance
    high = bounds[:, 2:].max(axis=0) + max_distance
    tree = STRtree(roads)
    rng = np.random.default_rng(seed)  # Cache misses must not consume episode RNG.
    seen = set()
    filled = 0
    while filled < count:
        batch_size = min(65536, max(4096, (count - filled) * 4))
        # Validate the exact float32 coordinates that the native environment uses.
        candidates = rng.uniform(low, high, size=(batch_size, 2)).astype(np.float32)
        matches = tree.query(shapely.points(candidates), predicate="dwithin",
                             distance=np.nextafter(max_distance, 0.))
        for point in candidates[np.unique(matches[0])]:
            key = tuple(point)
            if key in seen:
                continue
            seen.add(key)
            positions[filled] = point
            filled += 1
            if filled == count:
                break
    return positions


def write_obstacle_cache(cache_path, metadata, positions):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, prefix=cache_path.name + ".",
                                         suffix=".tmp", delete=False) as out:
            temporary_path = Path(out.name)
            np.savez(out, metadata=json.dumps(metadata, sort_keys=True), positions=positions)
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_obstacle_positions(map_path, origin, cache_path, lanelets, *,
                            count=131072, max_distance=10., seed=0):
    digest = hashlib.sha256()
    with open(map_path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = dict(version=CACHE_VERSION, map_sha256=digest.hexdigest(), origin=list(origin),
                    count=count, max_distance=max_distance, seed=seed)
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if json.loads(data["metadata"].item()) != metadata:
                raise ValueError("Obstacle spawn cache is out of date")
            positions = data["positions"]
        if (positions.shape != (count, 2) or positions.dtype != np.float32
                or not np.isfinite(positions).all()):
            raise ValueError("Invalid obstacle spawn coordinates")
    except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile):
        positions = precompute_obstacle_positions(
            lanelets, count=count, max_distance=max_distance, seed=seed)
        try:
            write_obstacle_cache(cache_path, metadata, positions)
        except OSError as error:
            warnings.warn(f"Could not save obstacle spawn cache {cache_path}: {error}", RuntimeWarning)
    positions.flags.writeable = False
    return positions
