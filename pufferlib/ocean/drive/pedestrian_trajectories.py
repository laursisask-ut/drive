"""Shared, memory-mapped replay pools for pedestrians and electric scooters."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

import numpy as np

from pufferlib.ocean.drive.pedestrian_routing import (
    CACHE_VERSION, DEFAULT_TRAJECTORY_POINTS, pedestrian_motion,
)


POOL_VERSION = 1  # Bump when routing, sampling, or the replay layout changes.
WALKER = 0
SCOOTER = 1
SPEED_RANGES = ((.8, 1.8), (3., 6.))
CHANNELS = ('x', 'y', 'vx', 'vy', 'heading')  # Native simulator coordinates.


class RouteTrees:
    """Reuse native routing trees while retaining independently sampled endpoints.

    Long trips head towards a routing anchor near their destination, then finish
    with a local route. The iterator computes that last part only if replay reaches
    it. This avoids solving hundreds of thousands of complete city-wide routes.
    """
    def __init__(self, network, count=128):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components, dijkstra

        self.network = network
        positions = np.asarray(network.positions)
        candidates = np.asarray(network.spawn_indices)
        xy = positions[candidates]
        chosen = []
        nearest = np.full(len(candidates), np.inf)
        index = int(np.argmax(np.sum((xy - xy.mean(axis=0)) ** 2, axis=1)))
        for _ in range(min(count, len(candidates))):
            chosen.append(int(candidates[index]))
            nearest = np.minimum(nearest, np.sum((xy - xy[index]) ** 2, axis=1))
            nearest[index] = -np.inf
            index = int(np.argmax(nearest))
        self.anchors = np.asarray(chosen)
        self.nearest_anchor = np.empty(len(positions), dtype=np.int32)
        for start in range(0, len(positions), 4096):
            delta = positions[start:start + 4096, None, :] - positions[self.anchors][None, :, :]
            self.nearest_anchor[start:start + 4096] = np.argmin(np.sum(delta * delta, axis=2), axis=1)
        degrees = np.diff(network.following_offsets)
        rows = np.repeat(np.arange(len(positions)), degrees)
        weights = np.repeat(network.lengths, degrees)
        graph = csr_matrix((weights, (rows, network.following_indices)),
                           shape=(len(positions), len(positions)))
        if connected_components(graph, directed=True, connection='strong', return_labels=False) != 1:
            raise ValueError('Pedestrian routing graph is not strongly connected')
        _, self.next_hop = dijkstra(graph.T.tocsr(), directed=True,
                                    indices=self.anchors, return_predecessors=True)
        # The native graph must permit all starting lanes to reach every anchor.
        invalid = self.next_hop < 0
        invalid[np.arange(len(self.anchors)), self.anchors] = False
        if invalid.any():
            raise ValueError('Pedestrian routing graph is not strongly connected')

    def path(self, start, goal, return_to_start=False):
        network = self.network
        if return_to_start or np.linalg.norm(np.subtract(network.positions[start], network.positions[goal])) < 100.:
            yield from network.path(start, goal, return_to_start)
            return
        tree = int(self.nearest_anchor[goal])
        anchor = int(self.anchors[tree])
        node = start
        while node != anchor:
            yield node
            node = int(self.next_hop[tree, node])
        yield from network.path(anchor, goal)


class TrajectoryPool:
    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        if json.loads((self.directory / 'metadata.json').read_text()) != metadata:
            raise ValueError('Stale pedestrian trajectory pool')
        self.metadata = metadata
        self.sequences = np.load(self.directory / 'sequences.npy', mmap_mode='r', allow_pickle=False)
        self.lengths = np.load(self.directory / 'lengths.npy', mmap_mode='r', allow_pickle=False)
        self.goals = np.load(self.directory / 'goals.npy', mmap_mode='r', allow_pickle=False)
        self.corridors = np.load(self.directory / 'corridors.npy', mmap_mode='r', allow_pickle=False)
        self.lane_ids = np.load(self.directory / 'lane_ids.npy', mmap_mode='r', allow_pickle=False)
        self.kinds = np.load(self.directory / 'kinds.npy', mmap_mode='r', allow_pickle=False)
        self.size = metadata['size']
        steps = metadata['max_points']
        if (self.sequences.shape != (self.size, len(CHANNELS), steps)
                or self.sequences.dtype != np.float32
                or self.lengths.shape != (self.size,) or self.lengths.dtype.kind not in 'iu'
                or np.any(self.lengths < 2) or np.any(self.lengths > steps)
                or self.goals.shape != (self.size, 2) or not np.isfinite(self.goals).all()
                or self.corridors.shape != (self.size,) or np.any(self.corridors < 0)
                or self.lane_ids.shape != (self.size,)
                or self.kinds.shape != (self.size,) or not np.isin(self.kinds, [WALKER, SCOOTER]).all()):
            raise ValueError('Invalid pedestrian trajectory pool')

    def trajectory(self, index):
        return self.sequences[index, :, :int(self.lengths[index])]


def pool_metadata(map_path, origin, size, max_points, sample_rate, scooter_fraction, seed):
    digest = hashlib.sha256()
    with open(map_path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(version=POOL_VERSION, routing_version=CACHE_VERSION,
                map_sha256=digest.hexdigest(), origin=list(origin), size=size,
                max_points=max_points, sample_rate=sample_rate,
                scooter_fraction=scooter_fraction, seed=seed,
                speed_ranges=[list(pair) for pair in SPEED_RANGES], channels=list(CHANNELS))


def build_pool(directory, metadata, network):
    size, steps = metadata['size'], metadata['max_points']
    sequences = np.lib.format.open_memmap(directory / 'sequences.npy', mode='w+',
                                          dtype=np.float32, shape=(size, len(CHANNELS), steps))
    lengths = np.empty(size, dtype=np.uint16)
    goals = np.empty((size, 2), dtype=np.float64)
    corridors = np.empty(size, dtype=np.int64)
    lane_ids = np.empty(size, dtype=np.int64)
    rng = random.Random(metadata['seed'])  # Never consume episode randomness.
    scooters = round(size * metadata['scooter_fraction'])
    kinds = [SCOOTER] * scooters + [WALKER] * (size - scooters)
    rng.shuffle(kinds)
    print(f'Precomputing {size:,} pedestrian/scooter trajectories...', flush=True)
    trees = RouteTrees(network)
    started = time.perf_counter()
    for i, kind in enumerate(kinds):
        start, sa = network.sample_position(rng)
        goal, sb = network.sample_position(rng)
        speed = rng.uniform(*metadata['speed_ranges'][kind])
        path = trees.path(start, goal, start == goal and sb < sa)
        points = network.trajectory(start, sa, goal, sb, speed,
                                    metadata['sample_rate'], steps, path=path)
        length = len(points)
        row = sequences[i]
        row[:2, :length] = -points.T
        row[:2, length:] = row[:2, length - 1, None]
        # Compute motion from the float32 coordinates actually replayed by C.
        heading, velocity = pedestrian_motion(-row[:2, :length].T.astype(np.float64),
                                               metadata['sample_rate'])
        row[2:4, :length] = velocity.T
        row[2:4, length:] = 0.
        row[4, :length] = heading
        row[4, length:] = heading[-1]
        lengths[i] = length
        goals[i] = network.point(goal, sb)
        corridors[i] = network.corridor_ids[start]
        lane_ids[i] = network.lane_ids[start]
        if (i + 1) % 8192 == 0 or i + 1 == size:
            print(f'  {i + 1:,}/{size:,} trajectories ({time.perf_counter() - started:.1f} s)', flush=True)
    sequences.flush()
    del sequences
    for name, values in [('lengths', lengths), ('goals', goals), ('corridors', corridors),
                         ('lane_ids', lane_ids), ('kinds', np.asarray(kinds, dtype=np.uint8))]:
        np.save(directory / f'{name}.npy', values, allow_pickle=False)
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


def load_trajectory_pool(map_path, origin, cache_root, network_loader, *,
                         size=262144, max_points=DEFAULT_TRAJECTORY_POINTS, sample_rate=.1,
                         scooter_fraction=.1, seed=0):
    if size < 1 or not 2 <= max_points <= 65535 or sample_rate <= 0 or not 0 <= scooter_fraction <= 1:
        raise ValueError('Invalid pedestrian trajectory pool settings')
    metadata = pool_metadata(map_path, origin, size, max_points, sample_rate, scooter_fraction, seed)
    cache_root = Path(cache_root)
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:24]
    directory = cache_root / key
    try:
        return TrajectoryPool(directory, metadata)
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        pass
    cache_root.mkdir(parents=True, exist_ok=True)
    # A worker builds once; other workers wait and then share the read-only files.
    with (cache_root / f'{key}.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return TrajectoryPool(directory, metadata)
        except (OSError, ValueError, KeyError, TypeError, EOFError):
            pass
        temporary = Path(tempfile.mkdtemp(prefix=f'.{key}.', dir=cache_root))
        try:
            build_pool(temporary, metadata, network_loader())
            # Validate the completed pool before publishing it to readers.
            checked = TrajectoryPool(temporary, metadata)
            del checked
            if directory.exists():
                shutil.rmtree(directory)
            os.replace(temporary, directory)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return TrajectoryPool(directory, metadata)
