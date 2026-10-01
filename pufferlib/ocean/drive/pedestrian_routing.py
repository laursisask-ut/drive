"""Compact, cached pedestrian geometry and native Lanelet2 following links."""
from bisect import bisect_right
from collections import defaultdict
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import random
import tempfile
import warnings
from zipfile import BadZipFile

import numpy as np
from lanelet2.io import loadRobust
from lanelet2.routing import RoutingGraph
from lanelet2.traffic_rules import create, Locations, Participants


CACHE_VERSION = 1
DEFAULT_TRAJECTORY_POINTS = 601  # Initial pose plus 600 steps (60 seconds at 10 Hz).


def pedestrian_motion(points, sample_rate=.1):
    """Return native headings and velocities for map-coordinate trajectory points."""
    delta = -np.diff(points, axis=0)
    headings = np.arctan2(delta[:, 1], delta[:, 0])
    moving = np.any(delta != 0., axis=1)
    if not moving.all():
        last_moving = np.maximum.accumulate(np.where(moving, np.arange(len(moving)), 0))
        headings = headings[last_moving]
    return np.r_[headings, headings[-1]], np.vstack((delta / sample_rate, [0., 0.]))


class PedestrianNetwork:
    def __init__(self, lane_ids, point_offsets, points, following_offsets,
                 following_indices, corridor_ids):
        self.lane_ids = np.asarray(lane_ids, dtype=np.int64)
        self.point_offsets = np.asarray(point_offsets, dtype=np.int64)
        self.points = np.asarray(points, dtype=np.float64)
        self.following_offsets = np.asarray(following_offsets, dtype=np.int64)
        self.following_indices = np.asarray(following_indices, dtype=np.int64)
        self.corridor_ids = np.asarray(corridor_ids, dtype=np.int64)
        n = len(self.lane_ids)
        if (not n or len(set(self.lane_ids)) != n
                or self.point_offsets.shape != (n + 1,)
                or self.following_offsets.shape != (n + 1,)
                or self.corridor_ids.shape != (n,)
                or self.points.ndim != 2 or self.points.shape[1] != 2
                or not np.isfinite(self.points).all()
                or self.point_offsets[0] != 0 or self.point_offsets[-1] != len(self.points)
                or np.any(np.diff(self.point_offsets) < 2)
                or self.following_offsets[0] != 0
                or self.following_offsets[-1] != len(self.following_indices)
                or np.any(np.diff(self.following_offsets) < 1)
                or np.any(self.following_indices < 0)
                or np.any(self.following_indices >= n)):
            raise ValueError('Invalid pedestrian routing cache')
        self.lines = tuple(self.points[a:b] for a, b in
                           zip(self.point_offsets[:-1], self.point_offsets[1:]))
        self.arcs = tuple(np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
                          for p in self.lines)
        if any(np.any(np.diff(s) <= 0) for s in self.arcs):
            raise ValueError('Pedestrian centerline contains repeated points')
        self.lengths = tuple(float(s[-1]) for s in self.arcs)
        self.positions = tuple(tuple(p[0]) for p in self.lines)
        self.following = tuple(tuple(map(int, self.following_indices[a:b])) for a, b in
                               zip(self.following_offsets[:-1], self.following_offsets[1:]))
        groups = defaultdict(list)
        for i, corridor in enumerate(self.corridor_ids):
            if corridor >= 0:
                groups[int(corridor)].append(i)
        self.corridors = tuple(tuple(groups[k]) for k in sorted(groups))
        if not self.corridors:
            raise ValueError('Pedestrian map contains no walking corridors')
        self.corridor_lengths = tuple(sum(self.lengths[i] for i in group)
                                      for group in self.corridors)
        self.spawn_indices = tuple(i for group in self.corridors for i in group)
        self.spawn_cumulative_lengths = np.cumsum(
            [self.lengths[i] for i in self.spawn_indices]).tolist()

    @classmethod
    def from_lanelet_map(cls, pedestrian_map):
        rules = create(Locations.Germany, Participants.Pedestrian)
        graph = RoutingGraph(pedestrian_map, rules)
        lanes = sorted((lane for lane in pedestrian_map.laneletLayer if rules.canPass(lane)),
                       key=lambda lane: lane.id)
        indices = {lane.id: i for i, lane in enumerate(lanes)}
        points, point_offsets = [], [0]
        following, following_offsets, corridors = [], [0], []
        for lane in lanes:
            xy = np.asarray([(p.x, p.y) for p in lane.centerline])
            keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-8]
            points.extend(xy[keep])
            point_offsets.append(len(points))
            following.extend(sorted(indices[n.id] for n in graph.following(lane)))
            following_offsets.append(len(following))
            corridors.append(int(dict(lane.attributes).get('pedestrian:corridor', -1)))
        return cls([lane.id for lane in lanes], point_offsets, points,
                   following_offsets, following, corridors)

    def sample_position(self, rng=random):
        lane = rng.choices(self.spawn_indices,
                           cum_weights=self.spawn_cumulative_lengths, k=1)[0]
        return lane, rng.uniform(0., self.lengths[lane])

    def point(self, lane, offset):
        offset = min(self.lengths[lane], max(0., offset))
        return np.array([np.interp(offset, self.arcs[lane], self.lines[lane][:, axis])
                         for axis in (0, 1)])

    def pose(self, lane, offset):
        p = self.point(lane, offset)
        i = min(bisect_right(self.arcs[lane], offset) - 1, len(self.lines[lane]) - 2)
        dx, dy = self.lines[lane][i + 1] - self.lines[lane][i]
        return float(p[0]), float(p[1]), math.atan2(dy, dx)

    def path(self, start, goal, return_to_start=False):
        """Find a goal-directed path using only native following relationships.

        Weighted A* avoids exhaustive city-wide searches for thousands of actors.
        Paths obey lane direction; they are not required to be shortest paths.
        """
        if start == goal and return_to_start:
            choices = [(start,) + self.path(n, goal) for n in self.following[start]]
            return min(choices, key=lambda p: sum(self.lengths[i] for i in p))
        target = self.positions[goal]
        frontier = [(0., 0., start)]
        cost = {start: 0.}
        previous = {}
        while frontier:
            _, traveled, node = heapq.heappop(frontier)
            if traveled > cost[node]:
                continue
            if node == goal:
                result = [goal]
                while result[-1] != start:
                    result.append(previous[result[-1]])
                return tuple(reversed(result))
            next_cost = traveled + self.lengths[node]
            for following in self.following[node]:
                if next_cost < cost.get(following, math.inf):
                    cost[following] = next_cost
                    previous[following] = node
                    heuristic = 2. * math.dist(self.positions[following], target)
                    heapq.heappush(frontier, (next_cost + heuristic, next_cost, following))
        raise ValueError(f'No pedestrian route from {self.lane_ids[start]} to {self.lane_ids[goal]}')

    def trajectory(self, start, start_offset, goal, goal_offset, speed,
                   sample_rate=.1, max_points=DEFAULT_TRAJECTORY_POINTS, path=None):
        if speed <= 0 or sample_rate <= 0 or max_points < 2:
            raise ValueError('Invalid pedestrian trajectory sampling settings')
        if not (0 <= start_offset <= self.lengths[start]
                and 0 <= goal_offset <= self.lengths[goal]):
            raise ValueError('Pedestrian route offset is outside its lane')
        if path is None:
            path = self.path(start, goal, start == goal and goal_offset < start_offset)
        step = speed * sample_rate
        remaining = (max_points - 1) * step
        chunks = []
        iterator = iter(path)
        lane = next(iterator)
        k = 0
        while True:
            following = next(iterator, None)
            begin = start_offset if k == 0 else 0.
            end = goal_offset if following is None else self.lengths[lane]
            end = min(end, begin + remaining)
            s = self.arcs[lane]
            chunks.extend((self.point(lane, begin)[None, :],
                           self.lines[lane][(s > begin) & (s < end)],
                           self.point(lane, end)[None, :]))
            remaining -= end - begin
            if remaining < 1e-8 or following is None:
                break
            lane = following
            k += 1
        points = np.concatenate(chunks)
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]]
        if len(points) == 1:
            return np.repeat(points, 2, axis=0)
        distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        # The last sample lands exactly on a reached goal and then stays there.
        ticks = min(max_points, int(math.ceil(distances[-1] / step)) + 1)
        samples = np.minimum(np.arange(ticks) * step, distances[-1])
        return np.column_stack([np.interp(samples, distances, points[:, axis]) for axis in (0, 1)])

    def arrays(self):
        return {name: getattr(self, name) for name in (
            'lane_ids', 'point_offsets', 'points', 'following_offsets',
            'following_indices', 'corridor_ids')}


def load_pedestrian_network(map_path, projector, origin, cache_path):
    """Cache only portable arrays; release the large native map after extraction."""
    digest = hashlib.sha256()
    with open(map_path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    metadata = dict(version=CACHE_VERSION, map_sha256=digest.hexdigest(), origin=list(origin))
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if json.loads(data['metadata'].item()) != metadata:
                raise ValueError('Stale pedestrian routing cache')
            return PedestrianNetwork(**{k: data[k] for k in (
                'lane_ids', 'point_offsets', 'points', 'following_offsets',
                'following_indices', 'corridor_ids')})
    except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile):
        pass
    pedestrian_map, errors = loadRobust(str(map_path), projector)
    if errors:
        raise ValueError(f'Could not load pedestrian map: {errors}')
    network = PedestrianNetwork.from_lanelet_map(pedestrian_map)
    del pedestrian_map
    cache_path = Path(cache_path)
    temporary_path = None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, prefix=cache_path.name + '.',
                                         suffix='.tmp', delete=False) as out:
            temporary_path = Path(out.name)
            np.savez(out, metadata=json.dumps(metadata, sort_keys=True), **network.arrays())
        os.replace(temporary_path, cache_path)
    except OSError as error:
        warnings.warn(f'Could not save pedestrian routing cache {cache_path}: {error}', RuntimeWarning)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return network
