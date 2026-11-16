import os
import math
import time
import random
import json
import struct
import hashlib
import tempfile
import warnings
import heapq
from pathlib import Path
from zipfile import BadZipFile
from dataclasses import dataclass
from itertools import accumulate
from bisect import bisect_right
from ctypes import (
    CDLL,
    POINTER,
    Structure,
    addressof,
    c_float,
    c_int,
    c_size_t,
    c_void_p,
    cast,
    memmove,
    sizeof
)

import numpy as np
from pufferlib.ocean.drive.pedestrian_routing import DEFAULT_TRAJECTORY_POINTS
from pufferlib.ocean.drive.obstacle_spawns import load_obstacle_positions
from pufferlib.ocean.drive.observation import HISTORY_CAPACITY
from pufferlib.ocean.drive.road_types import STOP_SIGN, TRAFFIC_LIGHT_STOP_LINE, RIGHT_HAND_STOP_LINE, TRAFFIC_LIGHT_YIELD_FLAG, stop_line_fallback_type

from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import LaneletMap, LineString3d, Point3d, BasicPoint2d, BasicPoint3d
from lanelet2.geometry import length, fromArcCoordinates, ArcCoordinates, to2D, interpolatedPointAtDistance, findWithin2d, toArcCoordinates, distance, inside
from lanelet2.routing import RoutingGraph
import lanelet2.traffic_rules

libc = CDLL(None)
libc.calloc.argtypes = (c_size_t, c_size_t)
libc.calloc.restype = c_void_p
libc.malloc.argtypes = (c_size_t,)
libc.malloc.restype = c_void_p

# MUST match the corresponding structs in drive.h!
class HiddenVehicle(Structure):
    _fields_ = [("entity_idx", c_int), ("until_timestep", c_int)]


class Entity(Structure):
    _fields_ = [
        ("type", c_int),
        ("array_size", c_int),
        ("traj_x", POINTER(c_float)),
        ("traj_y", POINTER(c_float)),
        ("traj_z", POINTER(c_float)),
        ("traj_vx", POINTER(c_float)),
        ("traj_vy", POINTER(c_float)),
        ("traj_vz", POINTER(c_float)),
        ("traj_heading", POINTER(c_float)),
        ("traj_valid", POINTER(c_int)),
        ("width", c_float),
        ("length", c_float),
        ("height", c_float),
        ("goal_position_x", c_float),
        ("goal_position_y", c_float),
        ("goal_position_z", c_float),
        ("init_goal_x", c_float),
        ("init_goal_y", c_float),
        ("multi_goal_count", c_int),
        ("multi_goal_idx", c_int),
        ("multi_goal_position_x", POINTER(c_float)),
        ("multi_goal_position_y", POINTER(c_float)),
        ("multi_goal_position_z", POINTER(c_float)),
        ("mark_as_expert", c_int),
        ("collision_state", c_int),
        ("collided_with_agent_idx", c_int),
        ("metrics_array", c_float * 6),
        ("prev_x", c_float),
        ("prev_y", c_float),
        ("x", c_float),
        ("y", c_float),
        ("z", c_float),
        ("vx", c_float),
        ("vy", c_float),
        ("vz", c_float),
        ("heading", c_float),
        ("heading_x", c_float),
        ("heading_y", c_float),
        ("lane_heading_angle_diff", c_float),
        ("goal_timestep", c_int),
        ("current_lane_idx", c_int),
        ("valid", c_int),
        ("respawn_timestep", c_int),
        ("respawn_count", c_int),
        ("collided_before_goal", c_int),
        ("sampled_new_goal", c_int),
        ("reached_goal_this_episode", c_int),
        ("reached_final_goal", c_int),
        ("num_goals_reached", c_int),
        ("active_agent", c_int),
        ("cumulative_displacement", c_float),
        ("displacement_sample_count", c_int),
        ("goal_radius", c_float),
        ("max_speed", c_float),
        ("lane_center_reward_coefficient", c_float),
        ("comfort_reward_coefficient", c_float),
        ("lane_align_reward_coefficient", c_float),
        ("collision_reward_coefficient", c_float),
        ("safety_reward_coefficient", c_float),
        ("offroad_reward_coefficient", c_float),
        ("stop_line_reward_coefficient", c_float),
        ("speed_limit_coefficient", c_float),
        ("lane_center_bias", c_float),
        ("object_detection_distance", c_float),
        ("hide_other_vehicles_fraction", c_float),
        ("friction_coefficient", c_float),
        ("steering_gain", c_float),
        ("acceleration_gain", c_float),
        # Jerk dynamics
        ("a_long", c_float),
        ("a_lat", c_float),
        ("jerk_long", c_float),
        ("jerk_lat", c_float),
        ("steering_angle", c_float),
        ("wheelbase", c_float),
        ("low_steering_rate", c_float),
        ("high_steering_rate", c_float),
        ("low_steering_rate_error", c_float),
        ("high_steering_rate_error", c_float),
        # Private actuator state; must match ACTION_QUEUE_CAPACITY in drive.h.
        ("action_delay_steps", c_int),
        ("action_queue_head", c_int),
        ("action_queue", (c_float * 2) * 4),
        ("ego_history_head", c_int),
        ("ego_history", (c_float * 7) * HISTORY_CAPACITY),
        ("hidden_vehicles", HiddenVehicle * 3),
        ("hidden_vehicles_update_step", c_int),
    ]

class StaticRoadEntity(Structure):
    _fields_ = [
        ("type", c_int),
        ("array_size", c_int),
        ("traj_x", POINTER(c_float)),
        ("traj_y", POINTER(c_float)),
        ("traj_z", POINTER(c_float)),
        ("traffic_light_idx", c_int),
        ("traffic_light_off_type", c_int),
        ("speed_limit", c_float),  # m/s
    ]

utm_origin_lat = 58.385345
utm_origin_lon = 26.726272
projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), True, False)
MAP_PATH = Path("tartu_large_processed.osm")
MAPGEN_CACHE_PATH = MAP_PATH.parent / ".cache" / "mapgen" / f"{MAP_PATH.name}.npz"
OBSTACLE_CACHE_PATH = MAPGEN_CACHE_PATH.with_name(f"{MAP_PATH.name}.obstacles.npz")
OBSTACLE_POOL_SIZE = 131072
OBSTACLE_MAX_ROAD_DISTANCE = 10.0
OBSTACLE_MAX_AREA = 0.3  # Footprint in square metres, after native float32 conversion.
PEDESTRIAN_MAP_PATH = MAP_PATH.with_name("pedestrian.osm")
PEDESTRIAN_CACHE_PATH = MAPGEN_CACHE_PATH.with_name(f"{PEDESTRIAN_MAP_PATH.name}.npz")
PEDESTRIAN_SAMPLE_RATE = 0.1
PEDESTRIAN_TRAJECTORY_POINTS = DEFAULT_TRAJECTORY_POINTS
PEDESTRIAN_POOL_SIZE = 262144
PEDESTRIAN_POOL_PATH = MAPGEN_CACHE_PATH.parent / "pedestrian_trajectories"
SCOOTER_FRACTION = 0.1
# Width, length, and rider-inclusive height ranges, in metres.
PEDESTRIAN_DIMENSIONS = ((0.36, 0.6), (0.2, 0.6), (1.4, 2.2))
SCOOTER_DIMENSIONS = ((0.5, 0.7), (1.2, 1.6), (1.7, 2.3))
# Bump when the precomputation algorithms or cache layout change.
MAPGEN_CACHE_VERSION = 2
lanelet2_map = load(str(MAP_PATH), projector)
map_lanelets = list(lanelet2_map.laneletLayer)
map_lanelet_cumulative_lengths = tuple(accumulate(length(lanelet.centerline) for lanelet in map_lanelets))

SPAWN_VEHICLE_WIDTH = 2.75
SPAWN_VEHICLE_LENGTH = 15.0
# These are additions to the total dimensions, not to each side.
SPAWN_SAFETY_WIDTH_MARGIN = 0.1
SPAWN_SAFETY_LENGTH_MARGIN = 2.0
SPAWN_SAFETY_SIZE = (SPAWN_VEHICLE_WIDTH + SPAWN_SAFETY_WIDTH_MARGIN,
                     SPAWN_VEHICLE_LENGTH + SPAWN_SAFETY_LENGTH_MARGIN)
SPAWN_SEARCH_STEP = 1.0
SPAWN_GRID_CELL_SIZE = 20.0
SPAWN_BORDER_CLEARANCE = 0.20  # Minimum distance from each vehicle edge, in metres.
ROAD_BORDER_TYPES = {"road_border", "curbstone", "guard_rail"}
PARKED_MIN_OVERHANG = 0.20
PARKED_MAX_BORDER_GAP = 0.50
PARKED_BORDER_DEVIATION = 0.10  # Maximum bend across a full parking position.

VEHICLE_PROFILES = [
    (0.30, 65, (1.90, 4.82, 1.46)),  # sedans
    (0.20, 45, (2.50, 5.00, 1.69)),  # Lexus 450h
    (0.12, 55, (2.00, 5.02, 1.78)),  # SUVs
    (0.05, 55, (2.03, 5.88, 1.97)),  # pickup trucks
    (0.07, 55, (1.95, 5.40, 1.97)),  # vans
    (0.07, 45, (2.55, 10.00, 2.30)), # buses
    (0.03, 55, (2.55, 8.00, 3.60)),  # trucks
    (0.10, 20, (0.60, 1.80, 1.10)),  # bicycles
    (0.03, 80, (0.80, 2.20, 1.25)),  # motorcycles
    (0.03, 20, (2.30, 6.40, 3.00)),  # street maintenance vehicle
]

vehicle_weights = [w for w, _, _ in VEHICLE_PROFILES]

# Parked slots contain vehicles; pedestrians have their own fixed population.
STATIONARY_OBJECT_PROFILES = list(VEHICLE_PROFILES)
stationary_object_weights = [w for w, _, _ in STATIONARY_OBJECT_PROFILES]

# TODO: Different traffic rules based on vehicle, e.g. different
# rules for buses and bicycles
traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, lanelet2.traffic_rules.Participants.VehicleTaxi)

routing_graph = RoutingGraph(lanelet2_map, traffic_rules)

def iter_linestring_points_with_spacing(linestring):
    max_linestring_point_spacing = 25.0

    if not linestring:
        return

    prev = linestring[0].basicPoint()
    yield prev

    for i in range(1, len(linestring)):
        point = linestring[i].basicPoint()
        segment_length = distance(prev, point)

        if segment_length > max_linestring_point_spacing:
            num_segments = math.ceil(segment_length / max_linestring_point_spacing)
            for segment_idx in range(1, num_segments):
                frac = segment_idx / num_segments
                yield prev + (point - prev) * frac

        yield point
        prev = point


def linear_interpolate(p1, p2, speed, sample_rate=0.1):
    tick_distance = sample_rate * speed
    if type(p1) != BasicPoint2d:
        p1 = p1.basicPoint()
    if type(p2) != BasicPoint2d:
        p2 = p2.basicPoint()
    delta = p2 - p1
    dist = distance(p1, p2)
    if dist < 0.1:
        return []
    v = delta * (1 / dist) * tick_distance
    
    n = int(dist // tick_distance)
    pos = p1
    points = [p1]
    for _ in range(n):
        pos += v
        points.append(pos)

    return points


def calculate_normal(linestring, i):
    if i + 1 < len(linestring):
        p1 = linestring[i]
        p2 = linestring[i + 1]
    else:
        p1 = linestring[i - 1]
        p2 = linestring[i]
    normal_angle = math.atan2(p2.y - p1.y, p2.x - p1.x) + math.pi/2
    return BasicPoint2d(math.cos(normal_angle), math.sin(normal_angle))


def sample_points_along_line(linestring, start_offset, end_offset, speed, sample_rate=0.1, rng=random):
    lateral_shift = rng.uniform(-1.5, 1.5)
    
    points = []

    i = 0
    offset = 0
    while i + 1 < len(linestring) and offset + distance(linestring[i], linestring[i + 1]) < start_offset:
        offset += distance(linestring[i], linestring[i + 1])
        i += 1

    # start point is between i'th and i + 1'th point
    # interpolate between them
    frac = (start_offset - offset) / distance(linestring[i], linestring[i + 1])
    
    avg_distance_per_tick = sample_rate * speed

    current_pos = linestring[i].basicPoint() + frac * (linestring[i + 1].basicPoint() - linestring[i].basicPoint())
    points.append(current_pos + lateral_shift * calculate_normal(linestring, i))
    offset = start_offset

    while i + 1 < len(linestring) and offset < end_offset:
        distance_per_tick = avg_distance_per_tick * rng.uniform(0.8, 1.2)
        if distance(current_pos, linestring[i + 1]) < distance_per_tick:
            distance_traveled = distance(current_pos, linestring[i + 1])
            i += 1

            while i + 1 < len(linestring) and distance_traveled + distance(linestring[i], linestring[i + 1]) < distance_per_tick:
                distance_traveled += distance(linestring[i], linestring[i + 1])
                i += 1
    
            if i + 1 >= len(linestring):
                current_pos = linestring[-1].basicPoint()
            else:
                frac = (distance_per_tick - distance_traveled) / distance(linestring[i], linestring[i + 1])
                current_pos = linestring[i].basicPoint() + frac * (linestring[i + 1].basicPoint() - linestring[i].basicPoint())
        else:
            frac = distance_per_tick / distance(current_pos, linestring[i + 1])
            current_pos += frac * (linestring[i + 1].basicPoint() - current_pos)
        
        offset += distance_per_tick
        points.append(current_pos + lateral_shift * calculate_normal(linestring, i))

    return points

def crossing_route_lanelets(lanelet):
    """Identify the complete crossing corridor, including connected lanelets."""
    start_pos = BasicPoint2d(lanelet.centerline[0].x, lanelet.centerline[0].y)
    end_pos = BasicPoint2d(lanelet.centerline[-1].x, lanelet.centerline[-1].y)

    walked_lanelets = [lanelet]
    while len(walked_lanelets) < 3:
        preceding_lanelets = [
            l
            for _, l in findWithin2d(lanelet2_map.laneletLayer, start_pos, 10.0)
            if "subtype" in l.attributes and l.attributes["subtype"] == "crosswalk" and l not in walked_lanelets
        ]
        if not preceding_lanelets:
            break

        walked_lanelets.insert(0, preceding_lanelets[0])
        start_pos = BasicPoint2d(preceding_lanelets[0].centerline[0].x, preceding_lanelets[0].centerline[0].y)
    
    while len(walked_lanelets) < 6:
        following_lanelets = [
            l
            for _, l in findWithin2d(lanelet2_map.laneletLayer, end_pos, 10.0)
            if "subtype" in l.attributes and l.attributes["subtype"] == "crosswalk" and l not in walked_lanelets 
        ]
        if not following_lanelets:
            break

        walked_lanelets.append(following_lanelets[0])
        end_pos = BasicPoint2d(following_lanelets[0].centerline[-1].x, following_lanelets[0].centerline[-1].y)

    return walked_lanelets


def build_pedestrian_crossing_routes(lanelet, rng=random):
    """Build the approach/crossing/exit variants for one crosswalk at startup."""
    walked_lanelets = crossing_route_lanelets(lanelet)
    start_pos = BasicPoint2d(walked_lanelets[0].centerline[0].x, walked_lanelets[0].centerline[0].y)
    end_pos = BasicPoint2d(walked_lanelets[-1].centerline[-1].x, walked_lanelets[-1].centerline[-1].y)

    origins = findWithin2d(lanelet2_map.lineStringLayer, start_pos, 3.0)
    origins = [origin for _, origin in origins if "type" in origin.attributes and origin.attributes["type"] in {"curbstone", "road_border", "guard_rail"}]
    if origins:
        origin_road_edge = to2D(origins[0])
        connector = toArcCoordinates(origin_road_edge, start_pos)

        origin_option1 = sample_points_along_line(origin_road_edge, max(connector.length - 20, 0), connector.length, speed=1.4, rng=rng)
        # plt.plot([p.x for p in origin_option1], [p.y for p in origin_option1], "og", markersize=1)
        origin_option2 = list(reversed(sample_points_along_line(origin_road_edge, connector.length, connector.length + 20, speed=1.4, rng=rng)))
        # plt.plot([p.x for p in origin_option2], [p.y for p in origin_option2], "o", color="darkgreen", markersize=1)
        origin_choices = [origin_option1, origin_option2]
    else:
        origin_choices = [[start_pos]]

    destinations = findWithin2d(lanelet2_map.lineStringLayer, end_pos, 3.0)
    destinations = [dest for _, dest in destinations if "type" in dest.attributes and dest.attributes["type"] in {"curbstone", "road_border", "guard_rail"}]

    if destinations:
        dest_road_edge = to2D(destinations[0])
        connector = toArcCoordinates(dest_road_edge, end_pos)

        dest_option1 = sample_points_along_line(dest_road_edge, connector.length, connector.length + 20, speed=1.4, rng=rng)
        # plt.plot([p.x for p in dest_option1], [p.y for p in dest_option1], "ob", markersize=1)
        dest_option2 = list(reversed(sample_points_along_line(dest_road_edge, max(connector.length - 20, 0), connector.length, speed=1.4, rng=rng)))
        # plt.plot([p.x for p in dest_option2], [p.y for p in dest_option2], "oc", markersize=1)

        destination_choices = [dest_option1, dest_option2]
    else:
        destination_choices = [[end_pos]]

    crosswalk = []
    prev = None
    for l in walked_lanelets:
        if prev is not None:
            crosswalk.extend(linear_interpolate(to2D(prev.centerline[-1]), to2D(l.centerline[0]), speed=1.4))
        crosswalk.extend(sample_points_along_line(to2D(l.centerline), 0, math.inf, speed=1.4, rng=rng))
        prev = l
    # plt.plot([p.x for p in crosswalk], [p.y for p in crosswalk], "or", markersize=1)

    routes = []
    for origin in origin_choices:
        for destination in destination_choices:
            routes.append(tuple(origin + linear_interpolate(origin[-1], crosswalk[0], speed=1.4) + crosswalk +
                                linear_interpolate(crosswalk[-1], destination[0], speed=1.4) + destination))

    # plt.savefig("x.png", dpi=500)

    return tuple(routes)


def precompute_pedestrian_crossing_routes(lanelets):
    """Cache every crosswalk without consuming the episode random generator."""
    rng = random.Random(0)
    return {
        lanelet.id: build_pedestrian_crossing_routes(lanelet, rng=rng)
        for lanelet in sorted(lanelets, key=lambda lanelet: lanelet.id)
        if "subtype" in lanelet.attributes and lanelet.attributes["subtype"] == "crosswalk"
    }


def get_pedestrian_crossing_route(lanelet=None):
    # Sample crosswalks uniformly, then choose one of that crossing's variants.
    routes = (random.choice(map_pedestrian_crossing_routes) if lanelet is None
              else pedestrian_crossing_routes_cache[lanelet.id])
    return random.choice(routes)

def alloc_entities(count):
    buf = libc.calloc(c_size_t(count), c_size_t(sizeof(Entity)))
    if not buf:
        raise MemoryError("failed to allocate Entity*")
    return cast(buf, POINTER(Entity))

def alloc_static_road_entities(count):
    buf = libc.calloc(c_size_t(count), c_size_t(sizeof(StaticRoadEntity)))
    if not buf:
        raise MemoryError("failed to allocate StaticRoadEntity*")
    return cast(buf, POINTER(StaticRoadEntity))

def alloc_float_array(values):
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise ValueError("Native float arrays must be one-dimensional")
    length = len(values)
    buf = libc.malloc(c_size_t(length * sizeof(c_float)))
    if not buf:
        raise MemoryError("malloc failed for float array")
    if isinstance(values, np.ndarray):
        tmp = np.ascontiguousarray(values, dtype=np.float32)
        memmove(buf, tmp.ctypes.data, tmp.nbytes)
    else:
        tmp = (c_float * length)(*values)
        memmove(buf, tmp, sizeof(tmp))
    return cast(buf, POINTER(c_float))

def alloc_int_array(values):
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise ValueError("Native int arrays must be one-dimensional")
    length = len(values)
    buf = libc.malloc(c_size_t(length * sizeof(c_int)))
    if not buf:
        raise MemoryError("malloc failed for int array")
    if isinstance(values, np.ndarray):
        tmp = np.ascontiguousarray(values, dtype=np.int32)
        memmove(buf, tmp.ctypes.data, tmp.nbytes)
    else:
        tmp = (c_int * length)(*values)
        memmove(buf, tmp, sizeof(tmp))
    return cast(buf, POINTER(c_int))


def pack_float_array(ptr, length):
    if length <= 0:
        return b""
    if not ptr:
        return struct.pack(f"{length}f", *([0.0] * length))
    arr = (c_float * length).from_address(cast(ptr, c_void_p).value)
    return struct.pack(f"{length}f", *arr)


def pack_int_array(ptr, length):
    if length <= 0:
        return b""
    if not ptr:
        return struct.pack(f"{length}i", *([0] * length))
    arr = (c_int * length).from_address(cast(ptr, c_void_p).value)
    return struct.pack(f"{length}i", *arr)


def negate(arr):
    return [-value for value in arr]


def rectangle_corners(x, y, heading, width, length):
    cos_heading = math.cos(heading)
    sin_heading = math.sin(heading)
    half_width = width / 2
    half_length = length / 2
    return [
        (
            x + length_sign * half_length * cos_heading - width_sign * half_width * sin_heading,
            y + length_sign * half_length * sin_heading + width_sign * half_width * cos_heading,
        )
        for length_sign, width_sign in ((1, 1), (1, -1), (-1, -1), (-1, 1))
    ]


def projections_overlap(corners_a, corners_b, axis):
    axis_x, axis_y = axis
    projections_a = [x * axis_x + y * axis_y for x, y in corners_a]
    projections_b = [x * axis_x + y * axis_y for x, y in corners_b]
    return max(projections_a) >= min(projections_b) and max(projections_b) >= min(projections_a)


def rectangles_collide(pose_a, size_a, pose_b, size_b):
    x_a, y_a, heading_a = pose_a
    width_a, length_a = size_a
    x_b, y_b, heading_b = pose_b
    width_b, length_b = size_b
    corners_a = rectangle_corners(x_a, y_a, heading_a, width_a, length_a)
    corners_b = rectangle_corners(x_b, y_b, heading_b, width_b, length_b)
    axes = [
        (math.cos(heading_a), math.sin(heading_a)),
        (-math.sin(heading_a), math.cos(heading_a)),
        (math.cos(heading_b), math.sin(heading_b)),
        (-math.sin(heading_b), math.cos(heading_b)),
    ]
    return all(projections_overlap(corners_a, corners_b, axis) for axis in axes)


def vehicle_safety_size(dimensions):
    return (dimensions[0] + SPAWN_SAFETY_WIDTH_MARGIN,
            dimensions[1] + SPAWN_SAFETY_LENGTH_MARGIN)


def vehicle_collides(pose, dimensions, placed_vehicles):
    """Check both vehicles' safety boxes; inputs contain their actual dimensions."""
    size = vehicle_safety_size(dimensions)
    if isinstance(placed_vehicles, VehicleSpawnIndex):
        index = placed_vehicles
        placed_vehicles = (index.vehicles[i] for i in index.nearby_indices(pose, size))
    return any(rectangles_collide(pose, size, other_pose, vehicle_safety_size(other_size))
               for other_pose, other_size in placed_vehicles)


def sample_dimensions(base_dimensions, variation):
    return tuple(dimension * (1 + random.uniform(-variation, variation)) for dimension in base_dimensions)


def spatial_cells(points):
    """Grid cells covered by the bounding box of a set of XY tuples."""
    min_x = math.floor(min(p[0] for p in points) / SPAWN_GRID_CELL_SIZE)
    max_x = math.floor(max(p[0] for p in points) / SPAWN_GRID_CELL_SIZE)
    min_y = math.floor(min(p[1] for p in points) / SPAWN_GRID_CELL_SIZE)
    max_y = math.floor(max(p[1] for p in points) / SPAWN_GRID_CELL_SIZE)
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            yield x, y


class VehicleSpawnIndex:
    """Index placed safety boxes so generation only compares nearby actors."""

    def __init__(self):
        self.vehicles = []
        self.grid = {}

    def append(self, vehicle):
        pose, size = vehicle
        index = len(self.vehicles)
        self.vehicles.append((pose, size[:2]))
        for cell in spatial_cells(rectangle_corners(*pose, *vehicle_safety_size(size))):
            self.grid.setdefault(cell, []).append(index)

    def nearby_indices(self, pose, size):
        return {index for cell in spatial_cells(rectangle_corners(*pose, *size))
                for index in self.grid.get(cell, ())}


def rectangle_intersects_segments(pose, size, segments):
    """Intersect closed segments with a rectangle, including contained segments."""
    x, y, heading = pose
    width, vehicle_length = size
    cos_heading, sin_heading = math.cos(heading), math.sin(heading)
    for start, end in segments:
        # Transform into the vehicle frame, then clip the segment to its box.
        sx, sy = start[0] - x, start[1] - y
        ex, ey = end[0] - x, end[1] - y
        local_start = (sx * cos_heading + sy * sin_heading,
                       -sx * sin_heading + sy * cos_heading)
        local_end = (ex * cos_heading + ey * sin_heading,
                     -ex * sin_heading + ey * cos_heading)
        t_min, t_max = 0.0, 1.0
        for a, b, half_extent in zip(local_start, local_end, (vehicle_length / 2, width / 2)):
            delta = b - a
            if abs(delta) < 1e-12:
                if abs(a) > half_extent:
                    break
                continue
            t1, t2 = (-half_extent - a) / delta, (half_extent - a) / delta
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
            if t_min > t_max:
                break
        else:
            return True
    return False


class RoadBorderIndex:
    """Static spatial index of physical borders; lane markings are excluded."""

    def __init__(self, linestrings):
        self.segments = []
        self.segment_border_keys = []
        self.lines = {}
        self.grid = {}
        for line in linestrings:
            if "type" not in line.attributes or line.attributes["type"] not in ROAD_BORDER_TYPES:
                continue
            self.lines[line.id] = line
            points = [(p.x, p.y) for p in line]
            for local_index, segment in enumerate(zip(points, points[1:])):
                index = len(self.segments)
                self.segments.append(segment)
                self.segment_border_keys.append((line.id, local_index))
                for cell in spatial_cells(segment):
                    self.grid.setdefault(cell, []).append(index)

    def collides(self, pose, size, clearance=0.0, ignore_border_section=None):
        # Expand every side, including front/rear, before querying nearby cells.
        size = (size[0] + 2 * clearance, size[1] + 2 * clearance)
        corners = rectangle_corners(*pose, *size)
        indices = {index for cell in spatial_cells(corners) for index in self.grid.get(cell, ())}
        if ignore_border_section is not None:
            border_id, start, end = ignore_border_section
            indices = {i for i in indices if not (
                self.segment_border_keys[i][0] == border_id
                and start <= self.segment_border_keys[i][1] < end
            )}
        return rectangle_intersects_segments(pose, size, (self.segments[i] for i in indices))


@dataclass(frozen=True)
class SpawnPosition:
    lanelet: object
    offset: float
    # Map coordinates and forward road heading. Simulation XY is negated later.
    pose: tuple


@dataclass(frozen=True)
class ParkedSpawnPosition:
    border_id: int
    pose: tuple  # Center on the physical border, heading along its local tangent.
    outward_sign: int  # +1 means left of the tangent; -1 means right.
    border_min: float  # Border's outward-coordinate range across a 15 m footprint.
    border_max: float
    first_segment: int = 0
    end_segment: int = 1  # Exclusive; other parts of a looping border remain obstacles.


def precompute_parked_spawn_positions(lanelets, borders):
    """Sample physical borders, using road polygons only to identify the outside."""
    roads = LaneletMap()
    for lanelet in lanelets:
        if "subtype" in lanelet.attributes and lanelet.attributes["subtype"] == "road":
            roads.add(lanelet)
    positions = []
    half_length = SPAWN_VEHICLE_LENGTH / 2
    for border_id, border in sorted(borders.lines.items()):
        points = [(p.x, p.y) for p in border]
        if len(points) < 2:
            continue
        offsets = [0.0]
        for first, second in zip(points, points[1:]):
            offsets.append(offsets[-1] + math.dist(first, second))

        def point_at(s):
            index = min(bisect_right(offsets, s) - 1, len(points) - 2)
            span = offsets[index + 1] - offsets[index]
            fraction = (s - offsets[index]) / span if span else 0.0
            return tuple(a + fraction * (b - a) for a, b in zip(points[index], points[index + 1]))

        s = half_length
        while s <= offsets[-1] - half_length:
            start, end = point_at(s - half_length), point_at(s + half_length)
            x, y = point_at(s)
            heading = math.atan2(end[1] - start[1], end[0] - start[0])
            nx, ny = -math.sin(heading), math.cos(heading)
            window = [start, end, *points[bisect_right(offsets, s - half_length):
                                        bisect_right(offsets, s + half_length)]]
            lateral = [(px - x) * nx + (py - y) * ny for px, py in window]
            if max(lateral) - min(lateral) > PARKED_BORDER_DEVIATION:
                s += SPAWN_SEARCH_STEP
                continue

            # Neither the linestring direction nor a lane's travel direction
            # reliably identifies the land side. Require road on exactly one side
            # at both ends and the middle, rejecting internal/intersection lines.
            sides = []
            for px, py in (start, (x, y), end):
                nearby = [lane for _, lane in findWithin2d(roads.laneletLayer, BasicPoint2d(px, py), 1.0)]
                left = any(inside(lane, BasicPoint2d(px + nx, py + ny)) for lane in nearby)
                right = any(inside(lane, BasicPoint2d(px - nx, py - ny)) for lane in nearby)
                sides.append(-1 if left and not right else 1 if right and not left else 0)
            if sides[0] == 0 or len(set(sides)) != 1:
                s += SPAWN_SEARCH_STEP
                continue
            outward_sign = sides[0]
            outward = [value * outward_sign for value in lateral]
            positions.append(ParkedSpawnPosition(
                border_id, (x, y, heading), outward_sign, min(outward), max(outward),
                max(0, bisect_right(offsets, s - half_length) - 1),
                min(len(points) - 1, bisect_right(offsets, s + half_length)),
            ))
            s += SPAWN_SAFETY_SIZE[1] + 0.01
    return tuple(positions)


def sample_parked_spawn_pose(position, dimensions, placed_vehicles, borders):
    """Allow curb overhang or fully outside parking, with at most a 50 cm gap."""
    width, vehicle_length = dimensions[:2]
    if vehicle_length > SPAWN_VEHICLE_LENGTH:
        return None
    # A millimetre of slack protects the limits after native float32 conversion.
    low = position.border_max + PARKED_MIN_OVERHANG - width / 2 + 0.001
    high = position.border_min + PARKED_MAX_BORDER_GAP + width / 2 - 0.001
    if low > high:
        return None
    outward_offset = random.uniform(low, high)
    x, y, heading = position.pose
    dx = -math.sin(heading) * position.outward_sign * outward_offset
    dy = math.cos(heading) * position.outward_sign * outward_offset
    pose = (x + dx, y + dy, heading)
    # Permit crossing the selected curb, but do not cross or jump over another
    # physical border (for example, the opposite edge of a narrow traffic island).
    sweep_pose = (x + dx / 2, y + dy / 2, heading)
    sweep_size = (width + abs(outward_offset), vehicle_length)
    if (borders.collides(sweep_pose, sweep_size, ignore_border_section=(
            position.border_id, position.first_segment, position.end_segment,
        ))
            or vehicle_collides(pose, dimensions, placed_vehicles)):
        return None
    return pose


def rectangle_fits_lanelet(lanelet, pose, size, boundary_segments):
    if not all(inside(lanelet, BasicPoint2d(x, y)) for x, y in rectangle_corners(*pose, *size)):
        return False
    # Corner containment alone misses a curved/concave border cutting into the
    # footprint. Touching a lane marking is fine; entering the box is not.
    interior_size = (size[0] - 1e-7, size[1] - 1e-7)
    return not rectangle_intersects_segments(pose, interior_size, boundary_segments)


def precompute_spawn_positions(lanelets, borders):
    """Pack non-overlapping 17 x 2.85 m safety boxes across all straight lanes.

    Scan at one-metre resolution where a candidate does not fit. The accepted
    centerline poses are deterministic and shared by every generated episode.
    """
    positions = []
    occupied_cells = {}
    vehicle_size = (SPAWN_VEHICLE_WIDTH, SPAWN_VEHICLE_LENGTH)
    for lanelet in sorted(lanelets, key=lambda lanelet: lanelet.id):
        attributes = lanelet.attributes
        if ("subtype" not in attributes or attributes["subtype"] != "road"
                or "turn_direction" not in attributes or attributes["turn_direction"] != "straight"
                or len(lanelet.centerline) < 2):
            continue
        line = to2D(lanelet.centerline)
        line_length = length(line)
        boundary = [(p.x, p.y) for p in lanelet.polygon2d()]
        boundary_segments = list(zip(boundary, boundary[1:] + boundary[:1]))
        s = SPAWN_VEHICLE_LENGTH / 2
        while s <= line_length - SPAWN_VEHICLE_LENGTH / 2:
            pos = interpolatedPointAtDistance(line, s)
            near = interpolatedPointAtDistance(line, s + 0.05)
            pose = (pos.x, pos.y, math.atan2(near.y - pos.y, near.x - pos.x))
            cells = tuple(spatial_cells(rectangle_corners(*pose, *SPAWN_SAFETY_SIZE)))
            neighbors = {index for cell in cells for index in occupied_cells.get(cell, ())}
            if (any(rectangles_collide(pose, SPAWN_SAFETY_SIZE, positions[i].pose, SPAWN_SAFETY_SIZE)
                    for i in neighbors)
                    or not rectangle_fits_lanelet(lanelet, pose, vehicle_size, boundary_segments)
                    or borders.collides(pose, vehicle_size, clearance=SPAWN_BORDER_CLEARANCE)):
                s += SPAWN_SEARCH_STEP
                continue
            index = len(positions)
            positions.append(SpawnPosition(lanelet, s, pose))
            for cell in cells:
                occupied_cells.setdefault(cell, []).append(index)
            # A small gap also keeps touching safety boxes from colliding.
            s += SPAWN_SAFETY_SIZE[1] + 0.01
    return tuple(positions)


def sample_spawn_pose(position, dimensions, placed_vehicles, borders, lateral_range=(-1.5, 1.5)):
    """Keep 20 cm to physical borders and disjoint vehicle safety boxes after offsets."""
    x, y, heading = position.pose
    normal_x, normal_y = -math.sin(heading), math.cos(heading)
    # Keep the centerline as a fallback when a sampled offset is obstructed.
    offsets = [random.uniform(*lateral_range) for _ in range(12)] + [0.0]
    for offset in offsets:
        pose = (x + normal_x * offset, y + normal_y * offset, heading)
        # Check the entire lateral sweep so a large offset cannot jump a curb.
        swept_pose = (x + normal_x * offset / 2, y + normal_y * offset / 2, heading)
        swept_size = (dimensions[0] + abs(offset), dimensions[1])
        if (not borders.collides(swept_pose, swept_size, clearance=SPAWN_BORDER_CLEARANCE)
                and not vehicle_collides(pose, dimensions, placed_vehicles)):
            return pose
    return None


def mapgen_cache_metadata(map_path):
    digest = hashlib.sha256()
    with open(map_path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "version": MAPGEN_CACHE_VERSION,
        "map_sha256": digest.hexdigest(),
        "origin": [utm_origin_lat, utm_origin_lon],
        "spawn_size": [SPAWN_VEHICLE_WIDTH, SPAWN_VEHICLE_LENGTH],
        "safety_size": list(SPAWN_SAFETY_SIZE),
        "safety_margin": [SPAWN_SAFETY_WIDTH_MARGIN, SPAWN_SAFETY_LENGTH_MARGIN],
        "search_step": SPAWN_SEARCH_STEP,
        "border_clearance": SPAWN_BORDER_CLEARANCE,
        "border_types": sorted(ROAD_BORDER_TYPES),
        "parked_min_overhang": PARKED_MIN_OVERHANG,
        "parked_max_border_gap": PARKED_MAX_BORDER_GAP,
        "parked_border_deviation": PARKED_BORDER_DEVIATION,
    }


def read_mapgen_cache(cache_path, metadata, lanelets, borders):
    """Restore numeric cache data against the freshly loaded lanelet map."""
    with np.load(cache_path, allow_pickle=False) as data:
        if json.loads(data["metadata"].item()) != metadata:
            raise ValueError("Mapgen cache is out of date")
        spawn_ids = data["spawn_ids"]
        spawn_offsets = data["spawn_offsets"]
        spawn_poses = data["spawn_poses"]
        parked_border_ids = data["parked_border_ids"]
        parked_data = data["parked_data"]
        parked_segments = data["parked_segments"]
        crosswalk_ids = data["crosswalk_ids"]
        crosswalk_offsets = data["crosswalk_offsets"]
        route_offsets = data["route_offsets"]
        point_indices = data["point_indices"]
        point_xy = data["point_xy"]

    integer_arrays = (spawn_ids, parked_border_ids, crosswalk_ids, crosswalk_offsets, route_offsets, point_indices)
    if any(a.ndim != 1 or a.dtype.kind not in "iu" for a in integer_arrays):
        raise ValueError("Invalid mapgen cache indices")
    if (spawn_offsets.shape != spawn_ids.shape or spawn_poses.shape != (len(spawn_ids), 3)
            or point_xy.ndim != 2 or point_xy.shape[1] != 2
            or any(a.dtype.kind != "f" or not np.isfinite(a).all()
                   for a in (spawn_offsets, spawn_poses, point_xy))
            or np.any(spawn_offsets < 0)):
        raise ValueError("Invalid mapgen cache coordinates")
    if (parked_data.shape != (len(parked_border_ids), 6) or parked_data.dtype.kind != "f"
            or not np.isfinite(parked_data).all()
            or not set(parked_border_ids).issubset(borders.lines)
            or not np.isin(parked_data[:, 3], [-1, 1]).all()
            or np.any(parked_data[:, 4] > parked_data[:, 5])):
        raise ValueError("Invalid mapgen parking cache")
    if (parked_segments.shape != (len(parked_border_ids), 2) or parked_segments.dtype.kind not in "iu"
            or np.any(parked_segments[:, 0] < 0)
            or np.any(parked_segments[:, 1] <= parked_segments[:, 0])
            or any(end >= len(borders.lines[int(border_id)]) for border_id, (_, end)
                   in zip(parked_border_ids, parked_segments))):
        raise ValueError("Invalid mapgen parking border sections")
    if (crosswalk_offsets.shape != (len(crosswalk_ids) + 1,) or len(route_offsets) == 0
            or crosswalk_offsets[0] != 0 or route_offsets[0] != 0
            or crosswalk_offsets[-1] != len(route_offsets) - 1
            or route_offsets[-1] != len(point_indices)
            or np.any(crosswalk_offsets[1:] <= crosswalk_offsets[:-1])
            or np.any(route_offsets[1:] < route_offsets[:-1] + 2)
            or np.any(point_indices < 0) or np.any(point_indices >= len(point_xy))):
        raise ValueError("Invalid mapgen cache route offsets")
    expected_crosswalks = {lanelet.id for lanelet in lanelets
                          if "subtype" in lanelet.attributes and lanelet.attributes["subtype"] == "crosswalk"}
    if len(set(crosswalk_ids)) != len(crosswalk_ids) or set(crosswalk_ids) != expected_crosswalks:
        raise ValueError("Incomplete mapgen crossing cache")

    lanelets_by_id = {lanelet.id: lanelet for lanelet in lanelets}
    positions = tuple(SpawnPosition(lanelets_by_id[int(lane_id)], float(offset), tuple(map(float, pose)))
                      for lane_id, offset, pose in zip(spawn_ids, spawn_offsets, spawn_poses))
    # Share points across route variants, just as the in-memory builder does.
    points = [BasicPoint2d(float(x), float(y)) for x, y in point_xy]
    routes = [tuple(points[index] for index in point_indices[start:end])
              for start, end in zip(route_offsets[:-1], route_offsets[1:])]
    crossings = {int(lane_id): tuple(routes[start:end]) for lane_id, start, end in
                 zip(crosswalk_ids, crosswalk_offsets[:-1], crosswalk_offsets[1:])}
    parked = tuple(ParkedSpawnPosition(int(border_id), tuple(map(float, row[:3])), int(row[3]),
                                       float(row[4]), float(row[5]), int(section[0]), int(section[1]))
                   for border_id, row, section in zip(parked_border_ids, parked_data, parked_segments))
    return positions, crossings, parked


def write_mapgen_cache(cache_path, metadata, positions, crossings, parked):
    """Atomically publish portable arrays, with no pickled Python objects."""
    point_ids = {}
    point_xy = []
    point_indices = []
    route_offsets = [0]
    crosswalk_offsets = [0]
    crosswalk_ids = sorted(crossings)
    for lane_id in crosswalk_ids:
        for route in crossings[lane_id]:
            for point in route:
                index = point_ids.get(id(point))
                if index is None:
                    index = len(point_xy)
                    point_ids[id(point)] = index
                    point_xy.append((point.x, point.y))
                point_indices.append(index)
            route_offsets.append(len(point_indices))
        crosswalk_offsets.append(len(route_offsets) - 1)

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, prefix=cache_path.name + ".",
                                         suffix=".tmp", delete=False) as out:
            temporary_path = Path(out.name)
            np.savez(
                out, metadata=json.dumps(metadata, sort_keys=True),
                spawn_ids=np.asarray([p.lanelet.id for p in positions], dtype=np.int64),
                spawn_offsets=np.asarray([p.offset for p in positions], dtype=np.float64),
                spawn_poses=np.asarray([p.pose for p in positions], dtype=np.float64).reshape(-1, 3),
                parked_border_ids=np.asarray([p.border_id for p in parked], dtype=np.int64),
                parked_data=np.asarray([(*p.pose, p.outward_sign, p.border_min, p.border_max)
                                        for p in parked], dtype=np.float64).reshape(-1, 6),
                parked_segments=np.asarray([(p.first_segment, p.end_segment) for p in parked],
                                           dtype=np.int64).reshape(-1, 2),
                crosswalk_ids=np.asarray(crosswalk_ids, dtype=np.int64),
                crosswalk_offsets=np.asarray(crosswalk_offsets, dtype=np.int64),
                route_offsets=np.asarray(route_offsets, dtype=np.int64),
                point_indices=np.asarray(point_indices, dtype=np.int64),
                point_xy=np.asarray(point_xy, dtype=np.float64).reshape(-1, 2),
            )
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_or_build_mapgen_cache(map_path, cache_path, lanelets, borders):
    """Reuse preprocessing across runs until map contents or cache settings change."""
    metadata = mapgen_cache_metadata(map_path)
    try:
        return read_mapgen_cache(cache_path, metadata, lanelets, borders)
    except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile):
        pass  # Missing, stale or incomplete caches are disposable.

    positions = precompute_spawn_positions(lanelets, borders)
    crossings = precompute_pedestrian_crossing_routes(lanelets)
    parked = precompute_parked_spawn_positions(lanelets, borders)
    try:
        write_mapgen_cache(cache_path, metadata, positions, crossings, parked)
    except OSError as error:
        warnings.warn(f"Could not save mapgen cache {cache_path}: {error}", RuntimeWarning)
    return positions, crossings, parked


# Load the persistent cache at startup; only a cache miss runs precomputation.
road_border_index = RoadBorderIndex(lanelet2_map.lineStringLayer)
map_spawn_positions, pedestrian_crossing_routes_cache, map_parked_spawn_positions = load_or_build_mapgen_cache(
    MAP_PATH, MAPGEN_CACHE_PATH, map_lanelets, road_border_index,
)
map_pedestrian_crossing_routes = tuple(pedestrian_crossing_routes_cache.values())
map_obstacle_positions = load_obstacle_positions(
    MAP_PATH, (utm_origin_lat, utm_origin_lon), OBSTACLE_CACHE_PATH, map_lanelets,
    count=OBSTACLE_POOL_SIZE, max_distance=OBSTACLE_MAX_ROAD_DISTANCE,
)

PEDESTRIANS_PER_ROUTE = 4
_pedestrian_network = None
_pedestrian_trajectory_pool = None


def get_pedestrian_network():
    global _pedestrian_network
    if _pedestrian_network is None:
        from pufferlib.ocean.drive.pedestrian_routing import load_pedestrian_network
        _pedestrian_network = load_pedestrian_network(
            PEDESTRIAN_MAP_PATH, projector, (utm_origin_lat, utm_origin_lon),
            PEDESTRIAN_CACHE_PATH,
        )
    return _pedestrian_network


def get_pedestrian_trajectory_pool():
    global _pedestrian_trajectory_pool
    if _pedestrian_trajectory_pool is None:
        from pufferlib.ocean.drive.pedestrian_trajectories import load_trajectory_pool
        _pedestrian_trajectory_pool = load_trajectory_pool(
            PEDESTRIAN_MAP_PATH, (utm_origin_lat, utm_origin_lon),
            PEDESTRIAN_POOL_PATH, get_pedestrian_network,
            size=PEDESTRIAN_POOL_SIZE, max_points=PEDESTRIAN_TRAJECTORY_POINTS,
            sample_rate=PEDESTRIAN_SAMPLE_RATE, scooter_fraction=SCOOTER_FRACTION,
        )
    return _pedestrian_trajectory_pool


def sample_limited_routes(routes, count, weights=None):
    """Sample route slots without replacement, allowing small groups of up to four."""
    if count > len(routes) * PEDESTRIANS_PER_ROUTE:
        raise RuntimeError(f"Requested {count} pedestrian routes, but only "
                           f"{len(routes) * PEDESTRIANS_PER_ROUTE} route slots are available")
    if weights is None:
        slots = random.sample(range(len(routes) * PEDESTRIANS_PER_ROUTE), count)
        return [routes[slot // PEDESTRIANS_PER_ROUTE] for slot in slots]
    # Exponential priorities give weighted sampling without replacement. Each
    # corridor has a fixed number of slots, regardless of its length or lane count.
    priorities = ((-math.log1p(-random.random()) / max(weight, 1e-6), i, slot)
                  for i, weight in enumerate(weights) for slot in range(PEDESTRIANS_PER_ROUTE))
    return [routes[i] for _, i, _ in heapq.nsmallest(count, priorities)]


def pedestrian_route_plans(count):
    """Sample starts across physical pedestrian corridors, then choose a direction."""
    network = get_pedestrian_network()
    corridors = sample_limited_routes(network.corridors, count, network.corridor_lengths)
    return [random.choice(corridor) for corridor in corridors]


def build_pedestrian_route(start_lane, start_offset):
    network = get_pedestrian_network()
    goal_lane, goal_offset = network.sample_position()
    points = network.trajectory(start_lane, start_offset, goal_lane, goal_offset,
                                speed=random.uniform(0.8, 1.8),
                                sample_rate=PEDESTRIAN_SAMPLE_RATE,
                                max_points=PEDESTRIAN_TRAJECTORY_POINTS)
    return points, network.point(goal_lane, goal_offset)


def pedestrian_motion(points):
    """Native coordinates negate map x/y; velocities are metres per second."""
    from pufferlib.ocean.drive.pedestrian_routing import pedestrian_motion as motion
    return motion(points, PEDESTRIAN_SAMPLE_RATE)


def generate_pedestrians(count, placed_vehicles):
    """Sample distinct cached replays, checking footprints and corridor capacity."""
    if count == 0:
        return []
    pool = get_pedestrian_trajectory_pool()
    scooters = round(count * pool.metadata['scooter_fraction'])
    remaining = [count - scooters, scooters]
    pedestrians = []
    corridor_counts = {}
    for index in random.sample(range(pool.size), pool.size):
        kind = int(pool.kinds[index])
        corridor = int(pool.corridors[index])
        if not remaining[kind] or corridor_counts.get(corridor, 0) >= PEDESTRIANS_PER_ROUTE:
            continue
        sequence = pool.trajectory(index)
        dimensions = tuple(random.uniform(*bounds) for bounds in
                           (SCOOTER_DIMENSIONS if kind else PEDESTRIAN_DIMENSIONS))
        pose = (-float(sequence[0, 0]), -float(sequence[1, 0]), float(sequence[4, 0]) - math.pi)
        if vehicle_collides(pose, dimensions, placed_vehicles):
            continue
        placed_vehicles.append((pose, dimensions[:2]))
        pedestrians.append((sequence, dimensions, pool.goals[index]))
        corridor_counts[corridor] = corridor_counts.get(corridor, 0) + 1
        remaining[kind] -= 1
        if len(pedestrians) == count:
            return pedestrians
    raise RuntimeError(f"Only found {len(pedestrians)} of {count} collision-free pedestrian/scooter starts")


def sample_obstacle_dimensions():
    while True:
        width, obstacle_length = map(float, np.float32([
            random.uniform(0.2, 0.6), random.uniform(0.2, 0.6)]))
        if width * obstacle_length <= OBSTACLE_MAX_AREA:
            return width, obstacle_length, random.uniform(0.05, 0.8)


def generate_obstacles(count, placed_vehicles):
    """Sample cached points without replacement, rejecting occupied footprints."""
    if count < 0 or count > len(map_obstacle_positions):
        raise ValueError("Requested obstacles exceed the cached spawn pool")
    obstacles = []
    remaining = len(map_obstacle_positions)
    swaps = {}
    for _ in range(count):
        dimensions = sample_obstacle_dimensions()
        while remaining:
            # Lazy Fisher-Yates avoids shuffling all 128K points for 1K obstacles.
            choice = random.randrange(remaining)
            index = swaps.get(choice, choice)
            remaining -= 1
            swaps[choice] = swaps.get(remaining, remaining)
            swaps.pop(remaining, None)
            x, y = map_obstacle_positions[index]
            pose = (float(x), float(y), random.uniform(-math.pi, math.pi))
            if (road_border_index.collides(pose, dimensions[:2])
                    or vehicle_collides(pose, dimensions, placed_vehicles)):
                continue
            placed_vehicles.append((pose, dimensions[:2]))
            obstacles.append((pose, dimensions))
            break
        else:
            raise RuntimeError(f"Only found {len(obstacles)} of {count} collision-free obstacle spawns")
    return obstacles


def lanelet_speed_limit(lanelet):
    """Return the lanelet's speed limit in m/s, defaulting to 50 kph."""
    speed_limit_kph = float(lanelet.attributes["speed_limit"]) if "speed_limit" in lanelet.attributes else 50.0
    return speed_limit_kph / 3.6


def generate_shared_static_road_entities():
    def add_linestring_segments_in_box(entity_type, linestring, traffic_light_idx=-1, speed_limit=0.0):
        x = []
        y = []
        z = []
        for p in iter_linestring_points_with_spacing(linestring):
            x.append(p.x)
            y.append(p.y)
            z.append(p.z)
        if x:
            off_type = stop_line_fallback_type(linestring.attributes) if entity_type == TRAFFIC_LIGHT_STOP_LINE else 0
            roads.append((entity_type, x, y, z, traffic_light_idx, off_type, speed_limit))

    roads = []
    for lanelet in lanelet2_map.laneletLayer:
        if "subtype" not in lanelet.attributes:
            continue
        if lanelet.attributes["subtype"] == "road":
            add_linestring_segments_in_box(4, lanelet.centerline, speed_limit=lanelet_speed_limit(lanelet))
        elif lanelet.attributes["subtype"] == "crosswalk":
            add_linestring_segments_in_box(8, lanelet.leftBound)
            add_linestring_segments_in_box(8, lanelet.rightBound)

    seen_traffic_light_lines = set()
    traffic_light_counter = 0
    for reg_el in lanelet2_map.regulatoryElementLayer:
        if reg_el.attributes["subtype"] == "traffic_light":
            traffic_light_line = reg_el.parameters["ref_line"][0]
            if "subtype" in traffic_light_line.attributes and traffic_light_line.attributes["subtype"] == "yield_right":
                continue
            if traffic_light_line.id not in seen_traffic_light_lines:
                seen_traffic_light_lines.add(traffic_light_line.id)
                add_linestring_segments_in_box(9, traffic_light_line, traffic_light_counter)
                traffic_light_counter += 1

    for linestring in lanelet2_map.lineStringLayer:
        if "type" in linestring.attributes and linestring.attributes["type"] in ROAD_BORDER_TYPES:
            add_linestring_segments_in_box(6, linestring)
        if ("type" in linestring.attributes and linestring.attributes["type"] == "stop_line"
            and "subtype" in linestring.attributes and linestring.attributes["subtype"] == "yield_right"):
            add_linestring_segments_in_box(RIGHT_HAND_STOP_LINE, linestring)
        if ("subtype" in linestring.attributes and linestring.attributes["subtype"] in {"yield_manual", "yield", "yield_stop"}
            and linestring.id not in seen_traffic_light_lines):
            add_linestring_segments_in_box(7, linestring)

    entities = alloc_static_road_entities(len(roads))

    for i, (entity_type, x, y, z, traffic_light_idx, off_type, speed_limit) in enumerate(roads):
        ent = entities[i]
        ent.type = entity_type
        ent.array_size = len(x)
        ent.traj_x = alloc_float_array(negate(x))
        ent.traj_y = alloc_float_array(negate(y))
        ent.traj_z = alloc_float_array(z)
        ent.traffic_light_idx = traffic_light_idx
        ent.traffic_light_off_type = off_type
        ent.speed_limit = speed_limit

    return entities, len(roads)

# Keep one owning capsule alive across environment resampling. The binding
# retains a separate C reference for each environment, so either side can be
# closed first without freeing geometry still in use.
_shared_static_road_map = None


def get_shared_static_road_map():
    global _shared_static_road_map
    if _shared_static_road_map is None:
        from pufferlib.ocean.drive import binding

        roads, count = generate_shared_static_road_entities()
        address = cast(roads, c_void_p).value or 0
        owner = binding.create_static_road_map(address, count)
        _shared_static_road_map = (count, address, owner)
    return _shared_static_road_map

MAX_GENERATED_ACTORS = 16384  # Must match MAX_AGENTS in drive.h.
agents_per_generated_map = int(os.getenv("AGENTS_PER_MAP", "5120"))
stationary_objects_per_generated_map = 2048
pedestrians_per_generated_map = 8192
obstacles_per_generated_map = 1024
goals_per_agent = int(os.getenv("GOALS_PER_AGENT", "8"))

def generate_map():
    # Initialize shared geometry before allocating this episode's actors.
    shared_num_roads, shared_roads_address, shared_road_map = get_shared_static_road_map()
    num_agents = agents_per_generated_map
    num_stationary_objects = stationary_objects_per_generated_map
    num_walking_pedestrians = pedestrians_per_generated_map
    num_obstacles = obstacles_per_generated_map
    if num_agents + num_stationary_objects + num_walking_pedestrians + num_obstacles > MAX_GENERATED_ACTORS:
        raise ValueError(f"Generated population exceeds native limit of {MAX_GENERATED_ACTORS} actors")

    if len(map_spawn_positions) < num_agents:
        raise RuntimeError(f"Requested {num_agents} vehicles, but the map has only {len(map_spawn_positions)} valid spawn positions")
    # Moving vehicles and parked vehicles have independent lane/border pools.
    spawn_position_queue = random.sample(map_spawn_positions, len(map_spawn_positions))
    parked_position_queue = random.sample(map_parked_spawn_positions, len(map_parked_spawn_positions))

    vehicle_samples = random.choices(VEHICLE_PROFILES, weights=vehicle_weights, k=num_agents)
    vehicle_dimensions = [sample_dimensions(sample[2], 0.1) for sample in vehicle_samples]

    spawn_positions = []
    goal_positions = []
    spawn_headings = []
    spawn_vx = []
    spawn_vy = []
    placed_vehicles = VehicleSpawnIndex()
    i = 0
    while i < num_agents and spawn_position_queue:
        is_initially_parked = random.uniform(0, 1) < 0.05
        position = spawn_position_queue.pop()
        lanelet, s = position.lanelet, position.offset
        vehicle_pose = sample_spawn_pose(
            position, vehicle_dimensions[i], placed_vehicles, road_border_index,
            lateral_range=(-1.5, 0.0) if is_initially_parked else (-1.5, 1.5),
        )
        if vehicle_pose is None:
            continue
        spawn_pos = BasicPoint2d(*vehicle_pose[:2])
        heading = vehicle_pose[2] + math.pi

        vehicle_speed_limit = vehicle_samples[i][1] / 3.6
        speed_limit = min(lanelet_speed_limit(lanelet), vehicle_speed_limit)

        if is_initially_parked:
            initial_speed_fraction = 0
        else:
            initial_speed_fraction = random.uniform(0.2, 1.0)
        initial_speed = speed_limit * initial_speed_fraction

        current_lanelet = lanelet
        current_lanelet_offset = s
        agent_goal_positions = []

        for _ in range(goals_per_agent):
            can_add_more_goals = True
            # 3-9 seconds away
            goal_distance = speed_limit * random.uniform(3, 9)
            while True:
                line = to2D(current_lanelet.centerline)
                following_lanelets = routing_graph.following(current_lanelet, withLaneChanges=True)
                path_remaining = length(line) - current_lanelet_offset

                if not following_lanelets and path_remaining <= goal_distance:
                    goal_pos = interpolatedPointAtDistance(line, current_lanelet_offset+path_remaining)
                    can_add_more_goals = False
                    goal_distance -= path_remaining
                    break
                elif not following_lanelets and path_remaining > goal_distance:
                    goal_pos = interpolatedPointAtDistance(line, current_lanelet_offset+goal_distance)
                    current_lanelet_offset += goal_distance
                    goal_distance = 0
                    break
                elif path_remaining > goal_distance:
                    goal_pos = interpolatedPointAtDistance(line, current_lanelet_offset+goal_distance)
                    current_lanelet_offset += goal_distance
                    goal_distance = 0
                    break
                else:
                    goal_distance -= path_remaining
                    current_lanelet = random.choice(following_lanelets)
                    current_lanelet_offset = 0

            agent_goal_positions.append(goal_pos)

            if not can_add_more_goals:
                break

        # Every requested policy vehicle must be eligible for native control.
        if distance(spawn_pos, agent_goal_positions[0]) < 2.1:
            continue
        spawn_positions.append(spawn_pos)
        spawn_headings.append(heading)
        spawn_vy.append(math.sin(heading)*initial_speed)
        spawn_vx.append(math.cos(heading)*initial_speed)
        goal_positions.append(agent_goal_positions)
        placed_vehicles.append((vehicle_pose, vehicle_dimensions[i][:2]))
        i += 1

    if i < num_agents:
        raise RuntimeError(f"Ran out of spawn positions after placing {i} of {num_agents} vehicles")

    candidate_stationary_samples = random.choices(
        STATIONARY_OBJECT_PROFILES,
        weights=stationary_object_weights,
        k=num_stationary_objects,
    )
    stationary_vehicle_dimensions = []
    stationary_spawn_positions = []
    stationary_spawn_headings = []
    for sample in candidate_stationary_samples:
        dimensions = sample_dimensions(sample[2], 0.2)
        vehicle_pose = None
        while parked_position_queue:
            position = parked_position_queue.pop()
            vehicle_pose = sample_parked_spawn_pose(position, dimensions, placed_vehicles, road_border_index)
            if vehicle_pose is not None:
                break
        if vehicle_pose is None:
            raise RuntimeError(f"Ran out of spawn positions after placing "
                               f"{len(stationary_spawn_positions)} of {num_stationary_objects} parked vehicles")
        spawn_pos = BasicPoint2d(*vehicle_pose[:2])
        heading = vehicle_pose[2] + math.pi

        stationary_vehicle_dimensions.append(dimensions)
        stationary_spawn_positions.append(spawn_pos)
        stationary_spawn_headings.append(heading)
        placed_vehicles.append((vehicle_pose, dimensions[:2]))

    num_stationary_objects = len(stationary_spawn_positions)
    pedestrians = generate_pedestrians(num_walking_pedestrians, placed_vehicles)
    obstacles = generate_obstacles(num_obstacles, placed_vehicles)

    num_objects = num_agents + num_stationary_objects + num_walking_pedestrians + num_obstacles
    assert num_objects <= MAX_GENERATED_ACTORS
    entities = alloc_entities(num_objects)

    for i in range(num_agents):
        _, max_speed, _ = vehicle_samples[i]
        width, vehicle_length, height = vehicle_dimensions[i]

        ent = entities[i]
        ent.type = 1
        ent.array_size = 1
        ent.traj_valid = alloc_int_array([1])
        ent.traj_x = alloc_float_array([-spawn_positions[i].x])
        ent.traj_y = alloc_float_array([-spawn_positions[i].y])
        ent.traj_z = alloc_float_array([0]) # TODO: Height dimension
        ent.traj_heading = alloc_float_array([spawn_headings[i]])
        ent.traj_vx = alloc_float_array([spawn_vx[i]])
        ent.traj_vy = alloc_float_array([spawn_vy[i]])
        ent.traj_vz = alloc_float_array([0])
        ent.width = width
        ent.length = vehicle_length
        ent.height = height
        ent.goal_position_x = -goal_positions[i][0].x
        ent.goal_position_y = -goal_positions[i][0].y
        ent.goal_position_z = 0 # TODO: Height dimension
        ent.multi_goal_count = len(goal_positions[i])
        ent.multi_goal_position_x = alloc_float_array([-pos.x for pos in goal_positions[i]])
        ent.multi_goal_position_y = alloc_float_array([-pos.y for pos in goal_positions[i]])
        ent.multi_goal_position_z = alloc_float_array([0 for _pos in goal_positions[i]]) # TODO: Height dimension
        ent.max_speed = max_speed * (1 + random.uniform(-0.1, 0.1)) / 3.6
        ent.lane_center_reward_coefficient = random.uniform(5e-5, 5e-4) if random.uniform(0, 1) < 0.5 else random.uniform(5e-4, 5e-3)
        ent.comfort_reward_coefficient = random.uniform(0, 0.01)
        ent.lane_align_reward_coefficient = random.uniform(5e-5, 5e-3) if random.uniform(0, 1) < 0.5 else random.uniform(5e-3, 5e-2)
        ent.collision_reward_coefficient = random.uniform(0.2, 1.0)
        ent.safety_reward_coefficient = random.uniform(0, 1.0)
        ent.offroad_reward_coefficient = random.uniform(0, 1.0)
        ent.stop_line_reward_coefficient = random.uniform(0, 1.0)
        ent.speed_limit_coefficient = random.uniform(0, 1e-3)
        ent.lane_center_bias = random.uniform(-0.3, 0.3)
        ent.goal_radius = random.uniform(0.5, 3)
        ent.object_detection_distance = random.uniform(40, 60)
        ent.hide_other_vehicles_fraction = random.uniform(0.01, 0.04)
        ent.friction_coefficient = random.uniform(0.7, 1.3)
        ent.acceleration_gain = random.uniform(0.9, 1.1)
        ent.steering_gain = random.uniform(0.9, 1.1)
        ent.wheelbase = random.uniform(0.55, 0.65) * ent.length
        ent.low_steering_rate = 0.018 * random.uniform(0.8, 1.2)
        ent.high_steering_rate = 0.033 * random.uniform(0.8, 1.2)
        ent.low_steering_rate_error = 0.01 * random.uniform(0.8, 1.2)
        ent.high_steering_rate_error = 0.1 * random.uniform(0.8, 1.2)
        ent.mark_as_expert = 0

    for i in range(num_stationary_objects):
        spawn_pos = stationary_spawn_positions[i]
        heading = stationary_spawn_headings[i]
        width, vehicle_length, height = stationary_vehicle_dimensions[i]

        ent = entities[i+num_agents]
        ent.type = 1
        ent.array_size = 1
        ent.traj_valid = alloc_int_array([1])
        ent.traj_x = alloc_float_array([-spawn_pos.x])
        ent.traj_y = alloc_float_array([-spawn_pos.y])
        ent.traj_z = alloc_float_array([0]) # TODO: Height dimension
        ent.traj_heading = alloc_float_array([heading])
        ent.traj_vx = alloc_float_array([0])
        ent.traj_vy = alloc_float_array([0])
        ent.traj_vz = alloc_float_array([0])
        ent.width = width
        ent.length = vehicle_length
        ent.height = height
        #ent.goal_position_x = ent.traj_x
        #ent.goal_position_y = ent.traj_y
        #ent.goal_position_z = ent.traj_z
        ent.mark_as_expert = 1

    for i, (sequence, dimensions, goal) in enumerate(pedestrians):
        ent = entities[i+num_agents+num_stationary_objects]
        ent.type = 1
        ent.array_size = sequence.shape[1]
        zeros = np.zeros(ent.array_size, dtype=np.float32)
        ent.traj_valid = alloc_int_array(np.ones(ent.array_size, dtype=np.int32))
        ent.traj_x = alloc_float_array(sequence[0])
        ent.traj_y = alloc_float_array(sequence[1])
        ent.traj_z = alloc_float_array(zeros)
        ent.traj_heading = alloc_float_array(sequence[4])
        ent.traj_vx = alloc_float_array(sequence[2])
        ent.traj_vy = alloc_float_array(sequence[3])
        ent.traj_vz = alloc_float_array(zeros)
        ent.width, ent.length, ent.height = dimensions
        ent.goal_position_x = -goal[0]
        ent.goal_position_y = -goal[1]
        ent.mark_as_expert = 1

    for i, (pose, dimensions) in enumerate(obstacles, num_agents + num_stationary_objects + num_walking_pedestrians):
        ent = entities[i]
        ent.type = 1  # Existing generic actor format keeps policy observations compatible.
        ent.array_size = 1
        ent.traj_valid = alloc_int_array([1])
        ent.traj_x = alloc_float_array([-pose[0]])
        ent.traj_y = alloc_float_array([-pose[1]])
        ent.traj_z = alloc_float_array([0])
        ent.traj_heading = alloc_float_array([pose[2] + math.pi])
        ent.traj_vx = alloc_float_array([0])
        ent.traj_vy = alloc_float_array([0])
        ent.traj_vz = alloc_float_array([0])
        ent.width, ent.length, ent.height = dimensions
        ent.goal_position_x = ent.traj_x[0]
        ent.goal_position_y = ent.traj_y[0]
        ent.mark_as_expert = 1

    # C takes ownership of actors; roads and their grid are owned by the capsule.
    return (num_objects, shared_num_roads, addressof(entities.contents),
            shared_roads_address, shared_road_map)


def save_map_pdf(map_tuple, path="generated_map.pdf"):
    """Draw the full map and initial actor poses as a zoomable vector PDF."""
    # Plotting is only needed for export; keep it out of training startup.
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    num_objects, num_roads, entities_addr = map_tuple[:3]
    entities = cast(c_void_p(entities_addr), POINTER(Entity))
    roads = cast(c_void_p(map_tuple[3]), POINTER(StaticRoadEntity)) if len(map_tuple) >= 4 else None

    road_styles = {
        4: ("Lane centerlines", "#9aafc0", 0.025),
        8: ("Crosswalks", "#b39ac9", 0.04),
        6: ("Physical road borders", "#454b52", 0.03),
    }
    road_lines = {kind: [] for kind in road_styles}
    for i in range(num_roads):
        road = roads[i] if roads is not None else entities[num_objects + i]
        if road.type in road_lines and road.array_size >= 2:
            x = np.ctypeslib.as_array(road.traj_x, shape=(road.array_size,))
            y = np.ctypeslib.as_array(road.traj_y, shape=(road.array_size,))
            # Undo the simulator's coordinate negation for a north-up map.
            road_lines[road.type].append(-np.column_stack((x, y)))

    vehicles = []
    stationary = []
    pedestrians = []
    scooters = []
    obstacles = []
    for i in range(num_objects):
        ent = entities[i]
        if ent.array_size < 1 or not ent.traj_valid[0]:
            continue
        x, y = -ent.traj_x[0], -ent.traj_y[0]
        # Pedestrians and scooters are expert actors with cached trajectories;
        # parked vehicles have a single stored pose, like controlled vehicles.
        if ent.mark_as_expert and ent.height < 1. and ent.width * ent.length <= OBSTACLE_MAX_AREA:
            obstacles.append((x, y))
        elif ent.mark_as_expert and ent.array_size > 1:
            (scooters if ent.length > 1. else pedestrians).append((x, y))
        else:
            footprints = stationary if ent.mark_as_expert else vehicles
            footprints.append(rectangle_corners(
                x, y, ent.traj_heading[0] - math.pi, ent.width, ent.length,
            ))

    figure = Figure(figsize=(20, 20), facecolor="white")
    ax = figure.add_axes((0.06, 0.06, 0.90, 0.84))
    legend = []
    for kind, (label, color, linewidth) in road_styles.items():
        ax.add_collection(LineCollection(
            road_lines[kind], colors=color, linewidths=linewidth, zorder=1,
        ))
        legend.append(Line2D([], [], color=color, linewidth=1, label=label))

    # This map is loaded only for PDF export, outside the generation hot path.
    pedestrian_color = "#808080"
    ax.add_collection(LineCollection(
        get_pedestrian_network().lines, colors=pedestrian_color,
        linewidths=0.15, alpha=0.25, zorder=1.5,
    ))
    legend.append(Line2D([], [], color=pedestrian_color, linewidth=1, alpha=0.25,
                         label="Pedestrian centerlines"))

    for footprints, color, label in (
        (vehicles, "#0072b2", "Controlled vehicles"),
        (stationary, "#d55e00", "Parked vehicles"),
    ):
        ax.add_collection(PolyCollection(
            footprints, facecolors=color, edgecolors="none", zorder=2,
        ))
        legend.append(Patch(facecolor=color, label=f"{label} ({len(footprints):,})"))
    for actors, color, marker, label in (
        (pedestrians, "#009e73", "o", "Pedestrians"),
        (scooters, "#cc79a7", "^", "Electric scooters"),
        (obstacles, "#666666", "s", "Small obstacles"),
    ):
        if actors:
            positions = np.asarray(actors)
            ax.scatter(positions[:, 0], positions[:, 1], s=0.15,
                       color=color, marker=marker, edgecolors="none", zorder=3)
        legend.append(Line2D([], [], marker=marker, linestyle="none", color=color,
                            markersize=5, label=f"{label} ({len(actors):,})"))

    ax.autoscale_view()
    ax.margins(0.01)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting from map origin (m)")
    ax.set_ylabel("Northing from map origin (m)")
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Generated map — initial actor positions", y=0.985, fontsize=18)
    figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.963),
                  ncol=3, frameon=False, fontsize=11)
    figure.text(0.5, 0.02,
                "Vehicle footprints are to scale; pedestrian, scooter, and obstacle markers are enlarged. "
                "Zoom in to inspect placement.",
                ha="center", fontsize=10, color="#555555")
    figure.savefig(path, format="pdf")


def save_map_binary(map_tuple, path):
    """Serialize split/shared maps using the existing object-then-road format.

    Legacy three-item tuples with a combined Entity array are also accepted.
    Exporting never mutates the source trajectories or shared road geometry.
    """
    num_objects, num_roads, entities_addr = map_tuple[:3]
    entities = cast(c_void_p(entities_addr), POINTER(Entity))
    roads = cast(c_void_p(map_tuple[3]), POINTER(StaticRoadEntity)) if len(map_tuple) >= 4 else None

    with open(path, "wb") as f:
        f.write(struct.pack("ii", num_objects, num_roads))
        for i in range(num_objects + num_roads):
            is_object = i < num_objects
            ent = entities[i] if is_object or roads is None else roads[i - num_objects]
            expand_static = is_object and ent.mark_as_expert and ent.array_size == 1
            size = PEDESTRIAN_TRAJECTORY_POINTS if expand_static else ent.array_size
            raw_type = ent.type | 256 if is_object else ent.type
            if ent.type == TRAFFIC_LIGHT_STOP_LINE and getattr(ent, "traffic_light_off_type", 0) == STOP_SIGN:
                raw_type |= TRAFFIC_LIGHT_YIELD_FLAG
            f.write(struct.pack("ii", raw_type, size))

            def write_float_array(values):
                if expand_static:
                    f.write(struct.pack("f", values[0]) * size)
                else:
                    f.write(pack_float_array(values, size))

            write_float_array(ent.traj_x)
            write_float_array(ent.traj_y)
            write_float_array(ent.traj_z)
            if is_object:
                write_float_array(ent.traj_vx)
                write_float_array(ent.traj_vy)
                write_float_array(ent.traj_vz)
                write_float_array(ent.traj_heading)
                if expand_static:
                    f.write(struct.pack("i", ent.traj_valid[0]) * size)
                else:
                    f.write(pack_int_array(ent.traj_valid, size))
                f.write(struct.pack("ffffffi", ent.width, ent.length, ent.height,
                                    ent.goal_position_x, ent.goal_position_y,
                                    ent.goal_position_z, ent.mark_as_expert))
                f.write(struct.pack("i", ent.multi_goal_count))
                f.write(pack_float_array(ent.multi_goal_position_x, ent.multi_goal_count))
                f.write(pack_float_array(ent.multi_goal_position_y, ent.multi_goal_count))
                f.write(pack_float_array(ent.multi_goal_position_z, ent.multi_goal_count))
            else:
                # Dimensions, goal position and expert flag are unused for roads.
                f.write(struct.pack("ffffffi", 0, 0, 0, 0, 0, 0, 0))

if __name__ == "__main__":
    t1 = time.time()
    map_data = generate_map()
    t2 = time.time()
    print(f"took {(t2-t1)*1000:.2f} ms")
    save_map_binary(map_data, "dynamic.bin")
    save_map_pdf(map_data, "generated_map.pdf")
    print("Saved dynamic.bin and generated_map.pdf")
