import os
import math
import time
import random
import json
import struct
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

from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import LaneletMap, LineString3d, Point3d, BasicPoint2d, BasicPoint3d, BoundingBox2d, GPSPoint
from lanelet2.geometry import length, fromArcCoordinates, ArcCoordinates, to2D, interpolatedPointAtDistance, findWithin2d, toArcCoordinates, distance
from lanelet2.routing import RoutingGraph
import lanelet2.traffic_rules

libc = CDLL(None)
libc.calloc.argtypes = (c_size_t, c_size_t)
libc.calloc.restype = c_void_p
libc.malloc.argtypes = (c_size_t,)
libc.malloc.restype = c_void_p

# MUST have same definition as the Entity struct in drive.h!
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
        ("offroad_reward_coefficient", c_float),
        ("lane_center_bias", c_float),
        ("hide_other_vehicles", c_int),
        # Jerk dynamics
        ("a_long", c_float),
        ("a_lat", c_float),
        ("jerk_long", c_float),
        ("jerk_lat", c_float),
        ("steering_angle", c_float),
        ("wheelbase", c_float),
    ]

utm_origin_lat = 58.385345
utm_origin_lon = 26.726272
projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), True, False)
lanelet2_map = load("tartu_large_processed.osm", projector)
map_points = list(lanelet2_map.pointLayer)

VEHICLE_PROFILES = [
    (0.30, 100, (1.90, 4.82, 1.46)),  # sedans
    (0.20, 100, (1.90, 4.89, 1.69)),  # Lexus 450h
    (0.12, 100, (2.00, 5.02, 1.78)),  # SUVs
    (0.05, 100, (2.03, 5.88, 1.97)),  # pickup trucks
    (0.07, 100, (1.95, 5.40, 1.97)),  # vans
    (0.07, 50, (2.55, 12.00, 2.30)), # buses
    (0.03, 60, (2.55, 8.00, 3.60)),  # trucks
    (0.10, 25, (0.60, 1.80, 1.10)),  # bicycles
    (0.03, 100, (0.80, 2.20, 1.25)),  # motorcycles
    (0.03, 20, (2.30, 6.40, 3.00)),  # street maintenance vehicle
]

vehicle_weights = [w for w, _, _ in VEHICLE_PROFILES]

STATIONARY_OBJECT_PROFILES = VEHICLE_PROFILES + [(1.00, 5, (0.40, 0.40, 1.80))]
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


def sample_points_along_line(linestring, start_offset, end_offset, speed, sample_rate=0.1):
    lateral_shift = random.uniform(-1.5, 1.5)
    
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
        distance_per_tick = avg_distance_per_tick * random.uniform(0.8, 1.2)
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

pedestrian_crossing_routes_cache = {}
def get_pedestrian_crossing_route(lanelet):
    # import matplotlib.pyplot as plt
    global pedestrian_crossing_routes_cache
    if lanelet.id in pedestrian_crossing_routes_cache:
        return random.choice(pedestrian_crossing_routes_cache[lanelet.id])
    
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

    origins = findWithin2d(lanelet2_map.lineStringLayer, start_pos, 3.0)
    origins = [origin for _, origin in origins if "type" in origin.attributes and origin.attributes["type"] in {"curbstone", "road_border", "guard_rail"}]
    if origins:
        origin_road_edge = to2D(origins[0])
        connector = toArcCoordinates(origin_road_edge, start_pos)

        origin_option1 = sample_points_along_line(origin_road_edge, max(connector.length - 20, 0), connector.length, speed=1.4)
        # plt.plot([p.x for p in origin_option1], [p.y for p in origin_option1], "og", markersize=1)
        origin_option2 = list(reversed(sample_points_along_line(origin_road_edge, connector.length, connector.length + 20, speed=1.4)))
        # plt.plot([p.x for p in origin_option2], [p.y for p in origin_option2], "o", color="darkgreen", markersize=1)
        origin_choices = [origin_option1, origin_option2]
    else:
        origin_choices = [[start_pos]]

    destinations = findWithin2d(lanelet2_map.lineStringLayer, end_pos, 3.0)
    destinations = [dest for _, dest in destinations if "type" in dest.attributes and dest.attributes["type"] in {"curbstone", "road_border", "guard_rail"}]

    if destinations:
        dest_road_edge = to2D(destinations[0])
        connector = toArcCoordinates(dest_road_edge, end_pos)

        dest_option1 = sample_points_along_line(dest_road_edge, connector.length, connector.length + 20, speed=1.4)
        # plt.plot([p.x for p in dest_option1], [p.y for p in dest_option1], "ob", markersize=1)
        dest_option2 = list(reversed(sample_points_along_line(dest_road_edge, max(connector.length - 20, 0), connector.length, speed=1.4)))
        # plt.plot([p.x for p in dest_option2], [p.y for p in dest_option2], "oc", markersize=1)

        destination_choices = [dest_option1, dest_option2]
    else:
        destination_choices = [[end_pos]]

    crosswalk = []
    prev = None
    for l in walked_lanelets:
        if prev is not None:
            crosswalk.extend(linear_interpolate(to2D(prev.centerline[-1]), to2D(l.centerline[0]), speed=1.4))
        crosswalk.extend(sample_points_along_line(to2D(l.centerline), 0, math.inf, speed=1.4))
        prev = l
    # plt.plot([p.x for p in crosswalk], [p.y for p in crosswalk], "or", markersize=1)

    pedestrian_crossing_routes_cache[lanelet.id] = []
    for origin in origin_choices:
        for destination in destination_choices:
            pedestrian_crossing_routes_cache[lanelet.id].append(origin + linear_interpolate(origin[-1], crosswalk[0], speed=1.4) + crosswalk +
                                                                linear_interpolate(crosswalk[-1], destination[0], speed=1.4) + destination)

    # plt.savefig("x.png", dpi=500)

    return random.choice(pedestrian_crossing_routes_cache[lanelet.id])

def alloc_entities(count):
    buf = libc.calloc(c_size_t(count), c_size_t(sizeof(Entity)))
    if not buf:
        raise MemoryError("failed to allocate Entity*")
    return cast(buf, POINTER(Entity))


def alloc_float_array(values):
    length = len(values)
    buf = libc.malloc(c_size_t(length * sizeof(c_float)))
    if not buf:
        raise MemoryError("malloc failed for float array")
    tmp = (c_float * length)(*values)
    memmove(buf, tmp, sizeof(tmp))
    return cast(buf, POINTER(c_float))

def alloc_int_array(values):
    length = len(values)
    buf = libc.malloc(c_size_t(length * sizeof(c_int)))
    if not buf:
        raise MemoryError("malloc failed for int array")
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


agents_per_generated_map = int(os.getenv("AGENTS_PER_MAP", "32"))
goals_per_agent = int(os.getenv("GOALS_PER_AGENT", "8"))

def generate_map():
    while True:
        center = random.choice(map_points)
        # center = BasicPoint2d(0.0, 0.0)
        if os.getenv("CENTER_LAT") is not None:
            lat = float(os.getenv("CENTER_LAT"))
            lon = float(os.getenv("CENTER_LON"))
            center = to2D(projector.forward(GPSPoint(lat, lon)))
        spawn_distance = 100

        spawn_box = BoundingBox2d(
            BasicPoint2d(center.x - spawn_distance, center.y - spawn_distance),
            BasicPoint2d(center.x + spawn_distance, center.y + spawn_distance)
        )

        spawn_box_lanelets = lanelet2_map.laneletLayer.search(spawn_box)

        allowed_spawn_lanelets = [
            lanelet
            for lanelet in spawn_box_lanelets
            if "subtype" in lanelet.attributes and lanelet.attributes["subtype"] == "road" and len(lanelet.centerline) >= 3
        ]

        allowed_spawn_lanelet_weights = [length(l.centerline) for l in allowed_spawn_lanelets]

        if allowed_spawn_lanelets:
            break

    num_agents = agents_per_generated_map
    num_stationary_objects = random.randint(16, 48)
    num_walking_pedestrians = random.randint(16, 48)

    used_spawn_positions = {}

    vehicle_samples = random.choices(VEHICLE_PROFILES, weights=vehicle_weights, k=num_agents)

    spawn_positions = []
    goal_positions = []
    spawn_headings = []
    spawn_vx = []
    spawn_vy = []
    i = 0
    while i < num_agents:
        is_initially_parked = random.uniform(0, 1) < 0.05
        
        lanelet, = random.choices(allowed_spawn_lanelets, weights=allowed_spawn_lanelet_weights, k=1)
        if is_initially_parked:
            while routing_graph.adjacentRight(lanelet) or routing_graph.right(lanelet):
                lanelet = routing_graph.adjacentRight(lanelet) or routing_graph.right(lanelet)

        if lanelet not in used_spawn_positions:
            used_spawn_positions[lanelet] = []

        line = to2D(lanelet.centerline)

        s = random.uniform(0.1, 0.9) * length(line)
        # If another vehicle is too close, make up to 5 
        # attempts to choose another spawn position
        vehicle_length = vehicle_samples[i][2][1]
        for _ in range(10):
            any_vehicle_near = False
            for other_offset, other_length in used_spawn_positions[lanelet]:
                if abs(s-other_offset) < (vehicle_length+other_length)/2 + 5:
                    any_vehicle_near = True
                    break
            if any_vehicle_near:
                s = random.uniform(0.1, 0.9) * length(line)
            else:
                break

        spawn_pos = fromArcCoordinates(line, ArcCoordinates(s, 0.0))

        near = interpolatedPointAtDistance(line, s+1)
        heading = math.atan2(near.y - spawn_pos.y, near.x - spawn_pos.x)+math.pi

        if is_initially_parked:
            spawn_pos.x += math.cos(heading+math.pi/2)
            spawn_pos.y += math.sin(heading+math.pi/2)

        speed_limit = 50
        if "speed_limit" in lanelet.attributes:
            speed_limit = float(lanelet.attributes["speed_limit"])

        vehicle_speed_limit = vehicle_samples[i][1]
        speed_limit = min(speed_limit, vehicle_speed_limit)

        if is_initially_parked:
            initial_speed_fraction = 0
        else:
            initial_speed_fraction = random.uniform(0.2, 1.0)
        initial_speed = speed_limit*initial_speed_fraction/3.6

        current_lanelet = lanelet
        current_lanelet_offset = s
        agent_goal_positions = []

        for _ in range(goals_per_agent):
            can_add_more_goals = True
            # 3-6 seconds away
            goal_distance = speed_limit / 3.6 * random.uniform(3, 6)
            while True:
                line = to2D(current_lanelet.centerline)
                following_lanelets = routing_graph.following(current_lanelet, withLaneChanges=True)
                path_remaining = length(line) - current_lanelet_offset

                if not following_lanelets and path_remaining <= goal_distance:
                    goal_pos = interpolatedPointAtDistance(line, current_lanelet_offset+path_remaining)
                    can_add_more_goals = False
                    break
                elif not following_lanelets and path_remaining > goal_distance:
                    goal_pos = interpolatedPointAtDistance(line, current_lanelet_offset+goal_distance)
                    current_lanelet_offset += goal_distance
                    break
                elif path_remaining > goal_distance:
                    goal_pos = interpolatedPointAtDistance(line, current_lanelet_offset+goal_distance)
                    current_lanelet_offset += goal_distance
                    break
                else:
                    goal_distance -= path_remaining
                    current_lanelet = random.choice(following_lanelets)
                    current_lanelet_offset = 0

            agent_goal_positions.append(goal_pos)

            if not can_add_more_goals:
                break

        if os.getenv("ONLY_WITHIN_BOUNDARY") == "1":
            all_in_boundary = True
            for gp in agent_goal_positions:
                if gp.x < spawn_box.min.x or gp.y < spawn_box.min.y or gp.x > spawn_box.max.x or gp.y > spawn_box.max.y:
                    all_in_boundary = False
                    break
            if not all_in_boundary:
                continue

        used_spawn_positions[lanelet].append((s, vehicle_length))
        spawn_positions.append(spawn_pos)
        spawn_headings.append(heading)
        spawn_vy.append(math.sin(heading)*initial_speed)
        spawn_vx.append(math.cos(heading)*initial_speed)
        goal_positions.append(agent_goal_positions)
        i += 1

    min_x = 1e9
    max_x = -1e9
    min_y = 1e9
    max_y = -1e9
    for point in spawn_positions + [goal for goals in goal_positions for goal in goals]:
        min_x = min(min_x, point.x)
        max_x = max(max_x, point.x)
        min_y = min(min_y, point.y)
        max_y = max(max_y, point.y)
    min_x -= 100
    min_y -= 100
    max_x += 100
    max_y += 100
    render_box = BoundingBox2d(BasicPoint2d(min_x, min_y), BasicPoint2d(max_x, max_y))

    def add_linestring_segments_in_box(entity_type, linestring):
        x = []
        y = []
        z = []
        for p in iter_linestring_points_with_spacing(linestring):
            if p.x < render_box.min.x or p.y < render_box.min.y or p.x > render_box.max.x or p.y > render_box.max.y:
                if x:
                    roads.append((entity_type, x, y, z))
                    x = []
                    y = []
                    z = []
            else:
                x.append(p.x)
                y.append(p.y)
                z.append(p.z)
        if x:
            roads.append((entity_type, x, y, z))

    roads = []
    render_box_lanelets = lanelet2_map.laneletLayer.search(render_box)
    crosswalk_lanelets = []
    for lanelet in render_box_lanelets:
        if "subtype" not in lanelet.attributes:
            continue
        if lanelet.attributes["subtype"] == "road":
            add_linestring_segments_in_box(4, lanelet.centerline)
        elif lanelet.attributes["subtype"] == "crosswalk":
            crosswalk_lanelets.append(lanelet)
            add_linestring_segments_in_box(8, lanelet.leftBound)
            add_linestring_segments_in_box(8, lanelet.rightBound)

    for linestring in lanelet2_map.lineStringLayer.search(render_box):
        if "subtype" not in linestring.attributes:
            continue
        if linestring.attributes["subtype"] == "yield_manual":
            add_linestring_segments_in_box(7, linestring)
    
    for reg_el in lanelet2_map.regulatoryElementLayer.search(render_box):
        if reg_el.attributes["subtype"] == "traffic_light":
            add_linestring_segments_in_box(9, reg_el.parameters["ref_line"][0])

    for linestring in lanelet2_map.lineStringLayer.search(render_box):
        if "type" in linestring.attributes and linestring.attributes["type"] in {"road_border", "curbstone", "guard_rail"}:
            add_linestring_segments_in_box(6, linestring)

    num_objects = num_agents + num_stationary_objects + num_walking_pedestrians
    assert num_objects <= 128
    entities = alloc_entities(len(roads) + num_objects)

    for i in range(num_agents):
        _, max_speed, obj_dims = vehicle_samples[i]

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
        ent.width = obj_dims[0] * (1 + random.uniform(-0.1, 0.1))
        ent.length = obj_dims[1] * (1 + random.uniform(-0.1, 0.1))
        ent.height = obj_dims[2] * (1 + random.uniform(-0.1, 0.1))
        ent.goal_position_x = -goal_positions[i][0].x
        ent.goal_position_y = -goal_positions[i][0].y
        ent.goal_position_z = 0 # TODO: Height dimension
        ent.multi_goal_count = len(goal_positions[i])
        ent.multi_goal_position_x = alloc_float_array([-pos.x for pos in goal_positions[i]])
        ent.multi_goal_position_y = alloc_float_array([-pos.y for pos in goal_positions[i]])
        ent.multi_goal_position_z = alloc_float_array([0 for _pos in goal_positions[i]]) # TODO: Height dimension
        ent.max_speed = max_speed * (1 + random.uniform(-0.1, 0.1)) / 3.6
        ent.lane_center_reward_coefficient = random.uniform(2.5e-5, 2.5e-4)
        ent.comfort_reward_coefficient = random.uniform(0, 0.01)
        ent.lane_align_reward_coefficient = random.uniform(2.5e-5, 2.5e-3)
        ent.collision_reward_coefficient = random.uniform(0, 1.0)
        ent.offroad_reward_coefficient = random.uniform(0, 1.0)
        ent.lane_center_bias = random.uniform(-0.3, 0.3)
        ent.goal_radius = random.uniform(1, 3)
        ent.hide_other_vehicles = int(random.uniform(0, 1) < 0.1)
        ent.mark_as_expert = 0

        allowed_spawn_lanelets = [
            lanelet
            for lanelet in spawn_box_lanelets
            if "subtype" in lanelet.attributes and lanelet.attributes["subtype"] == "road" and len(lanelet.centerline) >= 3
        ]

    spawn_lanelets = random.choices(render_box_lanelets, weights=[length(l.centerline) for l in render_box_lanelets], k=num_stationary_objects)
    vehicle_samples = random.choices(STATIONARY_OBJECT_PROFILES, weights=stationary_object_weights, k=num_stationary_objects)

    for i in range(num_stationary_objects):
        lanelet = spawn_lanelets[i]
        while routing_graph.adjacentRight(lanelet) or routing_graph.right(lanelet):
            lanelet = routing_graph.adjacentRight(lanelet) or routing_graph.right(lanelet)
        line = to2D(lanelet.rightBound)
        s = random.uniform(0.1, 0.9) * length(line)
        _, _, obj_dims = vehicle_samples[i]
        (_, vehicle_length, _) = obj_dims

        if lanelet in used_spawn_positions:
            for _ in range(5):
                any_vehicle_near = False
                for other_offset, other_length in used_spawn_positions[lanelet]:
                    if abs(s-other_offset) < (vehicle_length+other_length)/2 + 5:
                        any_vehicle_near = True
                        break
                if any_vehicle_near:
                    s = random.uniform(0.1, 0.9) * length(line)
                else:
                    break
        
        spawn_pos = fromArcCoordinates(line, ArcCoordinates(s, 0.0))

        near = interpolatedPointAtDistance(line, s+1)
        road_heading = math.atan2(near.y - spawn_pos.y, near.x - spawn_pos.x)

        road_normal = BasicPoint2d(math.cos(road_heading+math.pi/2), math.sin(road_heading+math.pi/2))
        spawn_pos += road_normal * random.uniform(-4, 0.25)
        
        heading = road_heading + random.choice([math.pi, 0]) + random.uniform(-0.5, 0.5)

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
        ent.width = obj_dims[0] * (1 + random.uniform(-0.2, 0.2))
        ent.length = obj_dims[1] * (1 + random.uniform(-0.2, 0.2))
        ent.height = obj_dims[2] * (1 + random.uniform(-0.2, 0.2))
        #ent.goal_position_x = ent.traj_x
        #ent.goal_position_y = ent.traj_y
        #ent.goal_position_z = ent.traj_z
        ent.mark_as_expert = 1
    

    spawn_lanelets = random.choices(render_box_lanelets, weights=[length(l.centerline) for l in render_box_lanelets], k=num_walking_pedestrians)

    for i in range(num_walking_pedestrians):
        speed_multiplier = random.choice([1, 1, 1, 1, 2, 2, 3, 4])

        if random.uniform(0, 1) < 0.4:
            lanelet = spawn_lanelets[i]
            while routing_graph.adjacentRight(lanelet) or routing_graph.right(lanelet):
                lanelet = routing_graph.adjacentRight(lanelet) or routing_graph.right(lanelet)
            line = to2D(lanelet.rightBound)
            start_offset = random.uniform(0, 0.5) * length(line)
            end_offset = min(start_offset + 200, length(line))
            s = start_offset
            distance_from_road = random.uniform(3, 8)
            points = [fromArcCoordinates(line, ArcCoordinates(s, -distance_from_road))]
            speed = random.uniform(0.5, 1.4)
            while s <= end_offset:
                segment_start = fromArcCoordinates(line, ArcCoordinates(s, -distance_from_road))
                s += random.uniform(1, 4)
                segment_end = fromArcCoordinates(line, ArcCoordinates(s, -distance_from_road))
                points.extend(linear_interpolate(segment_start, segment_end, speed=speed))
        elif not crosswalk_lanelets or random.uniform(0, 1) < 0.3:
            lanelet = spawn_lanelets[i]
            while routing_graph.adjacentRight(lanelet) or routing_graph.right(lanelet):
                lanelet = routing_graph.adjacentRight(lanelet) or routing_graph.right(lanelet)
            line = to2D(lanelet.rightBound)
            s = random.uniform(0.1, 0.9) * length(line)
            
            spawn_pos = fromArcCoordinates(line, ArcCoordinates(s, 0.0))
            near = interpolatedPointAtDistance(line, s+1)
            road_heading = math.atan2(near.y - spawn_pos.y, near.x - spawn_pos.x)
            road_normal = BasicPoint2d(math.cos(road_heading+math.pi/2), math.sin(road_heading+math.pi/2))

            points = linear_interpolate(spawn_pos - random.uniform(0, 2) * road_normal, spawn_pos + random.uniform(15, 20) * road_normal, speed=1.4)
        else:
            lanelet = random.choice(crosswalk_lanelets)
            points = get_pedestrian_crossing_route(lanelet)
        if speed_multiplier > 1:
            points = [points[j] for j in range(0, len(points), speed_multiplier)]
        if random.choice([True, False]):
            points = list(reversed(points))
        points = points[random.randint(0, len(points)//2):]

        heading = [0]
        vx = [0]
        vy = [0]
        prev = points[0]
        for point in points[1:]:
            dx = point.x - prev.x
            dy = point.y - prev.y
            heading.append(math.atan2(dy, dx)+math.pi)
            vx.append(dx)
            vy.append(dy)
            prev = point

        ent = entities[i+num_agents+num_stationary_objects]
        ent.type = 1
        ent.array_size = len(points)
        ent.traj_valid = alloc_int_array([1] * len(points))
        ent.traj_x = alloc_float_array([-p.x for p in points])
        ent.traj_y = alloc_float_array([-p.y for p in points])
        ent.traj_z = alloc_float_array([0 for p in points]) # TODO: Height dimension
        ent.traj_heading = alloc_float_array(heading)
        ent.traj_vx = alloc_float_array(vx)
        ent.traj_vy = alloc_float_array(vy)
        ent.traj_vz = alloc_float_array([0 for p in points])
        ent.width = random.uniform(0.36, 0.6)
        ent.length = random.uniform(0.2, 0.6)
        ent.height = random.uniform(1.4, 2.2)
        ent.mark_as_expert = 1

    for i, (entity_type, x, y, z) in enumerate(roads):
        ent = entities[i+num_objects]
        ent.type = entity_type
        ent.array_size = len(x)
        ent.traj_x = alloc_float_array(negate(x))
        ent.traj_y = alloc_float_array(negate(y))
        ent.traj_z = alloc_float_array(z)

    # Return raw pointer so C code owns the allocations.
    return (num_objects, len(roads), addressof(entities.contents))


def save_map_binary(map_tuple, path):
    num_objects, num_roads, entities_addr = map_tuple
    num_entities = num_objects + num_roads
    entities = cast(c_void_p(entities_addr), POINTER(Entity))

    with open(path, "wb") as f:
        f.write(struct.pack("i", num_objects))
        f.write(struct.pack("i", num_roads))

        for i in range(num_entities):
            ent = entities[i]
            size = ent.array_size

            if ent.mark_as_expert and size == 1:
                size = 600
                ent.traj_x = alloc_float_array([ent.traj_x[0]] * size)
                ent.traj_y = alloc_float_array([ent.traj_y[0]] * size)
                ent.traj_z = alloc_float_array([ent.traj_z[0]] * size)
                ent.traj_vx = alloc_float_array([ent.traj_vx[0]] * size)
                ent.traj_vy = alloc_float_array([ent.traj_vy[0]] * size)
                ent.traj_vz = alloc_float_array([ent.traj_vz[0]] * size)
                ent.traj_heading = alloc_float_array([ent.traj_heading[0]] * size)
                ent.traj_valid = alloc_int_array([ent.traj_valid[0]] * size)

            if ent.type in {1, 2, 3}:
                f.write(struct.pack("i", ent.type | 256))
            else:
                f.write(struct.pack("i", ent.type))
            f.write(struct.pack("i", size))

            f.write(pack_float_array(ent.traj_x, size))
            f.write(pack_float_array(ent.traj_y, size))
            f.write(pack_float_array(ent.traj_z, size))

            if ent.type in {1, 2, 3}:
                f.write(pack_float_array(ent.traj_vx, size))
                f.write(pack_float_array(ent.traj_vy, size))
                f.write(pack_float_array(ent.traj_vz, size))
                f.write(pack_float_array(ent.traj_heading, size))
                f.write(pack_int_array(ent.traj_valid, size))

            f.write(struct.pack("f", ent.width))
            f.write(struct.pack("f", ent.length))
            f.write(struct.pack("f", ent.height))
            f.write(struct.pack("f", ent.goal_position_x))
            f.write(struct.pack("f", ent.goal_position_y))
            f.write(struct.pack("f", ent.goal_position_z))
            f.write(struct.pack("i", ent.mark_as_expert))

            if ent.type in {1, 2, 3}:
                f.write(struct.pack("i", ent.multi_goal_count))
                f.write(pack_float_array(ent.multi_goal_position_x, ent.multi_goal_count))
                f.write(pack_float_array(ent.multi_goal_position_y, ent.multi_goal_count))
                f.write(pack_float_array(ent.multi_goal_position_z, ent.multi_goal_count))

if __name__ == "__main__":
    #random.seed(1)
    for _ in range(4):
        generate_map()
    t1 = time.time()
    map_data = generate_map()
    t2 = time.time()
    print(f"took {(t2-t1)*1000:.2f} ms")
    save_map_binary(map_data, "dynamic.bin")
