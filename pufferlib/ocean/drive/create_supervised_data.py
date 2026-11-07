import os
import sys
import json
import math
import itertools
from multiprocessing import Pool

import numpy as np
from pufferlib.ocean.drive.observation import ego_features
from pufferlib.ocean.drive.road_types import STOP_SIGN, RIGHT_HAND_STOP_LINE, stop_line_fallback_type

from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d, BoundingBox2d
from lanelet2.geometry import ArcCoordinates, fromArcCoordinates, toArcCoordinates, to2D, distance

utm_origin_lat = 58.385345
utm_origin_lon = 26.726272
projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), True, False)
lanelet2_map = None

def get_lanelet_map():
    global lanelet2_map
    if lanelet2_map is None:
        lanelet2_map = load("tartu_large_processed.osm", projector)
    return lanelet2_map

ACCELERATION_VALUES = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
STEERING_VALUES = [-0.667, -0.5, -0.333, -0.167, -0.08, -0.01, 0.0, 0.01, 0.08, 0.167, 0.333, 0.500, 0.667]

def to_egocentric_vector(x, y, ego_heading):
    rel_x = x * math.cos(ego_heading) + y * math.sin(ego_heading)
    rel_y = y * math.cos(ego_heading) - x * math.sin(ego_heading)
    return rel_x, rel_y

def to_egocentric_point(x, y, ego_x, ego_y, ego_heading):
    return to_egocentric_vector(x - ego_x, y - ego_y, ego_heading)

def iter_linestring(linestring):
    max_linestring_point_spacing = 25.0

    if len(linestring) < 2:
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


def normalize_heading(heading):
    # Make heading always be in range [0, 2*pi]
    while heading >= 2 * math.pi:
        heading -= 2 * math.pi
    while heading < 0:
        heading += 2 * math.pi
    return heading

def calculate_heading_change(a, b):
    return min(b - a, b - a - 2 * math.pi, b - a + 2 * math.pi, key=abs)


def write_history(data, t, x, y, heading, speed, policy_actions):
    """Write past states and normalized commands, newest first.

    Before the scenario begins, repeat the initial state and use neutral commands.
    policy_actions[s] is the command selected at s, before actuator delay.
    """
    past_steps = np.maximum(t - np.arange(1, 11), 0)
    cosine, sine = np.cos(heading[t]), np.sin(heading[t])
    dx, dy = x[past_steps] - x[t], y[past_steps] - y[t]
    positions = data[23:43].reshape(-1, 2)
    positions[:, 0] = (dx * cosine + dy * sine) * 0.02
    positions[:, 1] = (-dx * sine + dy * cosine) * 0.02
    delta = heading[past_steps] - heading[t]
    headings = data[43:63].reshape(-1, 2)
    headings[:, 0], headings[:, 1] = np.cos(delta), np.sin(delta)
    data[63:73] = speed[past_steps] / 27.7
    # Older speed samples repeat the initial speed if they predate the scenario.
    data[73:78] = speed[np.maximum(t - np.array([20, 30, 40, 50, 60]), 0)] / 27.7
    commands = data[78:98].reshape(10, 2)
    commands[:] = 0  # Normalized physical zero: no acceleration or steering.
    count = min(t, 10)
    commands[:count] = policy_actions[t-count:t][::-1]


def calculate_steering_rate(desired_steering, steering_angle):
    low_steering_rate = 3.0 / 16.135
    high_steering_rate = 5.5 / 16.135

    low_steering_rate_error = 0.2 / 16.135
    high_steering_rate_error = 1.5 / 16.135

    steering_error = abs(desired_steering - steering_angle)

    slope = (high_steering_rate - low_steering_rate) / (high_steering_rate_error - low_steering_rate_error)
    intercept = low_steering_rate - slope * low_steering_rate_error

    return min(high_steering_rate, max(low_steering_rate, steering_error * slope + intercept))


def process_scenario(filepath):
    with open(filepath) as file:
        scenario = json.load(file)
    print(f"processing {filepath}", file=sys.stderr)
    lanelet2_map = get_lanelet_map()
    hide_road_border = scenario.get("hide_road_border", False)
    hide_centerline = scenario.get("hide_centerline", False)

    trajectory = {key: np.asarray(scenario["ego"][key], dtype=np.float64)
                  for key in ("x", "y", "heading", "speed")}
    steps = len(trajectory["x"]) - 1
    sample_count = steps - 3
    if sample_count <= 0 or any(len(v) != steps + 1 for v in trajectory.values()):
        raise ValueError(f"{filepath}: need matching trajectories extending beyond 3 steps")
    ego_dim = ego_features()
    inputs = np.zeros((sample_count, ego_dim + 63 * 7 + 200 * 7), dtype=np.float32)
    expert_actions = np.zeros((steps, 2), dtype=np.int32)
    expert_actions[:, 0] = ACCELERATION_VALUES.index(0.0)
    expert_actions[:, 1] = STEERING_VALUES.index(0.0)
    waypoints = np.zeros((sample_count, 5, 2), dtype=np.float32)
    waypoint_mask = np.zeros((sample_count, 5), dtype=np.int8)

    prev_heading = scenario["ego"]["heading"][0]
    prev_speed = scenario["ego"]["speed"][0]
    steering_angles = []
    steering_angle = 0

    for t in range(steps):
        steering_angles.append(steering_angle)
        speed = scenario["ego"]["speed"][t]
        next_speed = scenario["ego"]["speed"][t + 1]
        next_heading = normalize_heading(scenario["ego"]["heading"][t + 1])

        desired_acceleration = (next_speed - prev_speed) / 0.1
        if next_speed < 0.01 and prev_speed <= 0.1:
            accel_action_idx = ACCELERATION_VALUES.index(-1.0)
        elif desired_acceleration > 0:
            accel_action_idx = max([i for i in range(len(ACCELERATION_VALUES)) if ACCELERATION_VALUES[i] >= 0 and desired_acceleration >= ACCELERATION_VALUES[i]], key=lambda x: ACCELERATION_VALUES[x])
        else:
            accel_action_idx = min([i for i in range(len(ACCELERATION_VALUES)) if ACCELERATION_VALUES[i] <= 0 and desired_acceleration <= ACCELERATION_VALUES[i]], key=lambda x: ACCELERATION_VALUES[x])
        expert_actions[t][0] = accel_action_idx
        prev_speed = max(0.1, prev_speed + ACCELERATION_VALUES[accel_action_idx] * 0.1)

        if speed > 0.1:
            wheelbase = 0.6 * scenario["ego"]["length"]
            desired_steering = math.atan(calculate_heading_change(prev_heading, next_heading) * wheelbase / speed / 0.1)
            if desired_steering > 0:
                steering_action_idx = max([i for i in range(len(STEERING_VALUES)) if STEERING_VALUES[i] >= 0 and desired_steering >= STEERING_VALUES[i]], key=lambda x: STEERING_VALUES[x])
            else:
                steering_action_idx = min([i for i in range(len(STEERING_VALUES)) if STEERING_VALUES[i] <= 0 and desired_steering <= STEERING_VALUES[i]], key=lambda x: STEERING_VALUES[x])
            expert_actions[t][1] = steering_action_idx
            max_steer_rate = calculate_steering_rate(STEERING_VALUES[steering_action_idx], steering_angle) * 0.1
            steering_angle += max(min(STEERING_VALUES[steering_action_idx] - steering_angle, max_steer_rate), -max_steer_rate)
            prev_heading = normalize_heading(prev_heading + speed * 0.1 * math.tan(steering_angle) / wheelbase)

    action_delay_steps = 2
    issued_actions = expert_actions[action_delay_steps:]
    policy_actions = np.column_stack((
        np.asarray(ACCELERATION_VALUES)[issued_actions[:, 0]] / 4.0,
        np.asarray(STEERING_VALUES)[issued_actions[:, 1]] / 0.667,
    ))

    for row in range(sample_count):
        t = row
        ego_x = scenario["ego"]["x"][t]
        ego_y = scenario["ego"]["y"][t]
        ego_heading = scenario["ego"]["heading"][t]
        speed = scenario["ego"]["speed"][t]

        rel_goal_x, rel_goal_y = to_egocentric_point(scenario["ego"]["goal"]["x"], scenario["ego"]["goal"]["y"],
                                                     ego_x, ego_y, ego_heading)
        rel_next_goal_x, rel_next_goal_y = to_egocentric_point(scenario["ego"]["next_goal"]["x"], scenario["ego"]["next_goal"]["y"],
                                                               ego_x, ego_y, ego_heading)

        for f in range(5):
            if t + f * 10 + 10 > steps:
                break
            waypoint_mask[row][f] = 1
            future_x = scenario["ego"]["x"][t + f * 10 + 10]
            future_y = scenario["ego"]["y"][t + f * 10 + 10]
            waypoints[row][f] = to_egocentric_point(future_x, future_y, ego_x, ego_y, ego_heading)

        centerline_search_box = BoundingBox2d(BasicPoint2d(ego_x - 10, ego_y - 10), BasicPoint2d(ego_x + 10, ego_y + 10))
        lateral_offset = None
        lane_heading_angle_diff = None
        for lanelet in lanelet2_map.laneletLayer.search(centerline_search_box):
            if (
                not hide_centerline
                and "subtype" in lanelet.attributes
                and lanelet.attributes["subtype"] == "road"
            ):
                arc = toArcCoordinates(to2D(lanelet.centerline), BasicPoint2d(ego_x, ego_y))
                if abs(arc.distance) < 4 and (lateral_offset is None or abs(arc.distance) < abs(lateral_offset)):
                    lateral_offset = -arc.distance

                    r1 = fromArcCoordinates(to2D(lanelet.centerline), ArcCoordinates(arc.length, 0))
                    r2 = fromArcCoordinates(to2D(lanelet.centerline), ArcCoordinates(arc.length + 1, 0))
                    dr = r2 - r1
                    if math.hypot(dr.x, dr.y) > 0.5:
                        road_direction = to_egocentric_vector(dr.x, dr.y, ego_heading)
                        lane_heading_angle_diff = abs(math.atan2(road_direction[1], road_direction[0]))
                    else:
                        lane_heading_angle_diff = 0
        if lateral_offset is None:
            lateral_offset = 0.0
            lane_heading_angle_diff = 0.0

        data = inputs[row]
        # Safety penalties, comfort, then lane-following preferences.
        data[0] = 1.0  # Collision reward coefficient
        data[1] = 1.0  # Safety reward coefficient
        data[2] = 1.0  # Offroad reward coefficient
        data[3] = 1.0  # Stop-line reward coefficient
        data[4] = 0.01  # Comfort reward coefficient
        data[5] = 2.5e-3  # Lane-center reward coefficient
        data[6] = 2.5e-2  # Lane-alignment reward coefficient
        data[7] = 5e-4  # Speed-limit reward coefficient
        data[8] = 0.0  # Lane-center bias
        data[9] = speed / 27.7
        data[10] = steering_angles[t] / 0.667
        data[11] = 50 / 3.6 / 27.7
        data[12] = scenario["ego"]["width"] / 15.0
        data[13] = scenario["ego"]["length"] / 30.0
        data[14] = 1.0  # Goal radius / 3 m
        data[15:17] = rel_goal_x * 0.005, rel_goal_y * 0.005
        data[17:19] = rel_next_goal_x * 0.005, rel_next_goal_y * 0.005
        remaining = scenario["ego"].get("remaining_goals", 3)
        if isinstance(remaining, list):
            remaining = remaining[t]
        data[19] = remaining >= 2
        data[20] = remaining >= 1
        data[21] = lane_heading_angle_diff
        data[22] = lateral_offset / 4.0
        write_history(data, t, **trajectory, policy_actions=policy_actions)

        assert len(scenario["others"]) <= 63

        nearby = (obj for obj in scenario["others"]
                  if (obj["x"][t] - ego_x)**2 + (obj["y"][t] - ego_y)**2 <= 2500)
        for i, obj in enumerate(nearby):
            obj_x, obj_y = to_egocentric_point(obj["x"][t], obj["y"][t], ego_x, ego_y, ego_heading)
            obj_heading_x, obj_heading_y = to_egocentric_vector(math.cos(obj["heading"][t]), math.sin(obj["heading"][t]), ego_heading)
            data[ego_dim + 7 * i] = obj_x * 0.02
            data[ego_dim + 7 * i + 1] = obj_y * 0.02
            data[ego_dim + 7 * i + 2] = obj["width"] / 15.0
            data[ego_dim + 7 * i + 3] = obj["length"] / 30.0
            data[ego_dim + 7 * i + 4] = obj_heading_x
            data[ego_dim + 7 * i + 5] = obj_heading_y
            data[ego_dim + 7 * i + 6] = obj["speed"][t] / 27.7

        search_box = BoundingBox2d(BasicPoint2d(ego_x - 150, ego_y - 150), BasicPoint2d(ego_x + 150, ego_y + 150))

        default_line_attribute = 0.0

        centerline_linestrings = [
            (lanelet.centerline, 0, default_line_attribute)
            for lanelet in lanelet2_map.laneletLayer.search(search_box)
            if not hide_centerline
            and "subtype" in lanelet.attributes
            and lanelet.attributes["subtype"] == "road"
        ]

        nearby_linestrings = lanelet2_map.lineStringLayer.search(search_box)

        border_linestrings = [
            (linestring, 2, default_line_attribute)
            for linestring in nearby_linestrings
            if not hide_road_border
            and "type" in linestring.attributes
            and linestring.attributes["type"] in {"road_border", "curbstone", "guard_rail"}
        ]

        stop_line_linestrings = []
        for linestring in nearby_linestrings:
            if "type" in linestring.attributes and linestring.attributes["type"] == "stop_line":
                if "subtype" in linestring.attributes and linestring.attributes["subtype"] == "yield_right":
                    stop_line_linestrings.append((linestring, RIGHT_HAND_STOP_LINE - 4, default_line_attribute))
                    continue
                traffic_light_states = scenario.get("traffic_light_states", {})
                state_sequence = traffic_light_states.get(str(linestring.id))
                if state_sequence is not None and state_sequence[t] != "off":
                    status = state_sequence[t]
                    if status == "red":
                        line_attribute = -0.5
                    elif status == "yellow":
                        line_attribute = 0.0
                    elif status == "green":
                        line_attribute = 0.5
                    else:
                        raise ValueError(
                            f"Unknown traffic-light state {status!r} for stop line "
                            f"{linestring.id} in {filepath}"
                        )
                    stop_line_linestrings.append((linestring, 5, line_attribute))
                elif stop_line_fallback_type(linestring.attributes) == STOP_SIGN:
                    stop_line_linestrings.append((linestring, 3, default_line_attribute))

        crosswalk_linestrings = [
            (linestring, 4, default_line_attribute)
            for lanelet in lanelet2_map.laneletLayer.search(search_box)
            if not (hide_road_border and hide_centerline)
            and "subtype" in lanelet.attributes
            and lanelet.attributes["subtype"] == "crosswalk"
            for linestring in [lanelet.leftBound, lanelet.rightBound]
        ]

        all_linestrings = itertools.chain(centerline_linestrings, border_linestrings, stop_line_linestrings, crosswalk_linestrings)

        road_data = []
        for linestring, road_type, line_attribute in all_linestrings:
            prev_p = None
            for p in iter_linestring(linestring):
                if prev_p is not None:
                    midpoint_x = (prev_p.x + p.x) / 2.0
                    midpoint_y = (prev_p.y + p.y) / 2.0
                    segment_length = math.hypot(p.x - midpoint_x, p.y - midpoint_y)

                    segment_dx = p.x - prev_p.x
                    segment_dy = p.y - prev_p.y
                    if math.hypot(segment_dx, segment_dy) < 0.001:
                        continue

                    segment_dir_length = math.hypot(segment_dx, segment_dy)
                    segment_dx /= segment_dir_length
                    segment_dy /= segment_dir_length
                    segment_rel_dx, segment_rel_dy = to_egocentric_vector(segment_dx, segment_dy, ego_heading)

                    midpoint_rel_x, midpoint_rel_y = to_egocentric_point(midpoint_x, midpoint_y, ego_x, ego_y, ego_heading)

                    road_data.append([
                        midpoint_rel_x * 0.02,
                        midpoint_rel_y * 0.02,
                        segment_length / 100.0,
                        line_attribute,
                        segment_rel_dx,
                        segment_rel_dy,
                        road_type
                    ])

                prev_p = p

        road_data = sorted(road_data, key=lambda row: math.hypot(row[0], row[1]))[:200]
        data[ego_dim + 7 * 63:ego_dim + 7 * 63 + len(road_data) * 7] = np.array(road_data).ravel()

    actions = issued_actions[:sample_count]

    if os.environ.get("DEBUG") == "1":
        with open(f"/tmp/supervised_debug_{os.path.basename(filepath)}.bin", "wb") as f:
            import lz4.block
            compressed = lz4.block.compress(inputs.tobytes(), store_size=False)

            import struct
            f.write(struct.pack("iii", inputs.shape[1], sample_count, len(compressed)))
            f.write(compressed)

            # Reward data, not relevant here so set to all zeros
            reward_compressed = lz4.block.compress(bytes([0]) * 4 * 8 * sample_count, store_size=False)
            f.write(struct.pack("ii", 8, len(reward_compressed)))
            f.write(reward_compressed)

            # Actions
            f.write(struct.pack("i", 2))
            f.write(actions.tobytes())

    return inputs, actions, waypoints, waypoint_mask


def main():
    scenario_dir = sys.argv[1]
    scenario_files = sorted([os.path.join(root, file) for root, _dirs, files in os.walk(scenario_dir) for file in files if file.endswith(".json")])
    with Pool() as pool:
        results = pool.map(process_scenario, scenario_files)

    inputs, actions, waypoints, waypoint_masks = zip(*results)
    inputs = np.stack(inputs, axis=0)
    actions = np.stack(actions, axis=0)
    waypoints = np.stack(waypoints, axis=0)
    waypoint_masks = np.stack(waypoint_masks, axis=0)

    np.savez("supervised_data.npz",
             inputs=inputs,
             actions=actions,
             waypoints=waypoints,
             waypoint_masks=waypoint_masks)

if __name__ == "__main__":
    main()
