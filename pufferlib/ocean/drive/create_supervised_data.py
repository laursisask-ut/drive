import os
import sys
import json
import math
import itertools
from multiprocessing import Pool

import numpy as np

from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d, BoundingBox2d
from lanelet2.geometry import ArcCoordinates, fromArcCoordinates, toArcCoordinates, to2D, distance

utm_origin_lat = 58.385345
utm_origin_lon = 26.726272
projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), True, False)
lanelet2_map = load("tartu_large_processed.osm", projector)

ACCELERATION_VALUES = [-4.0, -2.667, -1.333, 0.0, 1.333, 2.667, 4.0]
STEERING_VALUES = [-0.667, -0.5, -0.333, -0.167, -0.08, -0.035, 0.0, 0.035, 0.08, 0.167, 0.333, 0.500, 0.667]

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

def process_scenario(filepath):
    print(f"processing {filepath}", file=sys.stderr)
    scenario = json.load(open(filepath, "r"))
    hidden_road_border_ids = {
        int(value) for value in scenario.get("hidden_road_border_ids", [])
    }
    hidden_centerline_lanelet_ids = {
        int(value)
        for value in scenario.get("hidden_centerline_lanelet_ids", [])
    }

    inputs = np.zeros((90, 1861), dtype=np.float32)
    actions = np.zeros((90, 2), dtype=np.int32)
    actions[:, 0] = ACCELERATION_VALUES.index(0.0)
    actions[:, 1] = STEERING_VALUES.index(0.0)

    prev_heading = scenario["ego"]["heading"][0]
    prev_speed = scenario["ego"]["speed"][0]

    for t in range(90):
        ego_x = scenario["ego"]["x"][t]
        ego_y = scenario["ego"]["y"][t]
        ego_heading = scenario["ego"]["heading"][t]

        speed = scenario["ego"]["speed"][t]
        next_speed = scenario["ego"]["speed"][t + 1]
        next_heading = scenario["ego"]["heading"][t + 1]

        desired_acceleration = (next_speed - prev_speed) / 0.1
        accel_action_idx = min(range(len(ACCELERATION_VALUES)), key=lambda i: abs(ACCELERATION_VALUES[i] - desired_acceleration))
        actions[t][0] = accel_action_idx
        prev_speed = max(0.1, prev_speed + ACCELERATION_VALUES[accel_action_idx] * 0.1)

        if speed > 0.1:
            wheelbase = 0.6 * scenario["ego"]["length"]
            desired_steering = math.atan(calculate_heading_change(prev_heading, next_heading) * wheelbase / speed / 0.1)
            steering_action_idx = min(range(len(STEERING_VALUES)), key=lambda i: abs(STEERING_VALUES[i] - desired_steering))
            actions[t][1] = steering_action_idx
            prev_heading = normalize_heading(prev_heading + speed * 0.1 * math.tan(STEERING_VALUES[steering_action_idx]) / wheelbase)

        rel_goal_x, rel_goal_y = to_egocentric_point(scenario["ego"]["goal"]["x"], scenario["ego"]["goal"]["y"],
                                                     ego_x, ego_y, ego_heading)
        rel_next_goal_x, rel_next_goal_y = to_egocentric_point(scenario["ego"]["next_goal"]["x"], scenario["ego"]["next_goal"]["y"],
                                                               ego_x, ego_y, ego_heading)


        centerline_search_box = BoundingBox2d(BasicPoint2d(ego_x - 10, ego_y - 10), BasicPoint2d(ego_x + 10, ego_y + 10))
        lateral_offset = None
        lane_heading_angle_diff = None
        for lanelet in lanelet2_map.laneletLayer.search(centerline_search_box):
            if (
                lanelet.id not in hidden_centerline_lanelet_ids
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

        data = inputs[t]

        data[0] = rel_goal_x * 0.005
        data[1] = rel_goal_y * 0.005
        data[2] = speed / 27.7
        data[3] = scenario["ego"]["width"] / 15.0
        data[4] = scenario["ego"]["length"] / 30.0
        data[5] = 1.0 # stop line reward coefficient
        data[6] = 50 / 3.6 / 27.7 # max speed
        data[7] = 2.5e-3 / 7.5e-4 # lane center reward coefficient
        data[8] = 0.01 / 0.1 # comfort coefficient
        data[9] = 2.5e-2 / 2.5e-2 # lane align reward coefficient
        data[10] = 1.0 # collision reward coefficient
        data[11] = 1.0 # off-road reward coefficient
        data[12] = 0 # lane center bias
        data[13] = lane_heading_angle_diff
        data[14] = lateral_offset / 4.0
        data[15] = scenario["ego"]["height"] / 10.0 # height
        data[16] = rel_next_goal_x * 0.005
        data[17] = rel_next_goal_y * 0.005
        data[18] = 3.0 / 3.0 # goal radius
        data[19] = (3 - 1) / 2.0 # remaining goals = 3

        assert len(scenario["others"]) <= 63

        for i, obj in enumerate(scenario["others"]):
            obj_x, obj_y = to_egocentric_point(obj["x"][t], obj["y"][t], ego_x, ego_y, ego_heading)
            obj_heading_x, obj_heading_y = to_egocentric_vector(math.cos(obj["heading"][t]), math.sin(obj["heading"][t]), ego_heading)
            data[20 + 7 * i] = obj_x * 0.02
            data[20 + 7 * i + 1] = obj_y * 0.02
            data[20 + 7 * i + 2] = obj["width"] / 15.0
            data[20 + 7 * i + 3] = obj["length"] / 30.0
            data[20 + 7 * i + 4] = obj_heading_x
            data[20 + 7 * i + 5] = obj_heading_y
            data[20 + 7 * i + 6] = obj["speed"][t] / 27.7

        search_box = BoundingBox2d(BasicPoint2d(ego_x - 150, ego_y - 150), BasicPoint2d(ego_x + 150, ego_y + 150))

        default_line_attribute = 0.1 / 100

        centerline_linestrings = [
            (lanelet.centerline, 0, default_line_attribute)
            for lanelet in lanelet2_map.laneletLayer.search(search_box)
            if lanelet.id not in hidden_centerline_lanelet_ids
            and "subtype" in lanelet.attributes
            and lanelet.attributes["subtype"] == "road"
        ]

        nearby_linestrings = lanelet2_map.lineStringLayer.search(search_box)

        border_linestrings = [
            (linestring, 2, default_line_attribute)
            for linestring in nearby_linestrings
            if linestring.id not in hidden_road_border_ids
            and "type" in linestring.attributes
            and linestring.attributes["type"] in {"road_border", "curbstone", "guard_rail"}
        ]

        stop_line_linestrings = []
        for linestring in nearby_linestrings:
            if "type" in linestring.attributes and linestring.attributes["type"] == "stop_line":
                traffic_light_states = scenario.get("traffic_light_states", {})
                state_sequence = traffic_light_states.get(str(linestring.id))
                if state_sequence is not None:
                    status = state_sequence[t]
                    if status == "red":
                        line_attribute = -0.5
                    elif status == "yellow":
                        line_attribute = 0.0
                    elif status == "green":
                        line_attribute = 0.5
                    elif status == "off":
                        line_attribute = -1.0
                    else:
                        raise ValueError(
                            f"Unknown traffic-light state {status!r} for stop line "
                            f"{linestring.id} in {filepath}"
                        )
                    stop_line_linestrings.append((linestring, 5, line_attribute))
                elif linestring.attributes["subtype"] in {"yield_manual", "yield", "yield_stop"}:
                    stop_line_linestrings.append((linestring, 3, default_line_attribute))

        crosswalk_linestrings = [
            (linestring, 4, default_line_attribute)
            for lanelet in lanelet2_map.laneletLayer.search(search_box)
            if "subtype" in lanelet.attributes and lanelet.attributes["subtype"] == "crosswalk"
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
        data[20 + 7 * 63:20 + 7 * 63 + len(road_data) * 7] = np.array(road_data).ravel()

    if os.environ.get("DEBUG") == "1":
        with open(f"/tmp/supervised_debug_{os.path.basename(filepath)}.bin", "wb") as f:
            import lz4.block
            compressed = lz4.block.compress(inputs.tobytes(), store_size=False)

            import struct
            f.write(struct.pack("iii", 1861, 90, len(compressed)))
            f.write(compressed)

            # Reward data, not relevant here so set to all zeros
            reward_compressed = lz4.block.compress(bytes([0]) * 4 * 7 * 90, store_size=False)
            f.write(struct.pack("ii", 7, len(reward_compressed)))
            f.write(reward_compressed)

            # Actions
            f.write(struct.pack("i", 2))
            f.write(actions.tobytes())

    return inputs, actions


def main():
    scenario_dir = sys.argv[1]
    scenario_files = sorted([os.path.join(root, file) for root, _dirs, files in os.walk(scenario_dir) for file in files])
    with Pool() as pool:
        results = pool.map(process_scenario, scenario_files)

    inputs, actions = zip(*results)
    inputs = np.stack(inputs, axis=0)
    actions = np.stack(actions, axis=0)

    np.savez("supervised_data.npz", inputs=inputs, actions=actions)

if __name__ == "__main__":
    main()
