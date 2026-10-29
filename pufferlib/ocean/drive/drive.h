#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <limits.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <time.h>
#include "error.h"
#include <lz4.h>
#include "observation.h"

// Entity Types
#define NONE 0
#define VEHICLE 1
#define PEDESTRIAN 2
#define CYCLIST 3
#define ROAD_LANE 4
#define ROAD_LINE 5
#define ROAD_EDGE 6
#define STOP_SIGN 7
#define CROSSWALK 8
#define TRAFFIC_LIGHT_STOP_LINE 9
#define RIGHT_HAND_STOP_LINE 11
#define EVAL_TRAFFIC_LIGHT_LINE 98
#define EXIT_ORDER 99
#define MAX_EXIT_ORDER_AGENTS 32
#define MAP_ENTITY_TYPE_MASK 0xFF
#define MAP_TRAFFIC_LIGHT_YIELD_FLAG 0x200
#define MAP_TRAFFIC_LIGHT_SEQUENCE_FLAG 0x400

#define TRAFFIC_LIGHT_OFF 0
#define TRAFFIC_LIGHT_RED 1
#define TRAFFIC_LIGHT_YELLOW 2
#define TRAFFIC_LIGHT_GREEN 3
#define TRAFFIC_LIGHT_UPDATE_INTERVAL 5

#define INVALID_POSITION -10000.0f

// Minimum distance to goal position
#define MIN_DISTANCE_TO_GOAL 2.0f

// Actions
#define NOOP 0
#define MAX_ACTION_DELAY_STEPS 3
#define ACTION_QUEUE_CAPACITY (MAX_ACTION_DELAY_STEPS + 1)

// Dynamics Models
#define CLASSIC 0
#define JERK 1

// collision state
#define NO_COLLISION 0
#define VEHICLE_COLLISION 1
#define OFFROAD 2
#define RED_TRAFFIC_LIGHT_CROSSED 3

#define SAFETY_MIN_SPEED 2.0f
#define SAFETY_TIME_SECONDS 1.0f
#define SAFETY_CLEARANCE 0.1f
#define SAFETY_COLLISION_FRACTION 0.01f

// Metrics array indices
#define COLLISION_IDX 0
#define OFFROAD_IDX 1
#define REACHED_GOAL_IDX 2
#define LANE_ALIGNED_IDX 3
#define AVG_DISPLACEMENT_ERROR_IDX 4
#define LANE_LATERAL_OFFSET_IDX 5

// grid cell size
#define GRID_CELL_SIZE 5.0f
#define MAX_ENTITIES_PER_CELL 30    // Depends on resolution of data Formula: 3 * (2 + GRID_CELL_SIZE*sqrt(2)/resolution) => For each entity type in gridmap, diagonal poly-lines -> sqrt(2), include diagonal ends -> 2

// Max road segment observation entities
#define MAX_ROAD_SEGMENT_OBSERVATIONS 200
#define MAX_AGENTS 16384  // 5120 policy vehicles + 2048 parked + 8192 pedestrians + 1024 obstacles.
#define MAX_VISIBLE_AGENTS 64
// Observation Space Constants
#define MAX_SPEED 27.7f
#define MAX_VEH_LEN 30.0f
#define MAX_VEH_WIDTH 15.0f
#define MAX_VEH_HEIGHT 10.0f
#define MIN_REL_GOAL_COORD -1000.0f
#define MAX_REL_GOAL_COORD 1000.0f
#define MIN_REL_AGENT_POS -1000.0f
#define MAX_REL_AGENT_POS 1000.0f
#define MAX_ORIENTATION_RAD 2 * PI
#define MIN_RG_COORD -1000.0f
#define MAX_RG_COORD 1000.0f
#define MAX_ROAD_SCALE 100.0f
#define MAX_ROAD_SEGMENT_LENGTH 100.0f

#define MAX_EXPORT_STEPS 1000
#define MAX_REPLAY_OBSERVATION_AGENTS 64  // Trajectories include every policy slot.
#define REPLAY_ACTION_COMPONENT_COUNT 2
#define REPLAY_ACTION_ACCELERATION_IDX 0
#define REPLAY_ACTION_STEERING_IDX 1
#define REPLAY_REWARD_COMPONENT_COUNT 8

#define REWARD_COMPONENT_GOAL 0
#define REWARD_COMPONENT_LANE_CENTER 1
#define REWARD_COMPONENT_COMFORT 2
#define REWARD_COMPONENT_LANE_ALIGNMENT 3
#define REWARD_COMPONENT_COLLISION 4
#define REWARD_COMPONENT_SAFETY 5
#define REWARD_COMPONENT_OFFROAD 6
#define REWARD_COMPONENT_STOP_LINE 7
#define REWARD_COMPONENT_VELOCITY 8
#define REWARD_COMPONENT_ADE 9

#define COLLISION_COUNT_BINS 101
#define COLLISION_COUNT_BUCKETS 5151
#define COEFFICIENT_COUNT_BINS 100
#define LANE_CENTER_BIAS_BINS 100
#define LATERAL_OFFSET_BINS 100
#define LANE_CENTER_OFFSET_BUCKETS (LANE_CENTER_BIAS_BINS * LATERAL_OFFSET_BINS)
#define CLASSIC_ACCELERATION_ACTION_BINS 7
#define CLASSIC_STEERING_ACTION_BINS 13
#define SPEED_COUNT_BINS 100

// Jerk action space (for JERK dynamics model)
static const float JERK_LONG[4] = {-15.0f, -4.0f, 0.0f, 4.0f};
static const float JERK_LAT[3] = {-4.0f, 0.0f, 4.0f};

// Classic action space (for CLASSIC dynamics model)
static const float ACCELERATION_VALUES[7] = {-5.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 4.0f};
static const float STEERING_VALUES[13] = {-0.667f, -0.500f, -0.333f, -0.167f, -0.08f, -0.01f, 0.000f, 0.01f, 0.08f, 0.167f, 0.333f, 0.500f, 0.667f};

static const float offsets[4][2] = {
        {-1, 1},  // top-left
        {1, 1},   // top-right
        {1, -1},  // bottom-right
        {-1, -1}  // bottom-left
    };

static const int collision_offsets[25][2] = {
    {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2},  // Top row
    {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1},  // Second row
    {-2,  0}, {-1,  0}, {0,  0}, {1,  0}, {2,  0},  // Middle row (including center)
    {-2,  1}, {-1,  1}, {0,  1}, {1,  1}, {2,  1},  // Fourth row
    {-2,  2}, {-1,  2}, {0,  2}, {1,  2}, {2,  2}   // Bottom row
};

struct timespec ts;

typedef struct Drive Drive;
typedef struct Client Client;
typedef struct Log Log;
typedef struct Graph Graph;
typedef struct AdjListNode AdjListNode;
typedef struct ReplayExportLog ReplayExportLog;

struct Log {
    float episode_return;
    float goal_reward;
    float lane_center_reward;
    float comfort_reward;
    float lane_alignment_reward;
    float collision_reward;
    float safety_reward;
    float offroad_reward;
    float stop_line_reward;
    float velocity_reward;
    float ade_reward;
    float episode_length;
    float score;
    float offroad_rate;
    float stop_line_rate;
    float collision_rate;
    float num_goals_reached;
    float completion_rate;
    float dnf_rate;
    float n;
    float lane_alignment_rate;
    float avg_displacement_error;
    float active_agent_count;
    float expert_static_car_count;
    float static_car_count;
    float avg_offroad_per_agent;
    float avg_stop_line_per_agent;
    float avg_collisions_per_agent;
    float lane_lateral_offset;
    float lane_heading_angle_diff;
    float close_to_lane;
    float avg_distance_driven_per_agent;
    float priority_violations;
    float traffic_light_violations;
};

typedef struct Entity Entity;
struct Entity {
    int type;
    int array_size;
    float* traj_x;
    float* traj_y;
    float* traj_z;
    float* traj_vx;
    float* traj_vy;
    float* traj_vz;
    float* traj_heading;
    int* traj_valid;
    float width;
    float length;
    float height;
    float goal_position_x;
    float goal_position_y;
    float goal_position_z;
    float init_goal_x;
    float init_goal_y;
    int multi_goal_count;
    int multi_goal_idx;
    float* multi_goal_position_x;
    float* multi_goal_position_y;
    float* multi_goal_position_z;
    int mark_as_expert;
    int collision_state;
    int collided_with_agent_idx;
    float metrics_array[6]; // metrics_array: [collision, offroad, reached_goal, lane_aligned, avg_displacement_error, lane_lateral_offset]
    float prev_x;
    float prev_y;
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    float heading;
    float heading_x;
    float heading_y;
    float lane_heading_angle_diff;
    int goal_timestep;
    int current_lane_idx;
    int valid;
    int respawn_timestep;
    int respawn_count;
    int collided_before_goal;
    int sampled_new_goal;
    int reached_goal_this_episode;
    int reached_final_goal;
    int num_goals_reached;
    int active_agent;
    float cumulative_displacement;
    int displacement_sample_count;
    float goal_radius;
    float max_speed;
    float lane_center_reward_coefficient;
    float comfort_reward_coefficient;
    float lane_align_reward_coefficient;
    float collision_reward_coefficient;
    float safety_reward_coefficient;
    float offroad_reward_coefficient;
    float stop_line_reward_coefficient;
    float lane_center_bias;
    float object_detection_distance; // Actor visibility radius in meters; zero defaults to 50 m.
    float friction_coefficient;
    float steering_gain;
    float acceleration_gain;

    // Jerk dynamics
    float a_long;
    float a_lat;
    float jerk_long;
    float jerk_lat;
    float steering_angle;
    float wheelbase;
    float low_steering_rate;
    float high_steering_rate;
    float low_steering_rate_error;
    float high_steering_rate_error;

    // Private actuator state; delay is fixed for an episode. Discrete indices
    // and continuous commands both fit exactly in these two-component rows.
    int action_delay_steps;
    int action_queue_head;
    float action_queue[ACTION_QUEUE_CAPACITY][2];

    // Ring of pre-action states and normalized commands selected by the policy.
    int ego_history_head;
    float ego_history[EGO_HISTORY_CAPACITY][7];
};

typedef struct StaticRoadEntity StaticRoadEntity;
struct StaticRoadEntity {
    int type;
    int array_size;
    float* traj_x;
    float* traj_y;
    float* traj_z;
    int traffic_light_idx;
    int traffic_light_off_type; // STOP_SIGN for yield rules, NONE for priority.
};

typedef struct TrafficLightState TrafficLightState;

struct TrafficLightState {
    int traffic_light_color;
    int traffic_light_transition_in;
    int* sequence; // Optional map states, indexed by absolute scenario timestep.
    int sequence_length;
};

typedef struct EvalTrafficLightLine {
    float x[2], y[2];
    int traffic_light_idx; // Zero-based ordinal among the map's type-9 records.
} EvalTrafficLightLine;

typedef struct ExitOrder {
    int array_size;
    float* traj_x;
    float* traj_y;
    int exit_order[MAX_EXIT_ORDER_AGENTS];
    int expected_exit_order[MAX_EXIT_ORDER_AGENTS];
    int exit_count;
    int expected_exit_count;
} ExitOrder;

void free_entity(Entity* entity){
    // free trajectory arrays
    free(entity->traj_x);
    free(entity->traj_y);
    free(entity->traj_z);
    free(entity->traj_vx);
    free(entity->traj_vy);
    free(entity->traj_vz);
    free(entity->traj_heading);
    free(entity->traj_valid);
    free(entity->multi_goal_position_x);
    free(entity->multi_goal_position_y);
    free(entity->multi_goal_position_z);
}

// Utility functions
float relative_distance(float a, float b){
    float distance = sqrtf(powf(a - b, 2));
    return distance;
}

float relative_distance_2d(float x1, float y1, float x2, float y2){
    float dx = x2 - x1;
    float dy = y2 - y1;
    float distance = sqrtf(dx*dx + dy*dy);
    return distance;
}

float clip(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

static inline int rand_int_inclusive(int min_value, int max_value) {
    return min_value + rand() % (max_value - min_value + 1);
}

static inline int grid_map_entity_type(int type) {
    return (type >= ROAD_LANE && type <= TRAFFIC_LIGHT_STOP_LINE) || type == RIGHT_HAND_STOP_LINE;
}

static inline int sample_traffic_light_transition(int color) {
    switch (color) {
        case TRAFFIC_LIGHT_RED:
            return rand_int_inclusive(30, 60);
        case TRAFFIC_LIGHT_YELLOW:
            return rand_int_inclusive(30, 40);
        case TRAFFIC_LIGHT_GREEN:
            return rand_int_inclusive(30, 100);
        default:
            return 0;
    }
}

static inline float traffic_light_observation_value(int color) {
    switch (color) {
        case TRAFFIC_LIGHT_RED:
            return -0.5f;
        case TRAFFIC_LIGHT_YELLOW:
            return 0.0f;
        case TRAFFIC_LIGHT_GREEN:
            return 0.5f;
        case TRAFFIC_LIGHT_OFF:
        default:
            return -1.0f;
    }
}

static inline Color traffic_light_render_color(int color) {
    switch (color) {
        case TRAFFIC_LIGHT_RED:
            return RED;
        case TRAFFIC_LIGHT_YELLOW:
            return ORANGE;
        case TRAFFIC_LIGHT_GREEN:
            return GREEN;
        case TRAFFIC_LIGHT_OFF:
        default:
            return DARKGRAY;
    }
}

float compute_displacement_error(Entity* agent, int timestep) {
    // Check if timestep is within valid range
    if (timestep < 0 || timestep >= agent->array_size) {
        return 0.0f;
    }

    // Check if reference trajectory is valid at this timestep
    if (!agent->traj_valid[timestep]) {
        return 0.0f;
    }

    // Get reference position at current timestep, skip invalid ones
    float ref_x = agent->traj_x[timestep];
    float ref_y = agent->traj_y[timestep];

    if (ref_x == INVALID_POSITION || ref_y == INVALID_POSITION) {
        return 0.0f;
    }

    // Compute deltas: Euclidean distance between actual and reference position
    float dx = agent->x - ref_x;
    float dy = agent->y - ref_y;
    float displacement = sqrtf(dx*dx + dy*dy);

    return displacement;
}

typedef struct GridMapEntity GridMapEntity;
struct GridMapEntity {
    int entity_idx;  // Index into static_road_entities, independent of actor count.
    int geometry_idx;
};

typedef struct GridMap GridMap;
struct GridMap {
    float top_left_x;
    float top_left_y;
    float bottom_right_x;
    float bottom_right_y;
    int grid_cols;
    int grid_rows;
    int cell_size_x;
    int cell_size_y;
    int* cell_entities_count;  // number of entities in each cell of the GridMap
    GridMapEntity** cells;  // list of gridEntities in each cell of the GridMap

    // Extras/Optimizations
    int vision_range;
};

// Immutable geometry and spatial indices shared by environments. Each environment
// (and the Python map owner, when present) holds one reference. Traffic-light
// colors and timers belong to Drive, never to this shared map.
typedef struct StaticRoadMap {
    int ref_count;
    int num_roads;
    int num_traffic_lights;
    StaticRoadEntity* entities;
    GridMap* grid_map;
    int* neighbor_offsets;
    Graph* topology_graph;
    int topology_initialized;
} StaticRoadMap;

struct ReplayExportLog {
    int capacity;
    int num_agents; // Policy slots with observations/actions.
    int num_actors; // Policy slots first, then background actors (including experts).
    int* actor_indices;
    float* traj_x;
    float* traj_y;
    float* traj_z;
    float* traj_vx;
    float* traj_vy;
    float* traj_vz;
    float* heading;
    int* valid;
    int* traffic_light_states; // Light-major, indexed by absolute scenario timestep.
    float* observations;
    float* reward_components;
    int* action_indices;
};

struct Drive {
    Client* client;
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* truncations;
    Log log;
    Log* logs;
    int* collision_counts;
    int* expert_collision_counts;
    int* offroad_counts;
    int* stop_line_counts;
    int* comfort_violation_counts;
    int* lane_center_offset_counts;
    int* acceleration_action_counts;
    int* steering_action_counts;
    int* speed_counts;
    int num_agents;
    int active_agent_count;
    int* active_agent_indices;
    int action_type;
    int human_agent_idx;
    Entity* entities;
    ExitOrder* exit_orders;
    int num_exit_orders;
    EvalTrafficLightLine* eval_traffic_light_lines;
    int num_eval_traffic_light_lines;
    int eval_traffic_light_violation;
    StaticRoadEntity* static_road_entities;
    StaticRoadMap* static_road_map;
    TrafficLightState* traffic_light_states;
    Graph* topology_graph;
    int num_controllable_agents;
    int num_objects;
    int num_roads;
    int num_traffic_lights;
    int static_car_count;
    int* static_car_indices;
    int expert_static_car_count;
    int* expert_static_car_indices;
    int timestep;
    int init_steps;
    int dynamics_model;
    GridMap* grid_map;
    int* neighbor_offsets;
    float reward_vehicle_collision;
    float reward_offroad_collision;
    float reward_ade;
    char* map_name;
    float dt;
    float reward_goal;
    float reward_goal_post_respawn;
    float goal_radius;
    int control_all_agents;
    int deterministic_agent_selection;
    int play_exactly_once;
    int policy_agents_per_env;
    int logs_capacity;
    int use_goal_generation;
    char* ini_file;
    int scenario_length;
    int control_non_vehicles;
    int dynamic_map_generation;
    int traffic_lights_enabled;
    int has_traffic_light_sequences;
    int export_replay_enabled;
    char* replay_dir;
    ReplayExportLog export_log;
};

static int inside_exit_order(const ExitOrder* order, float x, float y) {
    int inside = 0;
    for (int i = 0, j = order->array_size - 1; i < order->array_size; j = i++) {
        float xi = order->traj_x[i], yi = order->traj_y[i];
        float xj = order->traj_x[j], yj = order->traj_y[j];
        // Points on the boundary are still inside.
        if ((x - xi) * (yj - yi) == (y - yi) * (xj - xi) &&
            x >= fminf(xi, xj) && x <= fmaxf(xi, xj) &&
            y >= fminf(yi, yj) && y <= fmaxf(yi, yj)) return 1;
        if ((yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi)
            inside = !inside;
    }
    return inside;
}

static int exited_polygon(const ExitOrder* order, float prev_x, float prev_y, float x, float y) {
    // An invalid/expired expert trajectory is not a polygon exit.
    if (!isfinite(prev_x) || !isfinite(prev_y) || !isfinite(x) || !isfinite(y) ||
        prev_x == INVALID_POSITION || prev_y == INVALID_POSITION ||
        x == INVALID_POSITION || y == INVALID_POSITION) return 0;
    return inside_exit_order(order, prev_x, prev_y) && !inside_exit_order(order, x, y);
}

static void append_exit(int* order, int* count, int agent_idx) {
    if (*count == MAX_EXIT_ORDER_AGENTS) return;
    for (int i = 0; i < *count; i++) {
        if (order[i] == agent_idx) return;
    }
    order[(*count)++] = agent_idx;
}

static void reset_exit_orders(Drive* env) {
    for (int i = 0; i < env->num_exit_orders; i++) {
        ExitOrder* order = &env->exit_orders[i];
        order->exit_count = order->expected_exit_count = 0;
        // add_log runs before movement at scenario_length. Match that window;
        // exits on the same step are ordered by entity index in both sequences.
        for (int t = env->init_steps + 1; t < env->scenario_length; t++) {
            for (int idx = 0; idx < env->num_objects; idx++) {
                const Entity* e = &env->entities[idx];
                if (e->type != VEHICLE || t <= 0 || t >= e->array_size ||
                    (e->traj_valid && (!e->traj_valid[t - 1] || !e->traj_valid[t]))) continue;
                if (exited_polygon(order, e->traj_x[t - 1], e->traj_y[t - 1], e->traj_x[t], e->traj_y[t]))
                    append_exit(order->expected_exit_order, &order->expected_exit_count, idx);
            }
        }
    }
}

static void update_exit_orders(Drive* env) {
    for (int i = 0; i < env->num_exit_orders; i++) {
        ExitOrder* order = &env->exit_orders[i];
        for (int idx = 0; idx < env->num_objects; idx++) {
            const Entity* e = &env->entities[idx];
            if (e->type != VEHICLE) continue;
            if (!e->active_agent && e->mark_as_expert) {
                int t = env->timestep;
                if (t <= 0 || t >= e->array_size ||
                    (e->traj_valid && (!e->traj_valid[t - 1] || !e->traj_valid[t]))) continue;
            }
            if (exited_polygon(order, e->prev_x, e->prev_y, e->x, e->y))
                append_exit(order->exit_order, &order->exit_count, idx);
        }
    }
}

static int road_traffic_light_color(const Drive* env, const StaticRoadEntity* road) {
    int idx = road->traffic_light_idx;
    if (!env->traffic_lights_enabled || !env->traffic_light_states || idx < 0 || idx >= env->num_traffic_lights) {
        return TRAFFIC_LIGHT_OFF;
    }
    return env->traffic_light_states[idx].traffic_light_color;
}

static int effective_road_type(const Drive* env, const StaticRoadEntity* road) {
    if (road->type == TRAFFIC_LIGHT_STOP_LINE && road_traffic_light_color(env, road) == TRAFFIC_LIGHT_OFF)
        return road->traffic_light_off_type;
    return road->type;
}

typedef struct {
    int candidates[MAX_AGENTS];
    int candidates_count;
    int forced_experts[MAX_AGENTS];
    int forced_experts_count;
    int statics[MAX_AGENTS];
    int statics_count;
} SelectionBuckets;

static inline void push_capped(int* arr, int* count, int val, int cap) {
    if (*count < cap) {
        arr[(*count)++] = val;
    }
}

static inline float ego_goal_distance_t0(const Entity* e) {
    // Requires traj_* arrays and goal position
    float cos_heading = cosf(e->traj_heading[0]);
    float sin_heading = sinf(e->traj_heading[0]);
    float goal_x = e->goal_position_x - e->traj_x[0];
    float goal_y = e->goal_position_y - e->traj_y[0];
    float rel_goal_x = goal_x * cos_heading + goal_y * sin_heading;
    float rel_goal_y = -goal_x * sin_heading + goal_y * cos_heading;
    return relative_distance_2d(0, 0, rel_goal_x, rel_goal_y);
}

static inline int vehicle_eligible_t0(const Entity* e) {
    if (e->traj_valid == NULL || e->traj_valid[0] != 1) return 0;
    float dist = ego_goal_distance_t0(e);
    return dist >= 2.0f;
}

static inline void fisher_yates_shuffle(int* arr, int n) {
    for (int i = n - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

static void scan_vehicles_initial(const Drive* env, SelectionBuckets* out, int control_all_agents) {
    out->candidates_count = 0;
    out->forced_experts_count = 0;
    out->statics_count = 0;

    for (int i = 0; i < env->num_objects; i++) {
        const Entity* e = &env->entities[i];
        if (e->type != VEHICLE) continue;

        int eligible = vehicle_eligible_t0(e);
        if (!eligible) {
            push_capped(out->statics, &out->statics_count, i, MAX_AGENTS);
            continue;
        }

        if (control_all_agents) {
            push_capped(out->candidates, &out->candidates_count, i, MAX_AGENTS);
        } else {
            if (e->mark_as_expert == 1) {
                push_capped(out->forced_experts, &out->forced_experts_count, i, MAX_AGENTS);
            } else {
                push_capped(out->candidates, &out->candidates_count, i, MAX_AGENTS);
            }
        }
    }
}
void add_log(Drive* env) {
    env->log.traffic_light_violations += env->eval_traffic_light_violation;
    for (int i = 0; i < env->num_exit_orders; i++) {
        const ExitOrder* order = &env->exit_orders[i];
        if (order->exit_count != order->expected_exit_count ||
            memcmp(order->exit_order, order->expected_exit_order, order->exit_count * sizeof(int)) != 0) {
            env->log.priority_violations += 1.0f;
            break; // One binary violation per episode, regardless of agent/polygon count.
        }
    }
    for(int i = 0; i < env->active_agent_count; i++){
        Entity* e = &env->entities[env->active_agent_indices[i]];

        if (e->multi_goal_count) {
            env->log.completion_rate += ((float)e->multi_goal_idx) / e->multi_goal_count;
        } else if (e->reached_goal_this_episode) {
            env->log.completion_rate += 1.0f;
        }
        int offroad = env->logs[i].offroad_rate;
        env->log.offroad_rate += offroad;
        int stop_line = env->logs[i].stop_line_rate;
        env->log.stop_line_rate += stop_line;
        int collided = env->logs[i].collision_rate;
        env->log.collision_rate += collided;
        float avg_offroad_per_agent = env->logs[i].avg_offroad_per_agent;
        env->log.avg_offroad_per_agent += avg_offroad_per_agent;
        float avg_stop_line_per_agent = env->logs[i].avg_stop_line_per_agent;
        env->log.avg_stop_line_per_agent += avg_stop_line_per_agent;
        float avg_collisions_per_agent = env->logs[i].avg_collisions_per_agent;
        env->log.avg_collisions_per_agent += avg_collisions_per_agent;
        int num_goals_reached = env->logs[i].num_goals_reached;
        env->log.num_goals_reached += num_goals_reached;
        if(e->reached_goal_this_episode && !e->collided_before_goal){
            env->log.score += 1.0f;
        }
        if(!offroad && !collided && !e->reached_goal_this_episode){
            env->log.dnf_rate += 1.0f;
        }
        int lane_aligned = env->logs[i].lane_alignment_rate;
        env->log.lane_alignment_rate += lane_aligned;
        float displacement_error = env->logs[i].avg_displacement_error;
        env->log.avg_displacement_error += displacement_error;
        env->log.lane_heading_angle_diff += env->logs[i].lane_heading_angle_diff;
        env->log.lane_lateral_offset += env->logs[i].lane_lateral_offset;
        env->log.close_to_lane += env->logs[i].close_to_lane;
        env->log.avg_distance_driven_per_agent += env->logs[i].avg_distance_driven_per_agent;
        env->log.episode_length += env->logs[i].episode_length;
        env->log.episode_return += env->logs[i].episode_return;
        env->log.goal_reward += env->logs[i].goal_reward;
        env->log.lane_center_reward += env->logs[i].lane_center_reward;
        env->log.comfort_reward += env->logs[i].comfort_reward;
        env->log.lane_alignment_reward += env->logs[i].lane_alignment_reward;
        env->log.collision_reward += env->logs[i].collision_reward;
        env->log.safety_reward += env->logs[i].safety_reward;
        env->log.offroad_reward += env->logs[i].offroad_reward;
        env->log.stop_line_reward += env->logs[i].stop_line_reward;
        env->log.velocity_reward += env->logs[i].velocity_reward;
        env->log.ade_reward += env->logs[i].ade_reward;
        // Log composition counts per agent so vec_log averaging recovers the per-env value
        env->log.active_agent_count += env->active_agent_count;
        env->log.expert_static_car_count += env->expert_static_car_count;
        env->log.static_car_count += env->static_car_count;
        env->log.n += 1;
    }
}

static inline void accumulate_reward_component(Log* log, int component, float reward) {
    switch (component) {
    case REWARD_COMPONENT_GOAL:
        log->goal_reward += reward;
        break;
    case REWARD_COMPONENT_LANE_CENTER:
        log->lane_center_reward += reward;
        break;
    case REWARD_COMPONENT_COMFORT:
        log->comfort_reward += reward;
        break;
    case REWARD_COMPONENT_LANE_ALIGNMENT:
        log->lane_alignment_reward += reward;
        break;
    case REWARD_COMPONENT_COLLISION:
        log->collision_reward += reward;
        break;
    case REWARD_COMPONENT_SAFETY:
        log->safety_reward += reward;
        break;
    case REWARD_COMPONENT_OFFROAD:
        log->offroad_reward += reward;
        break;
    case REWARD_COMPONENT_STOP_LINE:
        log->stop_line_reward += reward;
        break;
    case REWARD_COMPONENT_VELOCITY:
        log->velocity_reward += reward;
        break;
    case REWARD_COMPONENT_ADE:
        log->ade_reward += reward;
        break;
    default:
        break;
    }
}

static inline void record_reward_component(Drive* env, int agent_slot, int component, float reward) {
    if (reward == 0.0f) {
        return;
    }

    env->logs[agent_slot].episode_return += reward;
    accumulate_reward_component(&env->logs[agent_slot], component, reward);

    if (env->export_replay_enabled &&
        env->export_log.reward_components != NULL &&
        agent_slot < env->export_log.num_agents &&
        env->timestep < env->export_log.capacity &&
        component < REPLAY_REWARD_COMPONENT_COUNT) {
        size_t export_idx =
            ((size_t)agent_slot * env->export_log.capacity + (size_t)env->timestep) * REPLAY_REWARD_COMPONENT_COUNT +
            (size_t)component;
        env->export_log.reward_components[export_idx] += reward;
    }
}

static inline int clipped_scaled_bin(float value, float scale, int bin_count) {
    if (!isfinite(value)) {
        return 0;
    }

    int bin = (int)(value * scale);
    if (bin < 0) {
        return 0;
    }
    if (bin >= bin_count) {
        return bin_count - 1;
    }
    return bin;
}

static inline int clipped_range_bin(float value, float min_value, float max_value, int bin_count) {
    if (!isfinite(value) || value <= min_value) {
        return 0;
    }
    if (value >= max_value) {
        return bin_count - 1;
    }

    float scale = (float)bin_count / (max_value - min_value);
    int bin = (int)((value - min_value) * scale);
    if (bin < 0) {
        return 0;
    }
    if (bin >= bin_count) {
        return bin_count - 1;
    }
    return bin;
}

static inline void record_lane_center_offset_count(Drive* env, float lateral_offset, float lane_center_bias) {
    int lateral_offset_bin = clipped_range_bin(lateral_offset, -2.0f, 2.0f, LATERAL_OFFSET_BINS);
    int lane_center_bias_bin = clipped_range_bin(lane_center_bias, -0.3f, 0.3f, LANE_CENTER_BIAS_BINS);
    int joint_bin_idx = lateral_offset_bin * LANE_CENTER_BIAS_BINS + lane_center_bias_bin;
    env->lane_center_offset_counts[joint_bin_idx]++;
}

static inline void record_speed_count(Drive* env, Entity* agent) {
    float max_speed = MAX_SPEED;
    if (agent->max_speed > 0.0f && agent->max_speed < MAX_SPEED) {
        max_speed = agent->max_speed;
    }

    float speed = sqrtf(agent->vx*agent->vx + agent->vy*agent->vy);
    int speed_bin = clipped_scaled_bin(speed / max_speed, 100.0f, SPEED_COUNT_BINS);
    env->speed_counts[speed_bin]++;
}

static inline void record_discrete_classic_action(
    Drive* env, int agent_slot, int acceleration_index, int steering_index) {
    if (acceleration_index >= 0 && acceleration_index < CLASSIC_ACCELERATION_ACTION_BINS) {
        env->acceleration_action_counts[acceleration_index]++;
    }
    if (steering_index >= 0 && steering_index < CLASSIC_STEERING_ACTION_BINS) {
        env->steering_action_counts[steering_index]++;
    }

    if (!env->export_replay_enabled ||
        env->export_log.action_indices == NULL ||
        agent_slot >= env->export_log.num_agents ||
        env->timestep >= env->export_log.capacity) {
        return;
    }

    size_t export_idx =
        ((size_t)agent_slot * env->export_log.capacity + (size_t)env->timestep) * REPLAY_ACTION_COMPONENT_COUNT;
    env->export_log.action_indices[export_idx + REPLAY_ACTION_ACCELERATION_IDX] = acceleration_index;
    env->export_log.action_indices[export_idx + REPLAY_ACTION_STEERING_IDX] = steering_index;
}


static void record_replay_trajectories(Drive* env) {
    if (!env->export_replay_enabled || !env->export_log.traj_x ||
        env->timestep < 0 || env->timestep >= env->export_log.capacity) return;

    // The first num_agents slots still match agent_XX.bin / observation rows.
    for (int slot = 0; slot < env->export_log.num_actors; slot++) {
        const Entity* e = &env->entities[env->export_log.actor_indices[slot]];
        size_t index = (size_t)slot * env->export_log.capacity + env->timestep;
        env->export_log.traj_x[index] = e->x;
        env->export_log.traj_y[index] = e->y;
        env->export_log.traj_z[index] = e->z;
        env->export_log.traj_vx[index] = e->vx;
        env->export_log.traj_vy[index] = e->vy;
        env->export_log.traj_vz[index] = e->vz;
        env->export_log.heading[index] = e->heading;
        int valid = isfinite(e->x) && isfinite(e->y) &&
            e->x != INVALID_POSITION && e->y != INVALID_POSITION;
        if (!e->active_agent && e->mark_as_expert && !env->dynamic_map_generation)
            valid = valid && env->timestep < e->array_size && e->traj_valid[env->timestep];
        env->export_log.valid[index] = valid;
    }
    for (int i = 0; i < env->num_traffic_lights; i++)
        env->export_log.traffic_light_states[(size_t)i * env->export_log.capacity + env->timestep] =
            env->traffic_lights_enabled ? env->traffic_light_states[i].traffic_light_color : TRAFFIC_LIGHT_OFF;
}


struct AdjListNode {
    int dest;
    struct AdjListNode* next;
};

struct Graph {
    int V;
    struct AdjListNode** array;
};

// Function to create a new adjacency list node
struct AdjListNode* newAdjListNode(int dest) {
    struct AdjListNode* newNode = malloc(sizeof(struct AdjListNode));
    newNode->dest = dest;
    newNode->next = NULL;
    return newNode;
}

// Function to create a graph of V vertices
struct Graph* createGraph(int V) {
    struct Graph* graph = malloc(sizeof(struct Graph));
    graph->V = V;
    graph->array = calloc(V, sizeof(struct AdjListNode*));
    return graph;
}

// Function to get next lanes from a given lane entity index
// Returns the number of next lanes found, fills next_lanes array with entity indices
int getNextLanes(struct Graph* graph, int entity_idx, int* next_lanes, int max_lanes) {
    if (!graph || entity_idx < 0 || entity_idx >= graph->V) {
        return 0;
    }

    int count = 0;
    struct AdjListNode* node = graph->array[entity_idx];

    while (node && count < max_lanes) {
        next_lanes[count] = node->dest;
        count++;
        node = node->next;
    }

    return count;
}

// Function to free the topology graph
void freeTopologyGraph(struct Graph* graph) {
    if (!graph) return;

    for (int i = 0; i < graph->V; i++) {
        struct AdjListNode* node = graph->array[i];
        while (node) {
            struct AdjListNode* temp = node;
            node = node->next;
            free(temp);
        }
    }

    free(graph->array);
    free(graph);
}

static void free_static_road_entities(StaticRoadEntity* roads, int count) {
    if (!roads) return;
    for (int i = 0; i < count; i++) {
        free(roads[i].traj_x);
        free(roads[i].traj_y);
        free(roads[i].traj_z);
    }
    free(roads);
}

static void free_grid_map(GridMap* grid) {
    if (!grid) return;
    for (int i = 0; i < grid->grid_cols * grid->grid_rows; i++) free(grid->cells[i]);
    free(grid->cells);
    free(grid->cell_entities_count);
    free(grid);
}

static void release_static_road_map(StaticRoadMap* map) {
    if (!map || --map->ref_count > 0) return;
    free_static_road_entities(map->entities, map->num_roads);
    free_grid_map(map->grid_map);
    free(map->neighbor_offsets);
    freeTopologyGraph(map->topology_graph);
    free(map);
}

static void acquire_static_road_map(Drive* env, StaticRoadMap* map) {
    map->ref_count++;
    env->static_road_map = map;
    env->static_road_entities = map->entities;
    env->num_roads = map->num_roads;
    env->num_traffic_lights = map->num_traffic_lights;
    env->grid_map = map->grid_map;
    env->neighbor_offsets = map->neighbor_offsets;
    env->topology_graph = map->topology_graph;
}

// Compatibility for generators returning the old combined Entity allocation.
// Transfer trajectory ownership, then release the combined array itself.
static void split_legacy_entities(Drive* env) {
    Entity* combined = env->entities;
    env->entities = (Entity*)calloc(env->num_objects, sizeof(Entity));
    memcpy(env->entities, combined, env->num_objects * sizeof(Entity));
    env->static_road_entities = (StaticRoadEntity*)calloc(env->num_roads, sizeof(StaticRoadEntity));
    int traffic_light_idx = 0;
    for (int i = 0; i < env->num_roads; i++) {
        Entity* e = &combined[env->num_objects + i];
        StaticRoadEntity* road = &env->static_road_entities[i];
        road->type = e->type & MAP_ENTITY_TYPE_MASK;
        road->traffic_light_off_type = (e->type & MAP_TRAFFIC_LIGHT_YIELD_FLAG) ? STOP_SIGN : NONE;
        road->array_size = e->array_size;
        road->traj_x = e->traj_x;
        road->traj_y = e->traj_y;
        road->traj_z = e->traj_z;
        road->traffic_light_idx = road->type == TRAFFIC_LIGHT_STOP_LINE ? traffic_light_idx++ : -1;
        e->traj_x = e->traj_y = e->traj_z = NULL;
        free_entity(e);
    }
    free(combined);
}


Entity* load_map_binary(const char* filename, Drive* env) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;
    fread(&env->num_objects, sizeof(int), 1, file);
    fread(&env->num_roads, sizeof(int), 1, file);
    Entity* entities = (Entity*)calloc(env->num_objects, sizeof(Entity));
    env->static_road_entities = (StaticRoadEntity*)calloc(env->num_roads, sizeof(StaticRoadEntity));
    int num_records = env->num_objects + env->num_roads;
    env->num_roads = 0;
    int traffic_light_idx = 0;
    int initial_scenario_length = env->scenario_length;
    for (int i = 0; i < num_records; i++) {
        Entity road_record = {0};
        Entity* e = i < env->num_objects ? &entities[i] : &road_record;
	    // Read base entity data
        fread(&e->type, sizeof(int), 1, file);
        int is_v2 = (e->type & 256) != 0;
        int has_light_sequence = (e->type & MAP_TRAFFIC_LIGHT_SEQUENCE_FLAG) != 0;
        int light_off_type = (e->type & MAP_TRAFFIC_LIGHT_YIELD_FLAG) ? STOP_SIGN : NONE;
        e->type &= MAP_ENTITY_TYPE_MASK;
        fread(&e->array_size, sizeof(int), 1, file);
        // Allocate arrays based on type
        int size = e->array_size;
        e->traj_x = (float*)malloc(size * sizeof(float));
        e->traj_y = (float*)malloc(size * sizeof(float));
        e->traj_z = (float*)malloc(size * sizeof(float));
        if (e->type == VEHICLE || e->type == PEDESTRIAN || e->type == CYCLIST) {  // Object type
            if (initial_scenario_length == 0 && size > env->scenario_length) env->scenario_length = size;
            // Allocate arrays for object-specific data
            e->traj_vx = (float*)malloc(size * sizeof(float));
            e->traj_vy = (float*)malloc(size * sizeof(float));
            e->traj_vz = (float*)malloc(size * sizeof(float));
            e->traj_heading = (float*)malloc(size * sizeof(float));
            e->traj_valid = (int*)malloc(size * sizeof(int));
        } else {
            // Roads don't use these arrays
            e->traj_vx = NULL;
            e->traj_vy = NULL;
            e->traj_vz = NULL;
            e->traj_heading = NULL;
            e->traj_valid = NULL;
        }
        // Read array data
        fread(e->traj_x, sizeof(float), size, file);
        fread(e->traj_y, sizeof(float), size, file);
        fread(e->traj_z, sizeof(float), size, file);
        if (e->type == VEHICLE || e->type == PEDESTRIAN || e->type == CYCLIST) {  // Object type
            fread(e->traj_vx, sizeof(float), size, file);
            fread(e->traj_vy, sizeof(float), size, file);
            fread(e->traj_vz, sizeof(float), size, file);
            fread(e->traj_heading, sizeof(float), size, file);
            fread(e->traj_valid, sizeof(int), size, file);

            // Reasonable defaults, needed for validation runs
            e->max_speed = 90/3.6f;
            e->lane_center_reward_coefficient = 2.5e-3;
            e->comfort_reward_coefficient = 0.01f;
            e->lane_align_reward_coefficient = 2.5e-2f;
            e->collision_reward_coefficient = 1.0f;
            e->safety_reward_coefficient = 1.0f;
            e->offroad_reward_coefficient = 1.0f;
            e->stop_line_reward_coefficient = 1.0f;
            e->lane_center_bias = 0.0f;
            e->goal_radius = 3.0f;
        }
        // Read remaining scalar fields
        fread(&e->width, sizeof(float), 1, file);
        fread(&e->length, sizeof(float), 1, file);
        fread(&e->height, sizeof(float), 1, file);
        fread(&e->goal_position_x, sizeof(float), 1, file);
        fread(&e->goal_position_y, sizeof(float), 1, file);
        fread(&e->goal_position_z, sizeof(float), 1, file);
        fread(&e->mark_as_expert, sizeof(int), 1, file);
        if (is_v2) {
            fread(&e->multi_goal_count, sizeof(int), 1, file);
            if (e->multi_goal_count > 0) {
                e->multi_goal_position_x = (float*)malloc(e->multi_goal_count * sizeof(float));
                e->multi_goal_position_y = (float*)malloc(e->multi_goal_count * sizeof(float));
                e->multi_goal_position_z = (float*)malloc(e->multi_goal_count * sizeof(float));
                fread(e->multi_goal_position_x, sizeof(float), e->multi_goal_count, file);
                fread(e->multi_goal_position_y, sizeof(float), e->multi_goal_count, file);
                fread(e->multi_goal_position_z, sizeof(float), e->multi_goal_count, file);
            }
        }
        if (i >= env->num_objects) {
            if (e->type == EVAL_TRAFFIC_LIGHT_LINE) {
                assert(e->array_size == 2);
                env->eval_traffic_light_lines = realloc(env->eval_traffic_light_lines,
                    (env->num_eval_traffic_light_lines + 1) * sizeof(EvalTrafficLightLine));
                EvalTrafficLightLine* line = &env->eval_traffic_light_lines[env->num_eval_traffic_light_lines++];
                memcpy(line->x, e->traj_x, sizeof(line->x));
                memcpy(line->y, e->traj_y, sizeof(line->y));
                line->traffic_light_idx = e->mark_as_expert;
                free_entity(e);
                continue;
            }
            // Type 99 uses the existing road record format, but owns only XY.
            if (e->type == EXIT_ORDER) {
                env->exit_orders = realloc(env->exit_orders, (env->num_exit_orders + 1) * sizeof(ExitOrder));
                env->exit_orders[env->num_exit_orders++] = (ExitOrder){
                    .array_size = e->array_size, .traj_x = e->traj_x, .traj_y = e->traj_y,
                };
                e->traj_x = e->traj_y = NULL;
                free_entity(e);
                continue;
            }
            StaticRoadEntity* road = &env->static_road_entities[env->num_roads++];
            road->type = e->type;
            road->array_size = e->array_size;
            road->traj_x = e->traj_x;
            road->traj_y = e->traj_y;
            road->traj_z = e->traj_z;
            road->traffic_light_idx = road->type == TRAFFIC_LIGHT_STOP_LINE ? traffic_light_idx++ : -1;
            road->traffic_light_off_type = light_off_type;
            if (road->type == TRAFFIC_LIGHT_STOP_LINE) {
                env->traffic_light_states = realloc(env->traffic_light_states,
                    traffic_light_idx * sizeof(TrafficLightState));
                TrafficLightState* state = &env->traffic_light_states[road->traffic_light_idx];
                *state = (TrafficLightState){0};
                if (has_light_sequence) {
                    fread(&state->sequence_length, sizeof(int), 1, file);
                    assert(state->sequence_length > 0);
                    state->sequence = malloc(state->sequence_length * sizeof(int));
                    fread(state->sequence, sizeof(int), state->sequence_length, file);
                    for (int t = 0; t < state->sequence_length; t++)
                        assert(state->sequence[t] >= TRAFFIC_LIGHT_OFF && state->sequence[t] <= TRAFFIC_LIGHT_GREEN);
                    env->has_traffic_light_sequences = 1;
                }
            }
            e->traj_x = e->traj_y = e->traj_z = NULL;
            free_entity(e);
        }
    }
    env->num_traffic_lights = traffic_light_idx;
    for (int i = 0; i < env->num_eval_traffic_light_lines; i++) {
        int idx = env->eval_traffic_light_lines[i].traffic_light_idx;
        assert(idx >= 0 && idx < env->num_traffic_lights);
    }
    fclose(file);
    return entities;
}

static void clear_action_queue(const Drive* env, Entity* agent) {
    float longitudinal = 0.0f, lateral = 0.0f;
    if (env->action_type != 1) {
        longitudinal = env->dynamics_model == CLASSIC ? 3.0f : 2.0f;
        lateral = env->dynamics_model == CLASSIC ? 6.0f : 1.0f;
    }
    agent->action_queue_head = 0;
    for (int i = 0; i < ACTION_QUEUE_CAPACITY; i++) {
        agent->action_queue[i][0] = longitudinal;
        agent->action_queue[i][1] = lateral;
    }
}

static void reset_action_delay(const Drive* env, Entity* agent) {
    // Rejection avoids modulo bias, so each delay from 0 through 3 is equally likely.
    const unsigned int choices = MAX_ACTION_DELAY_STEPS + 1;
    const unsigned int limit = (unsigned int)RAND_MAX - ((unsigned int)RAND_MAX + 1u) % choices;
    unsigned int sample;
    do {
        sample = (unsigned int)rand();
    } while (sample > limit);
    agent->action_delay_steps = sample % choices;
    clear_action_queue(env, agent);
}

static void delay_agent_action(Entity* agent, float action[2]) {
    int head = agent->action_queue_head;
    agent->action_queue[head][0] = action[0];
    agent->action_queue[head][1] = action[1];
    int delayed = (head + ACTION_QUEUE_CAPACITY - agent->action_delay_steps) % ACTION_QUEUE_CAPACITY;
    action[0] = agent->action_queue[delayed][0];
    action[1] = agent->action_queue[delayed][1];
    agent->action_queue_head = (head + 1) % ACTION_QUEUE_CAPACITY;
}

static void record_ego_history(Entity* agent, float longitudinal, float lateral) {
    float* sample = agent->ego_history[agent->ego_history_head];
    sample[0] = agent->x;
    sample[1] = agent->y;
    sample[2] = agent->heading_x;
    sample[3] = agent->heading_y;
    sample[4] = hypotf(agent->vx, agent->vy);
    sample[5] = longitudinal;
    sample[6] = lateral;
    agent->ego_history_head = (agent->ego_history_head + 1) % EGO_HISTORY_CAPACITY;
}

static void reset_ego_history(Entity* agent) {
    agent->ego_history_head = 0;
    // Clamp unavailable history to the initial state, with neutral controls.
    record_ego_history(agent, 0.0f, 0.0f);
    for (int i = 1; i < EGO_HISTORY_CAPACITY; i++)
        memcpy(agent->ego_history[i], agent->ego_history[0], sizeof(agent->ego_history[0]));
    agent->ego_history_head = 0;
}

void set_start_position(Drive* env){
    //InitWindow(800, 600, "GPU Drive");
    //BeginDrawing();
    for(int i = 0; i < env->num_objects; i++){
        int is_active = 0;
        for(int j = 0; j < env->active_agent_count; j++){
            if(env->active_agent_indices[j] == i){
                is_active = 1;
                break;
            }
        }
        Entity* e = &env->entities[i];

        // Clamp init_steps to ensure we don't go out of bounds
        int step = env->init_steps;
        if (step >= e->array_size) step = e->array_size - 1;
        if (step < 0) step = 0;

        e->x = e->traj_x[step];
        e->y = e->traj_y[step];
        e->z = e->traj_z[step];
        e->prev_x = e->x;
        e->prev_y = e->y;

        if(e->type > CYCLIST || e->type == 0){
            continue;
        }
        if(is_active == 0){
            e->collided_before_goal = 0;
            if (e->mark_as_expert == 0) {
                e->vx = 0;
                e->vy = 0;
                e->vz = 0;
            } else {
                e->vx = e->traj_vx[step];
                e->vy = e->traj_vy[step];
                e->vz = e->traj_vz[step];
            }
        } else {
            e->vx = e->traj_vx[step];
            e->vy = e->traj_vy[step];
            e->vz = e->traj_vz[step];
        }
        e->heading = e->traj_heading[step];
        e->heading_x = cosf(e->heading);
        e->heading_y = sinf(e->heading);
        e->valid = e->traj_valid[step];
        e->collision_state = 0;
        e->metrics_array[COLLISION_IDX] = 0.0f; // vehicle collision
        e->metrics_array[OFFROAD_IDX] = 0.0f; // offroad
        e->metrics_array[REACHED_GOAL_IDX] = 0.0f; // reached goal
        e->metrics_array[LANE_ALIGNED_IDX] = 0.0f; // lane aligned
        e->metrics_array[AVG_DISPLACEMENT_ERROR_IDX] = 0.0f; // avg displacement error
        e->metrics_array[LANE_LATERAL_OFFSET_IDX] = -100.0f;
        e->cumulative_displacement = 0.0f;
        e->displacement_sample_count = 0;
        e->respawn_timestep = -1;
        e->respawn_count = 0;

        // Dynamics
        e->a_long = 0.0f;
        e->a_lat = 0.0f;
        e->jerk_long = 0.0f;
        e->jerk_lat = 0.0f;
        e->steering_angle = 0.0f;
        if (e->wheelbase == 0.0f) e->wheelbase = 0.6f * e->length;
        if (is_active) {
            reset_action_delay(env, e);
            reset_ego_history(e);
        }
    }
    reset_exit_orders(env);
    env->eval_traffic_light_violation = 0;
    //EndDrawing();
}

int getGridIndex(Drive* env, float x1, float y1) {
    GridMap* grid_map = env->grid_map;
    if (!isfinite(x1) || !isfinite(y1) ||
            grid_map->grid_cols <= 0 || grid_map->grid_rows <= 0 ||
            grid_map->top_left_x >= grid_map->bottom_right_x ||
            grid_map->bottom_right_y >= grid_map->top_left_y) {
        return -1;  // Invalid grid coordinates
    }

    if (x1 < grid_map->top_left_x || x1 > grid_map->bottom_right_x ||
            y1 < grid_map->bottom_right_y || y1 > grid_map->top_left_y) {
        return -1;  // Return -1 for out of bounds
    }

    float relativeX = x1 - grid_map->top_left_x;  // Distance from left
    float relativeY = y1 - grid_map->bottom_right_y;  // Distance from bottom
    int gridX = (int)floorf(relativeX / GRID_CELL_SIZE);  // Column index
    int gridY = (int)floorf(relativeY / GRID_CELL_SIZE);  // Row index

    // The maximum map coordinate is inclusive. When the map extent is an
    // exact multiple of the cell size it lies on the outer edge of the final
    // cell rather than in a new cell.
    if (gridX == grid_map->grid_cols) gridX--;
    if (gridY == grid_map->grid_rows) gridY--;
    if (gridX < 0 || gridX >= grid_map->grid_cols ||
            gridY < 0 || gridY >= grid_map->grid_rows) {
        return -1;
    }

    int index = (gridY*grid_map->grid_cols) + gridX;
    return index;
}

void add_entity_to_grid(Drive* env, int grid_index, int entity_idx, int geometry_idx, int* cell_entities_insert_index){
    int grid_cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    if(grid_index < 0 || grid_index >= grid_cell_count){
        return;
    }

    int count = cell_entities_insert_index[grid_index];
    if(count >= env->grid_map->cell_entities_count[grid_index]) {
        printf("Error: Exceeded precomputed entity count for grid cell %d. Current count: %d, Max count(Precomputed): %d\n", grid_index, count, env->grid_map->cell_entities_count[grid_index]);
        return;
    }

    env->grid_map->cells[grid_index][count].entity_idx = entity_idx;
    env->grid_map->cells[grid_index][count].geometry_idx = geometry_idx;
    cell_entities_insert_index[grid_index] = count + 1;
}


void init_topology_graph(Drive* env){
    // Count ROAD_LANE entities
    int road_lane_count = 0;
    for(int i = 0; i < env->num_roads; i++){
        if(env->static_road_entities[i].type == ROAD_LANE){
            road_lane_count++;
        }
    }

    if(road_lane_count == 0){
        env->topology_graph = NULL;
        return;
    }

    // Graph vertices use road-local indices (only ROAD_LANE vertices have edges).
    env->topology_graph = createGraph(env->num_roads);

    // Connect ROAD_LANE entities based on geometric connectivity
    for(int i = 0; i < env->num_roads; i++){
        if(env->static_road_entities[i].type != ROAD_LANE) continue;

        const StaticRoadEntity* lane_i = &env->static_road_entities[i];
        if(lane_i->array_size < 2) continue; // Need at least 2 points

        // Get end point of current lane
        float end_x = lane_i->traj_x[lane_i->array_size - 1];
        float end_y = lane_i->traj_y[lane_i->array_size - 1];
        float end_vector_x = lane_i->traj_x[lane_i->array_size - 1] - lane_i->traj_x[lane_i->array_size - 2];
        float end_vector_y = lane_i->traj_y[lane_i->array_size - 1] - lane_i->traj_y[lane_i->array_size - 2];
        float end_heading = atan2f(end_vector_y, end_vector_x);

        // Find lanes that start near this lane's end
        for(int j = 0; j < env->num_roads; j++){
            if(i == j || env->static_road_entities[j].type != ROAD_LANE) continue;

            const StaticRoadEntity* lane_j = &env->static_road_entities[j];
            if(lane_j->array_size < 2) continue;

            // Get start point of potential next lane
            float start_x = lane_j->traj_x[0];
            float start_y = lane_j->traj_y[0];
            float start_vector_x = lane_j->traj_x[1] - lane_j->traj_x[0];
            float start_vector_y = lane_j->traj_y[1] - lane_j->traj_y[0];
            float start_heading = atan2f(start_vector_y, start_vector_x);

            // Check if end of lane_i is close to start of lane_j
            float distance = relative_distance_2d(end_x, end_y, start_x, start_y);
            float heading_diff = fabsf(end_heading - start_heading);

            // Lane connectivity thresholds:
            // - 0.01m distance: lanes must connect within 1cm (very strict for clean topology)
            // - 0.1 (~5.7 degrees) heading difference: allow slight curves
            if(distance < 0.01f && heading_diff < 0.1f){
                // Add directed edge from i to j (lane i connects to lane j)
                struct AdjListNode* node = newAdjListNode(j);
                node->next = env->topology_graph->array[i];
                env->topology_graph->array[i] = node;
            }
        }
    }
}

void init_grid_map(Drive* env){
    // Allocate memory for the grid map structure
    env->grid_map = (GridMap*)calloc(1, sizeof(GridMap));

    // Find top left and bottom right points of the map
    float top_left_x = 0.0f;
    float top_left_y = 0.0f;
    float bottom_right_x = 0.0f;
    float bottom_right_y = 0.0f;
    int first_valid_point = 0;
    for(int i = 0; i < env->num_roads; i++){
        if(grid_map_entity_type(env->static_road_entities[i].type)){
            // Check all points in the trajectory for road elements
            const StaticRoadEntity* e = &env->static_road_entities[i];
            for(int j = 0; j < e->array_size; j++){
                if(!isfinite(e->traj_x[j]) || !isfinite(e->traj_y[j])) continue;
                if(e->traj_x[j] == INVALID_POSITION) continue;
                if(e->traj_y[j] == INVALID_POSITION) continue;
                if(!first_valid_point) {
                    top_left_x = bottom_right_x = e->traj_x[j];
                    top_left_y = bottom_right_y = e->traj_y[j];
                    first_valid_point = true;
                    continue;
                }
                if(e->traj_x[j] < top_left_x) top_left_x = e->traj_x[j];
                if(e->traj_x[j] > bottom_right_x) bottom_right_x = e->traj_x[j];
                if(e->traj_y[j] > top_left_y) top_left_y = e->traj_y[j];
                if(e->traj_y[j] < bottom_right_y) bottom_right_y = e->traj_y[j];
            }
        }
    }

    // Empty maps have an empty grid; straight roads still need one cell on
    // their zero-width axis. This also avoids zero-sized grid allocations.
    if (!first_valid_point) return;
    if (top_left_x == bottom_right_x) bottom_right_x += GRID_CELL_SIZE;
    if (bottom_right_y == top_left_y) top_left_y += GRID_CELL_SIZE;

    env->grid_map->top_left_x = top_left_x;
    env->grid_map->top_left_y = top_left_y;
    env->grid_map->bottom_right_x = bottom_right_x;
    env->grid_map->bottom_right_y = bottom_right_y;
    env->grid_map->cell_size_x = GRID_CELL_SIZE;
    env->grid_map->cell_size_y = GRID_CELL_SIZE;

    // Calculate grid dimensions
    float grid_width = bottom_right_x - top_left_x;
    float grid_height = top_left_y - bottom_right_y;
    env->grid_map->grid_cols = ceil(grid_width / GRID_CELL_SIZE);
    env->grid_map->grid_rows = ceil(grid_height / GRID_CELL_SIZE);
    int grid_cell_count = env->grid_map->grid_cols*env->grid_map->grid_rows;
    env->grid_map->cells = (GridMapEntity**)calloc(grid_cell_count, sizeof(GridMapEntity*));
    env->grid_map->cell_entities_count = (int*)calloc(grid_cell_count, sizeof(int));

    // Calculate number of entities in each grid cell
    for(int i = 0; i < env->num_roads; i++){
        if(grid_map_entity_type(env->static_road_entities[i].type)){
            for(int j = 0; j < env->static_road_entities[i].array_size - 1; j++){
                float x_center = (env->static_road_entities[i].traj_x[j] + env->static_road_entities[i].traj_x[j+1]) / 2;
                float y_center = (env->static_road_entities[i].traj_y[j] + env->static_road_entities[i].traj_y[j+1]) / 2;
                int grid_index = getGridIndex(env, x_center, y_center);
                if (grid_index < 0 || grid_index >= grid_cell_count) continue;
                env->grid_map->cell_entities_count[grid_index]++;
            }
        }
    }
    // The shared map can cover a whole city; keep this scratch array off the stack.
    int* cell_entities_insert_index = (int*)calloc(grid_cell_count, sizeof(int));

    // Initialize grid cells
    for(int grid_index = 0; grid_index < grid_cell_count; grid_index++){
        int count = env->grid_map->cell_entities_count[grid_index];
        if (count > 0) env->grid_map->cells[grid_index] = (GridMapEntity*)calloc(count, sizeof(GridMapEntity));
    }

    // Populate grid cells
    for(int i = 0; i < env->num_roads; i++){
        if(grid_map_entity_type(env->static_road_entities[i].type)){         // NOTE: Road edges, lines, lanes, and stop lines in grid map
            for(int j = 0; j < env->static_road_entities[i].array_size - 1; j++){
                float x_center = (env->static_road_entities[i].traj_x[j] + env->static_road_entities[i].traj_x[j+1]) / 2;
                float y_center = (env->static_road_entities[i].traj_y[j] + env->static_road_entities[i].traj_y[j+1]) / 2;
                int grid_index = getGridIndex(env, x_center, y_center);
                add_entity_to_grid(env, grid_index, i, j, cell_entities_insert_index);
            }
        }
    }
    free(cell_entities_insert_index);
}

void init_neighbor_offsets(Drive* env) {
    // Allocate memory for the offsets
    env->neighbor_offsets = (int*)calloc(env->grid_map->vision_range*env->grid_map->vision_range*2, sizeof(int));
    // neighbor offsets in a spiral pattern
    int dx[] = {1, 0, -1, 0};
    int dy[] = {0, 1, 0, -1};
    int x = 0;    // Current x offset
    int y = 0;    // Current y offset
    int dir = 0;  // Current direction (0: right, 1: up, 2: left, 3: down)
    int steps_to_take = 1; // Number of steps in current direction
    int steps_taken = 0;   // Steps taken in current direction
    int segments_completed = 0; // Count of direction segments completed
    int total = 0; // Total offsets added
    int max_offsets = env->grid_map->vision_range*env->grid_map->vision_range;
    // Start at center (0,0)
    int curr_idx = 0;
    env->neighbor_offsets[curr_idx++] = 0;  // x offset
    env->neighbor_offsets[curr_idx++] = 0;  // y offset
    total++;
    // Generate spiral pattern
    while (total < max_offsets) {
        // Move in current direction
        x += dx[dir];
        y += dy[dir];
        // Only add if within vision range bounds
        if (abs(x) <= env->grid_map->vision_range/2 && abs(y) <= env->grid_map->vision_range/2) {
            env->neighbor_offsets[curr_idx++] = x;
            env->neighbor_offsets[curr_idx++] = y;
            total++;
        }
        steps_taken++;
        // Check if we need to change direction
        if(steps_taken != steps_to_take) continue;
        steps_taken = 0;  // Reset steps taken
        dir = (dir + 1) % 4;  // Change direction (clockwise: right->up->left->down)
        segments_completed++;
        // Increase step length every two direction changes
        if (segments_completed % 2 == 0) {
            steps_to_take++;
        }
    }
}

// Takes ownership of roads and their coordinate arrays. The initial reference
// belongs to the caller; environments acquire additional references on attach.
static StaticRoadMap* create_static_road_map(StaticRoadEntity* roads, int num_roads) {
    StaticRoadMap* map = (StaticRoadMap*)calloc(1, sizeof(StaticRoadMap));
    map->ref_count = 1;
    map->entities = roads;
    map->num_roads = num_roads;
    for (int i = 0; i < num_roads; i++) {
        if (roads[i].type == TRAFFIC_LIGHT_STOP_LINE && roads[i].traffic_light_idx >= map->num_traffic_lights) {
            map->num_traffic_lights = roads[i].traffic_light_idx + 1;
        }
    }
    Drive builder = {0};
    builder.static_road_entities = roads;
    builder.num_roads = num_roads;
    init_grid_map(&builder);
    builder.grid_map->vision_range = 21;
    init_neighbor_offsets(&builder);
    map->grid_map = builder.grid_map;
    map->neighbor_offsets = builder.neighbor_offsets;
    return map;
}

void move_expert(Drive* env, float* actions, int agent_idx){
    Entity* agent = &env->entities[agent_idx];
    agent->prev_x = agent->x;
    agent->prev_y = agent->y;
    int t = env->timestep;
    if (t < 0 || t >= agent->array_size || (agent->traj_valid && agent->traj_valid[t] == 0)) {
        if (env->dynamic_map_generation) return;
        agent->x = INVALID_POSITION;
        agent->y = INVALID_POSITION;
        agent->z = 0.0f;
        agent->vx = 0.0f;
        agent->vy = 0.0f;
        agent->vz = 0.0f;
        agent->heading = 0.0f;
        agent->heading_x = 1.0f;
        agent->heading_y = 0.0f;
        return;
    }
    agent->x = agent->traj_x[t];
    agent->y = agent->traj_y[t];
    agent->z = agent->traj_z[t];
    agent->vx = agent->traj_vx[t];
    agent->vy = agent->traj_vy[t];
    agent->vz = agent->traj_vz[t];
    agent->heading = agent->traj_heading[t];
    agent->heading_x = cosf(agent->heading);
    agent->heading_y = sinf(agent->heading);
}

bool check_line_intersection(float p1[2], float p2[2], float q1[2], float q2[2]) {
    if (fmax(p1[0], p2[0]) < fmin(q1[0], q2[0]) || fmin(p1[0], p2[0]) > fmax(q1[0], q2[0]) ||
        fmax(p1[1], p2[1]) < fmin(q1[1], q2[1]) || fmin(p1[1], p2[1]) > fmax(q1[1], q2[1]))
        return false;

    // Calculate vectors
    float dx1 = p2[0] - p1[0];
    float dy1 = p2[1] - p1[1];
    float dx2 = q2[0] - q1[0];
    float dy2 = q2[1] - q1[1];

    // Calculate cross products
    float cross = dx1 * dy2 - dy1 * dx2;

    // If lines are parallel
    if (cross == 0) return false;

    // Calculate relative vectors between start points
    float dx3 = p1[0] - q1[0];
    float dy3 = p1[1] - q1[1];

    // Calculate parameters for intersection point
    float s = (dx1 * dy3 - dy1 * dx3) / cross;
    float t = (dx2 * dy3 - dy2 * dx3) / cross;

    // Check if intersection point lies within both line segments
    return (s >= 0 && s <= 1 && t >= 0 && t <= 1);
}

int checkNeighbors(Drive* env, float x, float y, GridMapEntity* entity_list, int max_size, const int (*local_offsets)[2], int offset_size) {
    // Get the grid index for the given position (x, y)
    int index = getGridIndex(env, x, y);
    if (index == -1) return 0;  // Return 0 size if position invalid
    // Calculate 2D grid coordinates
    int cellsX = env->grid_map->grid_cols;
    int gridX = index % cellsX;
    int gridY = index / cellsX;
    int entity_list_count = 0;
    // Fill the provided array
    for (int i = 0; i < offset_size; i++) {
        int nx = gridX + local_offsets[i][0];
        int ny = gridY + local_offsets[i][1];
        // Ensure the neighbor is within grid bounds
        if(nx < 0 || nx >= env->grid_map->grid_cols || ny < 0 || ny >= env->grid_map->grid_rows) continue;
        int neighborIndex = ny * env->grid_map->grid_cols + nx;
        int count = env->grid_map->cell_entities_count[neighborIndex];
        // Add entities from this cell to the list
        for (int j = 0; j < count && entity_list_count < max_size; j++) {
            int entityId = env->grid_map->cells[neighborIndex][j].entity_idx;
            int geometry_idx = env->grid_map->cells[neighborIndex][j].geometry_idx;
            entity_list[entity_list_count].entity_idx = entityId;
            entity_list[entity_list_count].geometry_idx = geometry_idx;
            entity_list_count += 1;
        }
    }
    return entity_list_count;
}

int check_aabb_collision(Entity* car1, Entity* car2) {
    // Get car corners in world space
    float cos1 = car1->heading_x;
    float sin1 = car1->heading_y;
    float cos2 = car2->heading_x;
    float sin2 = car2->heading_y;

    // Calculate half dimensions
    float half_len1 = car1->length * 0.5f;
    float half_width1 = car1->width * 0.5f;
    float half_len2 = car2->length * 0.5f;
    float half_width2 = car2->width * 0.5f;

    // Calculate car1's corners in world space
    float car1_corners[4][2] = {
        {car1->x + (half_len1 * cos1 - half_width1 * sin1), car1->y + (half_len1 * sin1 + half_width1 * cos1)},
        {car1->x + (half_len1 * cos1 + half_width1 * sin1), car1->y + (half_len1 * sin1 - half_width1 * cos1)},
        {car1->x + (-half_len1 * cos1 - half_width1 * sin1), car1->y + (-half_len1 * sin1 + half_width1 * cos1)},
        {car1->x + (-half_len1 * cos1 + half_width1 * sin1), car1->y + (-half_len1 * sin1 - half_width1 * cos1)}
    };

    // Calculate car2's corners in world space
    float car2_corners[4][2] = {
        {car2->x + (half_len2 * cos2 - half_width2 * sin2), car2->y + (half_len2 * sin2 + half_width2 * cos2)},
        {car2->x + (half_len2 * cos2 + half_width2 * sin2), car2->y + (half_len2 * sin2 - half_width2 * cos2)},
        {car2->x + (-half_len2 * cos2 - half_width2 * sin2), car2->y + (-half_len2 * sin2 + half_width2 * cos2)},
        {car2->x + (-half_len2 * cos2 + half_width2 * sin2), car2->y + (-half_len2 * sin2 - half_width2 * cos2)}
    };

    // Get the axes to check (normalized vectors perpendicular to each edge)
    float axes[4][2] = {
        {cos1, sin1},           // Car1's length axis
        {-sin1, cos1},          // Car1's width axis
        {cos2, sin2},           // Car2's length axis
        {-sin2, cos2}           // Car2's width axis
    };

    // Check each axis
    for(int i = 0; i < 4; i++) {
        float min1 = INFINITY, max1 = -INFINITY;
        float min2 = INFINITY, max2 = -INFINITY;

        // Project car1's corners onto the axis
        for(int j = 0; j < 4; j++) {
            float proj = car1_corners[j][0] * axes[i][0] + car1_corners[j][1] * axes[i][1];
            min1 = fminf(min1, proj);
            max1 = fmaxf(max1, proj);
        }

        // Project car2's corners onto the axis
        for(int j = 0; j < 4; j++) {
            float proj = car2_corners[j][0] * axes[i][0] + car2_corners[j][1] * axes[i][1];
            min2 = fminf(min2, proj);
            max2 = fmaxf(max2, proj);
        }

        // If there's a gap on this axis, the boxes don't intersect
        if(max1 < min2 || min1 > max2) {
            return 0;  // No collision
        }
    }

    // If we get here, there's no separating axis, so the boxes intersect
    return 1;  // Collision
}

int collision_check(Drive* env, int agent_idx) {
    Entity* agent = &env->entities[agent_idx];

    if(agent->x == INVALID_POSITION ) return -1;

    int car_collided_with_index = -1;

    if (agent->respawn_timestep != -1) return car_collided_with_index; // Skip respawning entities

    for(int i = 0; i < env->num_controllable_agents; i++){
        int index = -1;
        if(i < env->active_agent_count){
            index = env->active_agent_indices[i];
        } else if (i < env->num_controllable_agents){
            index = env->static_car_indices[i - env->active_agent_count];
        }
        if(index == -1) continue;
        if(index == agent_idx) continue;
        Entity* entity = &env->entities[index];
        if (entity->respawn_timestep != -1) continue; // Skip respawning entities
        float x1 = entity->x;
        float y1 = entity->y;
        float dist = ((x1 - agent->x)*(x1 - agent->x) + (y1 - agent->y)*(y1 - agent->y));
        if(dist > 225.0f) continue;
        if(check_aabb_collision(agent, entity)) {
            car_collided_with_index = index;
            break;
        }
    }

    return car_collided_with_index;
}

// Continuous separating-axis test for translating oriented rectangles. Heading
// and world-space velocity stay fixed; return first body contact, in seconds.
static float vehicle_time_to_collision(const Entity* ego, const Entity* other) {
    float dx = other->x - ego->x, dy = other->y - ego->y;
    float vx = other->vx - ego->vx, vy = other->vy - ego->vy;
    const float axes[4][2] = {
        {ego->heading_x, ego->heading_y}, {-ego->heading_y, ego->heading_x},
        {other->heading_x, other->heading_y}, {-other->heading_y, other->heading_x}
    };
    float enter = 0.0f, leave = SAFETY_TIME_SECONDS;
    for (int i = 0; i < 4; i++) {
        float ax = axes[i][0], ay = axes[i][1];
        float extent = 0.5f * (
            ego->length * fabsf(ax * ego->heading_x + ay * ego->heading_y) +
            ego->width * fabsf(-ax * ego->heading_y + ay * ego->heading_x) +
            other->length * fabsf(ax * other->heading_x + ay * other->heading_y) +
            other->width * fabsf(-ax * other->heading_y + ay * other->heading_x));
        float position = dx * ax + dy * ay;
        float velocity = vx * ax + vy * ay;
        if (fabsf(velocity) < 1e-6f) {
            if (fabsf(position) > extent) return INFINITY;
            continue;
        }
        float first = (-extent - position) / velocity;
        float last = (extent - position) / velocity;
        enter = fmaxf(enter, fminf(first, last));
        leave = fminf(leave, fmaxf(first, last));
        if (enter > leave) return INFINITY;
    }
    return enter;
}

static int vehicle_approaches_from_behind(const Entity* ego, const Entity* other) {
    // Similar headings: within 45 degrees. Also require the other center to
    // lie in the rear 90-degree cone, so crossing/side approaches still count.
    float heading_dot = ego->heading_x * other->heading_x + ego->heading_y * other->heading_y;
    float dx = other->x - ego->x, dy = other->y - ego->y;
    float longitudinal = dx * ego->heading_x + dy * ego->heading_y;
    float lateral = -dx * ego->heading_y + dy * ego->heading_x;
    return heading_dot >= 0.70710678f && longitudinal < 0.0f && fabsf(lateral) <= -longitudinal;
}

static float point_vehicle_distance_squared(float x, float y, const Entity* vehicle) {
    float dx = x - vehicle->x, dy = y - vehicle->y;
    float longitudinal = fmaxf(fabsf(dx * vehicle->heading_x + dy * vehicle->heading_y)
                               - 0.5f * vehicle->length, 0.0f);
    float lateral = fmaxf(fabsf(-dx * vehicle->heading_y + dy * vehicle->heading_x)
                          - 0.5f * vehicle->width, 0.0f);
    return longitudinal * longitudinal + lateral * lateral;
}

static float vehicle_clearance_squared(Entity* ego, Entity* other) {
    if (check_aabb_collision(ego, other)) return 0.0f;
    // For disjoint rectangles the closest pair includes a corner of at least
    // one rectangle. Check both directions for exact edge/corner clearance.
    float distance = INFINITY;
    for (int side = 0; side < 2; side++) {
        const Entity* from = side ? other : ego;
        const Entity* to = side ? ego : other;
        for (int corner = 0; corner < 4; corner++) {
            float x = offsets[corner][0] * 0.5f * from->length;
            float y = offsets[corner][1] * 0.5f * from->width;
            float world_x = from->x + x * from->heading_x - y * from->heading_y;
            float world_y = from->y + x * from->heading_y + y * from->heading_x;
            distance = fminf(distance, point_vehicle_distance_squared(world_x, world_y, to));
        }
    }
    return distance;
}

int check_lane_aligned(Entity* car, const StaticRoadEntity* lane, int geometry_idx) {
    // Validate lane geometry length
    if (!lane || lane->array_size < 2) {
        car->lane_heading_angle_diff = 0.0f;
        return 0;
    }

    // Clamp geometry index to valid segment range [0, array_size-2]
    if (geometry_idx < 0) geometry_idx = 0;
    if (geometry_idx >= lane->array_size - 1) geometry_idx = lane->array_size - 2;

    // Compute local lane segment heading
    float heading_x1, heading_y1;
    if (geometry_idx > 0) {
        heading_x1 = lane->traj_x[geometry_idx] - lane->traj_x[geometry_idx - 1];
        heading_y1 = lane->traj_y[geometry_idx] - lane->traj_y[geometry_idx - 1];
    } else {
        // For first segment, just use the forward direction
        heading_x1 = lane->traj_x[geometry_idx + 1] - lane->traj_x[geometry_idx];
        heading_y1 = lane->traj_y[geometry_idx + 1] - lane->traj_y[geometry_idx];
    }

    float heading_x2 = lane->traj_x[geometry_idx + 1] - lane->traj_x[geometry_idx];
    float heading_y2 = lane->traj_y[geometry_idx + 1] - lane->traj_y[geometry_idx];

    float heading_1 = atan2f(heading_y1, heading_x1);
    float heading_2 = atan2f(heading_y2, heading_x2);
    float heading = (heading_1 + heading_2) / 2.0f;

    // Normalize to [-pi, pi]
    if (heading > M_PI) heading -= 2.0f * M_PI;
    if (heading < -M_PI) heading += 2.0f * M_PI;

    // Compute heading difference
    float car_heading = car->heading; // radians
    float heading_diff = fabsf(car_heading - heading);

    while (heading_diff >= 2.0f * M_PI) heading_diff -= 2.0f * M_PI;

    if (heading_diff > M_PI) heading_diff = 2.0f * M_PI - heading_diff;

    car->lane_heading_angle_diff = heading_diff;

    // within 15 degrees
    return (heading_diff < (M_PI / 12.0f)) ? 1 : 0;
}

float compute_lane_lateral_offset(Entity* car, const StaticRoadEntity* lane, int geometry_idx) {
    if (lane->array_size < 3) return -100.0f;

    if (geometry_idx == 0) {
        geometry_idx++;
    } else if (geometry_idx == lane->array_size - 1) {
        geometry_idx--;
    }

    float lane_dx = lane->traj_x[geometry_idx + 1] - lane->traj_x[geometry_idx - 1];
    float lane_dy = lane->traj_y[geometry_idx + 1] - lane->traj_y[geometry_idx - 1];
    float lane_len = sqrtf(lane_dx * lane_dx + lane_dy * lane_dy);
    if (lane_len < 1e-9f) return -100.0f;

    float car_dx = car->x - lane->traj_x[geometry_idx - 1];
    float car_dy = car->y - lane->traj_y[geometry_idx - 1];

    return (car_dx * lane_dy - car_dy * lane_dx) / lane_len;
}

void reset_agent_metrics(Drive* env, int agent_idx){
    Entity* agent = &env->entities[agent_idx];
    agent->metrics_array[COLLISION_IDX] = 0.0f; // vehicle collision
    agent->metrics_array[OFFROAD_IDX] = 0.0f; // offroad
    agent->metrics_array[LANE_ALIGNED_IDX] = 0.0f; // lane aligned
    agent->metrics_array[AVG_DISPLACEMENT_ERROR_IDX] = 0.0f;
    agent->metrics_array[LANE_LATERAL_OFFSET_IDX] = -100.0f;
    agent->collision_state = 0;
}

float point_to_segment_distance_2d(float px, float py, float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;

    if (dx == 0 && dy == 0) {
        // The segment is a point
        return sqrtf((px - x1) * (px - x1) + (py - y1) * (py - y1));
    }

    // Calculate the t that minimizes the distance
    float t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy);

    // Clamp t to the segment
    if (t < 0) t = 0;
    else if (t > 1) t = 1;

    // Find the closest point on the segment
    float closestX = x1 + t * dx;
    float closestY = y1 + t * dy;

    // Return the distance from p to the closest point
    return sqrtf((px - closestX) * (px - closestX) + (py - closestY) * (py - closestY));
}

static int stop_line_overlaps_vehicle(const StaticRoadEntity* line, const Entity* vehicle,
                                      float x, float y, float cosine, float sine) {
    float half_length = vehicle->length * 0.5f;
    float half_width = vehicle->width * 0.5f;
    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        corners[i][0] = offsets[i][0] * half_length;
        corners[i][1] = offsets[i][1] * half_width;
    }
    float start[2];
    for (int i = 0; i < line->array_size; i++) {
        float dx = line->traj_x[i] - x, dy = line->traj_y[i] - y;
        float end[2] = {dx * cosine + dy * sine, -dx * sine + dy * cosine};
        // A short segment can be wholly inside the box without crossing an edge.
        if (fabsf(end[0]) <= half_length && fabsf(end[1]) <= half_width) return 1;
        if (i > 0) {
            for (int k = 0; k < 4; k++) {
                if (check_line_intersection(corners[k], corners[(k + 1) % 4], start, end)) return 1;
            }
        }
        start[0] = end[0];
        start[1] = end[1];
    }
    return 0;
}

// Evaluation-only geometry: never enters observations, collisions or rewards.
static void update_eval_traffic_light_lines(Drive* env) {
    if (env->eval_traffic_light_violation || !env->traffic_lights_enabled) return;
    for (int i = 0; i < env->num_eval_traffic_light_lines; i++) {
        EvalTrafficLightLine* hidden = &env->eval_traffic_light_lines[i];
        StaticRoadEntity line = {.array_size = 2, .traj_x = hidden->x, .traj_y = hidden->y,
                                .traffic_light_idx = hidden->traffic_light_idx};
        if (road_traffic_light_color(env, &line) != TRAFFIC_LIGHT_RED) continue;
        for (int idx = 0; idx < env->num_objects; idx++) {
            const Entity* e = &env->entities[idx];
            if (e->type != VEHICLE || (!e->active_agent && !e->mark_as_expert) ||
                e->x == INVALID_POSITION || e->prev_x == INVALID_POSITION) continue;
            int previous = (e->ego_history_head + EGO_HISTORY_CAPACITY - 1) % EGO_HISTORY_CAPACITY;
            float cosine = e->ego_history[previous][2], sine = e->ego_history[previous][3];
            if (!e->active_agent) {
                int t = env->timestep;
                if (t <= 0 || t >= e->array_size || !e->traj_valid[t - 1] || !e->traj_valid[t]) continue;
                cosine = cosf(e->traj_heading[t - 1]);
                sine = sinf(e->traj_heading[t - 1]);
            }
            if (stop_line_overlaps_vehicle(&line, e, e->x, e->y, cosf(e->heading), sinf(e->heading)) &&
                !stop_line_overlaps_vehicle(&line, e, e->prev_x, e->prev_y, cosine, sine)) {
                env->eval_traffic_light_violation = 1;
                return;
            }
        }
    }
}

float normalize_heading(float heading){
    if(heading > M_PI) heading -= 2*M_PI;
    if(heading < -M_PI) heading += 2*M_PI;
    return heading;
}

void compute_agent_metrics(Drive* env, int agent_idx) {
    Entity* agent = &env->entities[agent_idx];

    reset_agent_metrics(env, agent_idx);

    if(agent->x == INVALID_POSITION ) return; // invalid agent position

    // Compute displacement error
    float displacement_error = compute_displacement_error(agent, env->timestep);

    if (displacement_error > 0.0f) { // Only count valid displacements
        agent->cumulative_displacement += displacement_error;
        agent->displacement_sample_count++;

        // Compute running average
        agent->metrics_array[AVG_DISPLACEMENT_ERROR_IDX] =
            agent->cumulative_displacement / agent->displacement_sample_count;
    }

    int collided = 0;
    float half_length = agent->length/2.0f;
    float half_width = agent->width/2.0f;
    float cos_heading = cosf(agent->heading);
    float sin_heading = sinf(agent->heading);
    int previous_index = (agent->ego_history_head + EGO_HISTORY_CAPACITY - 1) % EGO_HISTORY_CAPACITY;
    const float* previous_pose = agent->ego_history[previous_index];
    float min_distance = (float)INT16_MAX;

    int closest_lane_entity_idx = -1;
    int closest_lane_geometry_idx = -1;

    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        corners[i][0] = agent->x + (offsets[i][0]*half_length*cos_heading - offsets[i][1]*half_width*sin_heading);
        corners[i][1] = agent->y + (offsets[i][0]*half_length*sin_heading + offsets[i][1]*half_width*cos_heading);
    }

    GridMapEntity entity_list[MAX_ENTITIES_PER_CELL*25];  // Array big enough for all neighboring cells
    int list_size = checkNeighbors(env, agent->x, agent->y, entity_list, MAX_ENTITIES_PER_CELL*25, collision_offsets, 25);
    for (int i = 0; i < list_size ; i++) {
        if(entity_list[i].entity_idx == -1) continue;
        const StaticRoadEntity* entity = &env->static_road_entities[entity_list[i].entity_idx];

        if (env->traffic_lights_enabled && entity->type == TRAFFIC_LIGHT_STOP_LINE && road_traffic_light_color(env, entity) == TRAFFIC_LIGHT_RED) {
            // Test entry into the whole line, regardless of its previous color.
            if (stop_line_overlaps_vehicle(entity, agent, agent->x, agent->y, cos_heading, sin_heading) &&
                !stop_line_overlaps_vehicle(entity, agent, agent->prev_x, agent->prev_y,
                                           previous_pose[2], previous_pose[3])) {
                collided = RED_TRAFFIC_LIGHT_CROSSED;
            }
        }

        // Check for offroad collision with road edges
        if(entity->type == ROAD_EDGE && collided != OFFROAD) {
            int geometry_idx = entity_list[i].geometry_idx;
            float start[2] = {entity->traj_x[geometry_idx], entity->traj_y[geometry_idx]};
            float end[2] = {entity->traj_x[geometry_idx + 1], entity->traj_y[geometry_idx + 1]};
            for (int k = 0; k < 4; k++) { // Check each edge of the bounding box
                int next = (k + 1) % 4;
                if (check_line_intersection(corners[k], corners[next], start, end)) {
                    collided = OFFROAD;
                    break;
                }
            }

            if (collided == OFFROAD) {
                float road_x = entity->traj_x[geometry_idx + 1] - entity->traj_x[geometry_idx];
                float road_y = entity->traj_y[geometry_idx + 1] - entity->traj_y[geometry_idx];
                float road_len = sqrtf(road_x * road_x + road_y * road_y);

                road_x /= road_len;
                road_y /= road_len;

                float road_agent_dot = road_x * agent->heading_x + road_y * agent->heading_y;
                if (road_agent_dot < 0) {
                    road_x *= -1;
                    road_y *= -1;
                }

                float road_normal_x = road_y;
                float road_normal_y = -road_x;
                if (road_normal_x * agent->heading_x + road_normal_y * agent->heading_y > 0) {
                    road_normal_x *= -1;
                    road_normal_y *= -1;
                }

                agent->x = agent->prev_x + road_normal_x * 0.2f;
                agent->y = agent->prev_y + road_normal_y * 0.2f;

                agent->heading_x = road_x;
                agent->heading_y = road_y;
                agent->heading = normalize_heading(atan2f(road_y, road_x));
                float speed = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
                agent->vx = speed * agent->heading_x;
                agent->vy = speed * agent->heading_y;
            }
        }

        if (collided == OFFROAD) break;

        // Find closest point on the road centerline to the agent
        if(entity->type == ROAD_LANE) {
            int entity_idx = entity_list[i].entity_idx;
            int geometry_idx = entity_list[i].geometry_idx;

            float start[2] = {entity->traj_x[geometry_idx], entity->traj_y[geometry_idx]};
            float end[2] = {entity->traj_x[geometry_idx + 1], entity->traj_y[geometry_idx + 1]};

            float dist = point_to_segment_distance_2d(agent->x, agent->y, start[0], start[1], end[0], end[1]);
            float heading_diff = fabsf(atan2f(end[1]-start[1], end[0]-start[0]) - agent->heading);

            // Normalize heading difference to [0, pi]
            if (heading_diff > M_PI) heading_diff = 2.0f * M_PI - heading_diff;

            // Penalize if heading differs by more than 30 degrees
            if (heading_diff > (M_PI / 6.0f)) dist += 3.0f;

            if (dist < min_distance) {
                min_distance = dist;
                closest_lane_entity_idx = entity_idx;
                closest_lane_geometry_idx = geometry_idx;
            }
        }
    }

    // check if aligned with closest lane and set current lane
    // 4.0m threshold: agents more than 4 meters from any lane are considered off-road
    if (min_distance > 4.0f || closest_lane_entity_idx == -1) {
        agent->metrics_array[LANE_ALIGNED_IDX] = 0.0f;
        agent->metrics_array[LANE_LATERAL_OFFSET_IDX] = -100.0f;
        agent->current_lane_idx = -1;
    } else {
        agent->current_lane_idx = closest_lane_entity_idx;

        int lane_aligned = check_lane_aligned(agent, &env->static_road_entities[closest_lane_entity_idx], closest_lane_geometry_idx);
        agent->metrics_array[LANE_ALIGNED_IDX] = lane_aligned;
        agent->metrics_array[LANE_LATERAL_OFFSET_IDX] = compute_lane_lateral_offset(agent, &env->static_road_entities[closest_lane_entity_idx], closest_lane_geometry_idx);
    }

    // Check for vehicle collisions
    int car_collided_with_index = collision_check(env, agent_idx);
    if (car_collided_with_index != -1) {
        collided = VEHICLE_COLLISION;
        agent->collided_with_agent_idx = car_collided_with_index;
    }

    agent->collision_state = collided;

    return;
}

int valid_active_agent(Drive* env, int agent_idx){
    if (env->entities[agent_idx].type != VEHICLE && !env->control_non_vehicles) return 0;
    if (env->entities[agent_idx].traj_valid[env->init_steps] != 1) return 0;
    float cos_heading = cosf(env->entities[agent_idx].traj_heading[0]);
    float sin_heading = sinf(env->entities[agent_idx].traj_heading[0]);
    float goal_x = env->entities[agent_idx].goal_position_x - env->entities[agent_idx].traj_x[0];
    float goal_y = env->entities[agent_idx].goal_position_y - env->entities[agent_idx].traj_y[0];
    // Rotate to ego vehicle's frame
    float rel_goal_x = goal_x*cos_heading + goal_y*sin_heading;
    float rel_goal_y = -goal_x*sin_heading + goal_y*cos_heading;
    float distance_to_goal = relative_distance_2d(0, 0, rel_goal_x, rel_goal_y);
    if(distance_to_goal >= MIN_DISTANCE_TO_GOAL && env->entities[agent_idx].mark_as_expert == 0 && env->active_agent_count < env->num_agents){
        return distance_to_goal;
    }
    return 0;
}

void set_active_agents(Drive* env){
    const char* map_name = env->map_name ? env->map_name : "(unset-map)";

    int capacity = env->num_agents;
    if (capacity < 0) {
        capacity = 0;
    } else if (capacity > MAX_AGENTS) {
        capacity = MAX_AGENTS;
    }

    env->active_agent_count = 0;
    env->static_car_count = 0;
    env->num_controllable_agents = 1;
    env->expert_static_car_count = 0;
    int active_agent_indices[MAX_AGENTS];
    int static_car_indices[MAX_AGENTS];
    int expert_static_car_indices[MAX_AGENTS];

    if (env->control_all_agents == 1) {
        SelectionBuckets b;
        scan_vehicles_initial(env, &b, 1);

        int desired = b.candidates_count;
        if (desired > MAX_AGENTS) desired = MAX_AGENTS;
        if (desired > capacity) desired = capacity;

        if (desired <= 0) {
            goto legacy_select;
        }

        if (!env->deterministic_agent_selection) {
            fisher_yates_shuffle(b.candidates, b.candidates_count);
        }

        for (int k = 0; k < desired; k++) {
            active_agent_indices[env->active_agent_count++] = b.candidates[k];
            env->entities[b.candidates[k]].active_agent = 1;
        }
        for (int i = 0; i < b.statics_count && env->static_car_count < MAX_AGENTS; i++) {
            static_car_indices[env->static_car_count++] = b.statics[i];
        }
        for (int k = desired; k < b.candidates_count && env->static_car_count < MAX_AGENTS; k++) {
            static_car_indices[env->static_car_count++] = b.candidates[k];
            env->entities[b.candidates[k]].active_agent = 0;
        }

        goto finalize;
    } else if (env->policy_agents_per_env > 0) {
        SelectionBuckets b;
        scan_vehicles_initial(env, &b, 0);

        int desired = env->policy_agents_per_env;
        if (desired > MAX_AGENTS) desired = MAX_AGENTS;
        if (desired > b.candidates_count) desired = b.candidates_count;
        if (desired > capacity) desired = capacity;

        if (!env->deterministic_agent_selection) {
            fisher_yates_shuffle(b.candidates, b.candidates_count);
        }
        if (desired > 0) {
            for (int k = 0; k < desired; k++) {
                active_agent_indices[env->active_agent_count++] = b.candidates[k];
                env->entities[b.candidates[k]].active_agent = 1;
            }
            for (int k = desired; k < b.candidates_count; k++) {
                int idx = b.candidates[k];
                if (env->expert_static_car_count < MAX_AGENTS) {
                    expert_static_car_indices[env->expert_static_car_count++] = idx;
                }
                if (env->static_car_count < MAX_AGENTS) {
                    static_car_indices[env->static_car_count++] = idx;
                }
                env->entities[idx].mark_as_expert = 1;
                env->entities[idx].active_agent = 0;
            }
            for (int k = 0; k < b.forced_experts_count; k++) {
                int idx = b.forced_experts[k];
                if (env->expert_static_car_count < MAX_AGENTS) {
                    expert_static_car_indices[env->expert_static_car_count++] = idx;
                }
                if (env->static_car_count < MAX_AGENTS) {
                    static_car_indices[env->static_car_count++] = idx;
                }
            }
            for (int i = 0; i < b.statics_count && env->static_car_count < MAX_AGENTS; i++) {
                static_car_indices[env->static_car_count++] = b.statics[i];
            }

            goto finalize;
        } else {
            int picked = -1;
            for (int i = 0; i < env->num_objects; i++) {
                if (env->entities[i].type != VEHICLE) continue;
                if (env->entities[i].traj_valid && env->entities[i].traj_valid[env->init_steps] == 1) { picked = i; break; }
            }
            if (picked == -1) {
                for (int i = 0; i < env->num_objects; i++) { if (env->entities[i].type == VEHICLE) { picked = i; break; } }
            }
            if (picked != -1) {
                active_agent_indices[env->active_agent_count++] = picked;
                env->entities[picked].active_agent = 1;

                for (int i = 0; i < env->num_objects; i++) {
                    if (i == picked) continue;
                    if (env->entities[i].type == VEHICLE) {
                        if (env->static_car_count < MAX_AGENTS) {
                            static_car_indices[env->static_car_count++] = i;
                        }
                        if (env->expert_static_car_count < MAX_AGENTS) {
                            expert_static_car_indices[env->expert_static_car_count++] = i;
                        }
                        env->entities[i].active_agent = 0;
                        env->entities[i].mark_as_expert = 1;
                    }
                }

                goto finalize;
            }
        }
    }

legacy_select:
    if(env->num_agents == 0){
        env->num_agents = MAX_AGENTS;
    }
    int first_agent_id = env->num_objects-1;
    float distance_to_goal = valid_active_agent(env, first_agent_id);
    int first_agent_selected = distance_to_goal > 0;
    if(distance_to_goal){
        env->active_agent_count = 1;
        active_agent_indices[0] = first_agent_id;
        env->entities[first_agent_id].active_agent = 1;
        env->num_controllable_agents = 1;
    } else {
        env->active_agent_count = 0;
        env->num_controllable_agents = 0;
    }
    for(int i = 0; i < env->num_objects && env->num_controllable_agents < MAX_AGENTS; i++){
        // The last actor is selected first only when it is policy-eligible.
        // Otherwise include it here, particularly when it is an expert pedestrian.
        if (i == first_agent_id && first_agent_selected) continue;

        // Check if the entity type is controllable
        int is_type_controllable;
        if (env->control_non_vehicles) {
            is_type_controllable = (env->entities[i].type == VEHICLE) ||
                                   (env->entities[i].type == PEDESTRIAN) ||
                                   (env->entities[i].type == CYCLIST);
        } else {
            is_type_controllable = (env->entities[i].type == VEHICLE);
        }

        if(!is_type_controllable) continue;

        env->num_controllable_agents++;

        // Return current distance to goal if agent meets other conditions
        float distance_to_goal = valid_active_agent(env, i);
        if(distance_to_goal > 0){
            active_agent_indices[env->active_agent_count] = i;
            env->active_agent_count++;
            env->entities[i].active_agent = 1;
        } else {
            static_car_indices[env->static_car_count] = i;
            env->static_car_count++;
            env->entities[i].active_agent = 0;
            if(env->entities[i].mark_as_expert == 1 || (distance_to_goal >=2.0f && env->active_agent_count == env->num_agents)){
                expert_static_car_indices[env->expert_static_car_count] = i;
                env->expert_static_car_count++;
                env->entities[i].mark_as_expert = 1;
            }
        }
    }
finalize: {
    // Excluding pedestrians/cyclists from policy control must not remove them
    // from the simulation. Keep their trajectories in every selection mode.
    unsigned char* registered = (unsigned char*)calloc(env->num_objects, sizeof(unsigned char));
    for (int i = 0; i < env->active_agent_count; i++) registered[active_agent_indices[i]] = 1;
    for (int i = 0; i < env->static_car_count; i++) registered[static_car_indices[i]] = 1;
    for (int i = 0; i < env->num_objects &&
                    env->active_agent_count + env->static_car_count < MAX_AGENTS; i++) {
        Entity* e = &env->entities[i];
        if (registered[i] || (e->type != PEDESTRIAN && e->type != CYCLIST)) continue;
        static_car_indices[env->static_car_count++] = i;
        expert_static_car_indices[env->expert_static_car_count++] = i;
        e->active_agent = 0;
        e->mark_as_expert = 1;
    }
    free(registered);
}
    // Publish the completed actor lists for rendering, observations and physics.
    env->active_agent_indices = (int*)malloc(env->active_agent_count * sizeof(int));
    env->static_car_indices = (int*)malloc(env->static_car_count * sizeof(int));
    env->expert_static_car_indices = (int*)malloc(env->expert_static_car_count * sizeof(int));
    for(int i=0;i<env->active_agent_count;i++){
        env->active_agent_indices[i] = active_agent_indices[i];
    };
    for(int i=0;i<env->static_car_count;i++){
        env->static_car_indices[i] = static_car_indices[i];

    }
    for(int i=0;i<env->expert_static_car_count;i++){
        env->expert_static_car_indices[i] = expert_static_car_indices[i];
    }
    // Collision and partner-observation loops visit active actors, then static
    // actors. Explicit selection modes must publish the same total as legacy mode.
    env->num_controllable_agents = env->active_agent_count + env->static_car_count;
    if (env->logs_capacity > 0 && env->active_agent_count > env->logs_capacity) {
        fprintf(stderr,
                "[set_active_agents] ERROR map=%s active=%d exceeds logs_capacity=%d\n",
                map_name,
                env->active_agent_count,
                env->logs_capacity);
        assert(env->active_agent_count <= env->logs_capacity);
    }
    return;
}

void init_goal_positions(Drive* env){
    for(int x = 0;x<env->active_agent_count; x++){
        int agent_idx = env->active_agent_indices[x];
        env->entities[agent_idx].init_goal_x = env->entities[agent_idx].goal_position_x;
        env->entities[agent_idx].init_goal_y = env->entities[agent_idx].goal_position_y;
    }
}

void init_traffic_lights(Drive* env) {
    if (!env->traffic_light_states && env->num_traffic_lights > 0) {
        env->traffic_light_states = (TrafficLightState*)calloc(env->num_traffic_lights, sizeof(TrafficLightState));
    }
    env->traffic_lights_enabled = rand_int_inclusive(0, 4) > 0;
    if (env->has_traffic_light_sequences) env->traffic_lights_enabled = 1;
    for (int i = 0; i < env->num_traffic_lights; i++) {
        TrafficLightState* state = &env->traffic_light_states[i];
        if (state->sequence) {
            state->traffic_light_color = state->sequence[(int)fmin(env->timestep, state->sequence_length - 1)];
            state->traffic_light_transition_in = 0;
            continue;
        }
        state->traffic_light_color = env->traffic_lights_enabled
            ? rand_int_inclusive(TRAFFIC_LIGHT_OFF, TRAFFIC_LIGHT_GREEN) : TRAFFIC_LIGHT_OFF;
        state->traffic_light_transition_in = sample_traffic_light_transition(state->traffic_light_color);
    }
}

void update_traffic_lights(Drive* env) {
    int random_update = env->timestep != env->init_steps + 1 && env->timestep % TRAFFIC_LIGHT_UPDATE_INTERVAL == 0;
    if (!env->traffic_lights_enabled || (!env->has_traffic_light_sequences && !random_update)) return;

    for (int i = 0; i < env->num_traffic_lights; i++) {
        TrafficLightState* state = &env->traffic_light_states[i];
        if (state->sequence) {
            state->traffic_light_color = state->sequence[(int)fmin(env->timestep, state->sequence_length - 1)];
            continue;
        }
        if (!random_update) continue;
        if (state->traffic_light_color == TRAFFIC_LIGHT_OFF) continue;

        state->traffic_light_transition_in -= TRAFFIC_LIGHT_UPDATE_INTERVAL;
        if (state->traffic_light_transition_in > 0) continue;

        if (state->traffic_light_color == TRAFFIC_LIGHT_RED) {
            state->traffic_light_color = TRAFFIC_LIGHT_YELLOW;
        } else if (state->traffic_light_color == TRAFFIC_LIGHT_YELLOW) {
            state->traffic_light_color = rand_int_inclusive(0, 1) ? TRAFFIC_LIGHT_GREEN : TRAFFIC_LIGHT_RED;
        } else if (state->traffic_light_color == TRAFFIC_LIGHT_GREEN) {
            state->traffic_light_color = TRAFFIC_LIGHT_YELLOW;
        }

        state->traffic_light_transition_in = sample_traffic_light_transition(state->traffic_light_color);
    }
}

void init(Drive* env){
    env->human_agent_idx = 0;
    env->timestep = 0;
    if (!env->dynamic_map_generation)
        env->entities = load_map_binary(env->map_name, env);
    if (!env->static_road_map) {
        StaticRoadMap* map = create_static_road_map(env->static_road_entities, env->num_roads);
        acquire_static_road_map(env, map);
        release_static_road_map(map);
    }
    if (env->use_goal_generation && !env->static_road_map->topology_initialized) {
        init_topology_graph(env);
        env->static_road_map->topology_graph = env->topology_graph;
        env->static_road_map->topology_initialized = 1;
    }
    env->topology_graph = env->static_road_map->topology_graph;
    init_traffic_lights(env);
    env->logs_capacity = 0;
    set_active_agents(env);
    env->logs_capacity = env->active_agent_count;
    set_start_position(env);
    init_goal_positions(env);
    env->logs = (Log*)calloc(env->active_agent_count, sizeof(Log));
    env->collision_counts = (int*)calloc(COLLISION_COUNT_BUCKETS, sizeof(int));
    env->expert_collision_counts = (int*)calloc(COEFFICIENT_COUNT_BINS, sizeof(int));
    env->offroad_counts = (int*)calloc(COEFFICIENT_COUNT_BINS, sizeof(int));
    env->stop_line_counts = (int*)calloc(COEFFICIENT_COUNT_BINS, sizeof(int));
    env->comfort_violation_counts = (int*)calloc(COEFFICIENT_COUNT_BINS, sizeof(int));
    env->lane_center_offset_counts = (int*)calloc(LANE_CENTER_OFFSET_BUCKETS, sizeof(int));
    env->acceleration_action_counts = (int*)calloc(CLASSIC_ACCELERATION_ACTION_BINS, sizeof(int));
    env->steering_action_counts = (int*)calloc(CLASSIC_STEERING_ACTION_BINS, sizeof(int));
    env->speed_counts = (int*)calloc(SPEED_COUNT_BINS, sizeof(int));
}

static void saveAgentObservationsReplayToFile(Drive* env) {
    int ego_dim = drive_ego_features(env->dynamics_model);
    int max_obs = ego_dim + 7*(MAX_VISIBLE_AGENTS - 1) + 7*MAX_ROAD_SEGMENT_OBSERVATIONS;

    float (*obs)[env->export_log.capacity][max_obs] = (float(*)[env->export_log.capacity][max_obs]) env->export_log.observations;
    float (*reward_components)[env->export_log.capacity][REPLAY_REWARD_COMPONENT_COUNT] =
        (float(*)[env->export_log.capacity][REPLAY_REWARD_COMPONENT_COUNT]) env->export_log.reward_components;
    int (*action_indices)[env->export_log.capacity][REPLAY_ACTION_COMPONENT_COUNT] =
        (int(*)[env->export_log.capacity][REPLAY_ACTION_COMPONENT_COUNT]) env->export_log.action_indices;
    int exported_steps = fmin(env->timestep+1, env->export_log.capacity);
    size_t obs_count = (size_t)exported_steps * (size_t)max_obs;
    size_t uncompressed_bytes = obs_count * sizeof(float);
    if (uncompressed_bytes > INT_MAX) {
        fprintf(stderr, "obs buffer too large to compress\n");
        return;
    }
    int src_size = (int)uncompressed_bytes;
    int max_compressed = LZ4_compressBound(src_size);
    if (max_compressed <= 0) {
        fprintf(stderr, "failed to compute LZ4 compression bound\n");
        return;
    }

    size_t reward_count = (size_t)exported_steps * REPLAY_REWARD_COMPONENT_COUNT;
    size_t reward_uncompressed_bytes = reward_count * sizeof(float);
    if (reward_uncompressed_bytes > INT_MAX) {
        fprintf(stderr, "reward buffer too large to compress\n");
        return;
    }
    int reward_src_size = (int)reward_uncompressed_bytes;
    int reward_max_compressed = LZ4_compressBound(reward_src_size);
    if (reward_max_compressed <= 0) {
        fprintf(stderr, "failed to compute reward compression bound\n");
        return;
    }

    char* compressed = (char*)malloc((size_t)max_compressed);
    if (compressed == NULL) {
        fprintf(stderr, "failed to allocate compression buffer\n");
        return;
    }

    char* reward_compressed = (char*)malloc((size_t)reward_max_compressed);
    if (reward_compressed == NULL) {
        fprintf(stderr, "failed to allocate reward compression buffer\n");
        free(compressed);
        return;
    }

    for (int i = 0; i < env->export_log.num_agents; i++) {
        char file_name[256];
        sprintf(file_name, "%s/agent_%02d.bin", env->replay_dir, i);

        FILE* file = fopen(file_name, "wb");
        if (file == NULL) {
            fprintf(stderr, "failed to open replay file for writing\n");
            free(compressed);
            free(reward_compressed);
            return;
        }

        const char* src = (const char*)&obs[i][0][0];
        int compressed_size = LZ4_compress_default(src, compressed, src_size, max_compressed);
        if (compressed_size <= 0) {
            fprintf(stderr, "failed to compress agent obs\n");
            fclose(file);
            free(compressed);
            free(reward_compressed);
            return;
        }

        const char* reward_src = (const char*)&reward_components[i][0][0];
        int reward_compressed_size = LZ4_compress_default(
            reward_src, reward_compressed, reward_src_size, reward_max_compressed);
        if (reward_compressed_size <= 0) {
            fprintf(stderr, "failed to compress agent rewards\n");
            fclose(file);
            free(compressed);
            free(reward_compressed);
            return;
        }

        // File format:
        //   max_obs (int), exported_steps (int), compressed_size (int), obs LZ4 data,
        //   reward_component_count (int), reward_compressed_size (int), reward LZ4 data,
        //   action_component_count (int), action index data
        if (fwrite(&max_obs, sizeof(int), 1, file) != 1 ||
            fwrite(&exported_steps, sizeof(int), 1, file) != 1 ||
            fwrite(&compressed_size, sizeof(int), 1, file) != 1 ||
            fwrite(compressed, 1, (size_t)compressed_size, file) != (size_t)compressed_size ||
            fwrite(&(int){REPLAY_REWARD_COMPONENT_COUNT}, sizeof(int), 1, file) != 1 ||
            fwrite(&reward_compressed_size, sizeof(int), 1, file) != 1 ||
            fwrite(reward_compressed, 1, (size_t)reward_compressed_size, file) != (size_t)reward_compressed_size ||
            fwrite(&(int){REPLAY_ACTION_COMPONENT_COUNT}, sizeof(int), 1, file) != 1 ||
            fwrite(&action_indices[i][0][0], sizeof(int),
                (size_t)exported_steps * REPLAY_ACTION_COMPONENT_COUNT, file) !=
                (size_t)exported_steps * REPLAY_ACTION_COMPONENT_COUNT) {
            fprintf(stderr, "failed to write compressed replay file\n");
            fclose(file);
            free(compressed);
            free(reward_compressed);
            return;
        }
        fclose(file);
    }
    free(compressed);
    free(reward_compressed);
};

static void saveReplayToFile(Drive* env) {
    char file_name[256];
    sprintf(file_name, "%s/%s", env->replay_dir, "replay.bin");

    FILE* file = fopen(file_name, "wb");
    if (file == NULL) {
        fprintf(stderr, "failed to open replay file for writing\n");
        return;
    }

    fwrite(&env->export_log.num_actors, sizeof(int), 1, file);
    int road_records = env->num_roads + env->num_eval_traffic_light_lines;
    fwrite(&road_records, sizeof(int), 1, file);

    int exported_steps = fmin(env->timestep+1, env->export_log.capacity);

    for (int i = 0; i < env->export_log.num_actors; i++) {
        Entity* e = &env->entities[env->export_log.actor_indices[i]];
        fwrite(&e->type, sizeof(int), 1, file);

        if (e->type == VEHICLE || e->type == PEDESTRIAN || e->type == CYCLIST) {
            fwrite(&exported_steps, sizeof(int), 1, file);
            fwrite(env->export_log.traj_x + i * env->export_log.capacity, sizeof(float), exported_steps, file);
            fwrite(env->export_log.traj_y + i * env->export_log.capacity, sizeof(float), exported_steps, file);
            fwrite(env->export_log.traj_z + i * env->export_log.capacity, sizeof(float), exported_steps, file);
            fwrite(env->export_log.traj_vx + i * env->export_log.capacity, sizeof(float), exported_steps, file);
            fwrite(env->export_log.traj_vy + i * env->export_log.capacity, sizeof(float), exported_steps, file);
            fwrite(env->export_log.traj_vz + i * env->export_log.capacity, sizeof(float), exported_steps, file);
            fwrite(env->export_log.heading + i * env->export_log.capacity, sizeof(float), exported_steps, file);
            fwrite(env->export_log.valid + i * env->export_log.capacity, sizeof(int), exported_steps, file);
        } else {
            fwrite(&e->array_size, sizeof(int), 1, file);
            fwrite(e->traj_x, sizeof(float), e->array_size, file);
            fwrite(e->traj_y, sizeof(float), e->array_size, file);
            fwrite(e->traj_z, sizeof(float), e->array_size, file);
        }

        fwrite(&e->width, sizeof(float), 1, file);
        fwrite(&e->length, sizeof(float), 1, file);
        fwrite(&e->height, sizeof(float), 1, file);

        if (e->multi_goal_count) {
            fwrite(e->multi_goal_position_x, sizeof(float), 1, file);
            fwrite(e->multi_goal_position_y, sizeof(float), 1, file);
            fwrite(e->multi_goal_position_z, sizeof(float), 1, file);
        } else {
            fwrite(&e->goal_position_x, sizeof(float), 1, file);
            fwrite(&e->goal_position_y, sizeof(float), 1, file);
            fwrite(&e->goal_position_z, sizeof(float), 1, file);
        }

        int mark_as_expert = 1;
        fwrite(&mark_as_expert, sizeof(int), 1, file);
    }

    for (int i = 0; i < env->num_roads; i++) {
        const StaticRoadEntity* road = &env->static_road_entities[i];
        int raw_type = road->type;
        const int* light_states = road->traffic_light_idx >= 0
            ? env->export_log.traffic_light_states + (size_t)road->traffic_light_idx * env->export_log.capacity : NULL;
        if (light_states) raw_type |= MAP_TRAFFIC_LIGHT_SEQUENCE_FLAG;
        if (road->type == TRAFFIC_LIGHT_STOP_LINE && road->traffic_light_off_type == STOP_SIGN)
            raw_type |= MAP_TRAFFIC_LIGHT_YIELD_FLAG;
        fwrite(&raw_type, sizeof(int), 1, file);
        fwrite(&road->array_size, sizeof(int), 1, file);
        fwrite(road->traj_x, sizeof(float), road->array_size, file);
        fwrite(road->traj_y, sizeof(float), road->array_size, file);
        fwrite(road->traj_z, sizeof(float), road->array_size, file);
        // Legacy road records carry unused dimensions, goal and expert fields.
        float unused[6] = {0};
        int mark_as_expert = 0;
        fwrite(unused, sizeof(float), 6, file);
        fwrite(&mark_as_expert, sizeof(int), 1, file);
        if (light_states) {
            // Reuse the map sequence field for actual colors at each recorded frame.
            fwrite(&exported_steps, sizeof(int), 1, file);
            fwrite(light_states, sizeof(int), exported_steps, file);
        }
    }
    for (int i = 0; i < env->num_eval_traffic_light_lines; i++) {
        const EvalTrafficLightLine* line = &env->eval_traffic_light_lines[i];
        int header[] = {EVAL_TRAFFIC_LIGHT_LINE, 2};
        float zeros[8] = {0}; // Z coordinates, dimensions and goal.
        fwrite(header, sizeof(int), 2, file);
        fwrite(line->x, sizeof(float), 2, file);
        fwrite(line->y, sizeof(float), 2, file);
        fwrite(zeros, sizeof(float), 8, file);
        fwrite(&line->traffic_light_idx, sizeof(int), 1, file);
    }
    fclose(file);

    saveAgentObservationsReplayToFile(env);
}

void c_close(Drive* env){
    for (int i = 0; i < env->num_exit_orders; i++) {
        free(env->exit_orders[i].traj_x);
        free(env->exit_orders[i].traj_y);
    }
    free(env->exit_orders);
    free(env->eval_traffic_light_lines);
    for(int i = 0; i < env->num_objects; i++){
        free_entity(&env->entities[i]);
    }
    free(env->entities);
    free(env->active_agent_indices);
    free(env->logs);
    free(env->collision_counts);
    free(env->expert_collision_counts);
    free(env->offroad_counts);
    free(env->stop_line_counts);
    free(env->comfort_violation_counts);
    free(env->lane_center_offset_counts);
    free(env->acceleration_action_counts);
    free(env->steering_action_counts);
    free(env->speed_counts);
    if (env->static_road_map) {
        release_static_road_map(env->static_road_map);
    } else {
        // Also used for maps loaded only to count active agents, before init().
        free_static_road_entities(env->static_road_entities, env->num_roads);
        free_grid_map(env->grid_map);
        free(env->neighbor_offsets);
        freeTopologyGraph(env->topology_graph);
    }
    if (env->traffic_light_states) {
        for (int i = 0; i < env->num_traffic_lights; i++) free(env->traffic_light_states[i].sequence);
        free(env->traffic_light_states);
    }
    free(env->static_car_indices);
    free(env->expert_static_car_indices);
    // free(env->map_name);
    free(env->ini_file);
    free(env->export_log.traj_x);
    free(env->export_log.traj_y);
    free(env->export_log.traj_z);
    free(env->export_log.traj_vx);
    free(env->export_log.traj_vy);
    free(env->export_log.traj_vz);
    free(env->export_log.heading);
    free(env->export_log.actor_indices);
    free(env->export_log.valid);
    free(env->export_log.traffic_light_states);
    free(env->export_log.observations);
    free(env->export_log.reward_components);
    free(env->export_log.action_indices);
    free(env->replay_dir);
}

void allocate(Drive* env){
    init(env);
    int ego_dim = drive_ego_features(env->dynamics_model);
    int max_obs = ego_dim + 7*(MAX_VISIBLE_AGENTS - 1) + 7*MAX_ROAD_SEGMENT_OBSERVATIONS;
    env->observations = (float*)calloc(env->active_agent_count*max_obs, sizeof(float));
    env->actions = (float*)calloc(env->active_agent_count*2, sizeof(float));
    env->rewards = (float*)calloc(env->active_agent_count, sizeof(float));
    env->terminals= (unsigned char*)calloc(env->active_agent_count, sizeof(unsigned char));
    env->truncations = (unsigned char*)calloc(env->active_agent_count, sizeof(unsigned char));
}

void free_allocated(Drive* env){
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->truncations);
    c_close(env);
}

float clipSpeed(float speed, Entity* agent) {
    float maxSpeed = MAX_SPEED;
    if (agent->max_speed > 0 && agent->max_speed < MAX_SPEED) {
        maxSpeed = agent->max_speed;
    }

    if (speed > maxSpeed) return maxSpeed;
    if (speed < 0) return 0;
    return speed;
}

float normalize_value(float value, float min, float max){
    return (value - min) / (max - min);
}

void move_dynamics(Drive* env, int action_idx, int agent_idx){
    Entity* agent = &env->entities[agent_idx];
    float action[2];
    if (env->action_type == 1) {
        const float (*issued)[2] = (const float(*)[2])env->actions;
        action[0] = issued[action_idx][0];
        action[1] = issued[action_idx][1];
    } else {
        const int (*issued)[2] = (const int(*)[2])env->actions;
        action[0] = issued[action_idx][0];
        action[1] = issued[action_idx][1];
        if (env->dynamics_model == CLASSIC) {
            // Logs/replays describe the issued policy action, as before. Only
            // the dynamics consume a delayed command; the input buffer is read-only.
            record_discrete_classic_action(env, action_idx, issued[action_idx][0], issued[action_idx][1]);
        }
    }
    // Preserve the selected command before delay, gains, or actuator limits.
    // Continuous actions already use these normalized units.
    float longitudinal = action[0], lateral = action[1];
    if (env->action_type != 1) {
        if (env->dynamics_model == CLASSIC) {
            longitudinal = ACCELERATION_VALUES[(int)action[0]] / 4.0f;
            lateral = STEERING_VALUES[(int)action[1]] / 0.667f;
        } else {
            longitudinal = JERK_LONG[(int)action[0]];
            longitudinal /= longitudinal < 0 ? 15.0f : 4.0f;
            lateral = JERK_LAT[(int)action[1]] / 4.0f;
        }
    }
    agent->prev_x = agent->x;
    agent->prev_y = agent->y;
    record_ego_history(agent, longitudinal, lateral);
    delay_agent_action(agent, action);

    if (env->dynamics_model == CLASSIC) {
        // Classic dynamics model
        float acceleration = 0.0f;
        float steering = 0.0f;

        if (env->action_type == 1) { // continuous
            acceleration = action[0];
            steering = action[1];

            acceleration *= ACCELERATION_VALUES[6];
            steering *= STEERING_VALUES[12];
        } else { // discrete
            int acceleration_index = (int)action[0];
            int steering_index = (int)action[1];
            acceleration = ACCELERATION_VALUES[acceleration_index];
            steering = STEERING_VALUES[steering_index];
        }
        if (agent->acceleration_gain != 0.0f) acceleration *= agent->acceleration_gain;
        if (agent->steering_gain != 0.0f) steering *= agent->steering_gain;

        float max_steering_rate;
        if (agent->low_steering_rate && agent->high_steering_rate) {
            float steering_error = fabsf(steering - agent->steering_angle);
            float slope = (agent->high_steering_rate - agent->low_steering_rate) / (agent->high_steering_rate_error - agent->low_steering_rate_error);
            float intercept = agent->low_steering_rate - slope * agent->low_steering_rate_error;
            max_steering_rate = fmaxf(agent->low_steering_rate, fminf(agent->high_steering_rate, steering_error * slope + intercept));
        } else {
            max_steering_rate = 0.033f;
        }

        steering = agent->steering_angle + fmaxf(fminf(steering - agent->steering_angle, max_steering_rate), -max_steering_rate);

        // Current state
        float x = agent->x;
        float y = agent->y;
        float heading = agent->heading;
        float vx = agent->vx;
        float vy = agent->vy;

        float friction_coefficient = agent->friction_coefficient;
        if (agent->friction_coefficient == 0.0f) friction_coefficient = 1.0f;

        float max_acceleration = 9.81 * friction_coefficient;
        if (acceleration > max_acceleration) acceleration = max_acceleration;
        if (acceleration < -max_acceleration) acceleration = -max_acceleration;

        float speed = sqrtf(vx*vx + vy*vy);
        float new_speed = speed + acceleration * env->dt;
        new_speed = clipSpeed(new_speed, agent);
        env->logs[action_idx].avg_distance_driven_per_agent += new_speed * env->dt;

        // Compute yaw rate
        float beta = tanh(.5*tanf(steering));

        float max_lateral_acceleration = sqrtf(max_acceleration*max_acceleration-acceleration*acceleration);
        float a_lat = new_speed * new_speed * cosf(beta)*tanf(steering) / agent->wheelbase;
        if (fabsf(a_lat) > max_lateral_acceleration) {
            steering = copysignf(atanf(max_lateral_acceleration / new_speed / new_speed * agent->wheelbase), steering);
            beta = tanh(.5*tanf(steering));
            a_lat = new_speed * new_speed * cosf(beta)*tanf(steering) / agent->wheelbase;
        }

        // New heading
        float yaw_rate = (new_speed*cosf(beta)*tanf(steering)) / agent->wheelbase;

        // New velocity
        float new_vx = new_speed * cosf(heading + beta);
        float new_vy = new_speed * sinf(heading + beta);

        // Update position
        x = x + (new_vx*env->dt);
        y = y + (new_vy*env->dt);
        heading = heading + yaw_rate*env->dt;

        // Apply updates to the agent's state
        agent->x = x;
        agent->y = y;
        agent->heading = heading;
        agent->heading_x = cosf(heading);
        agent->heading_y = sinf(heading);
        agent->vx = new_vx;
        agent->vy = new_vy;
        agent->steering_angle = steering;

        int comfort_violations = (fabsf(acceleration) > 3.0f) + (fabsf(a_lat) > 3.0f);
        if (comfort_violations > 0) {
            int comfort_bin = clipped_scaled_bin(agent->comfort_reward_coefficient, 10000.0f, COEFFICIENT_COUNT_BINS);
            env->comfort_violation_counts[comfort_bin] += comfort_violations;
        }

        float comfort_reward = -agent->comfort_reward_coefficient * comfort_violations;
        env->rewards[action_idx] += comfort_reward;
        record_reward_component(env, action_idx, REWARD_COMPONENT_COMFORT, comfort_reward);
    } else {
        // JERK dynamics model
        // Extract action components
        float a_long, a_lat;
        if (env->action_type == 1) { // continuous
            // Asymmetric scaling for longitudinal jerk to match discrete action space
            // Discrete: JERK_LONG = [-15, -4, 0, 4] (more braking than acceleration)
            float a_long_action = action[0];  // [-1, 1]
            if (a_long_action < 0) {
                a_long = a_long_action * (-JERK_LONG[0]);  // Negative: [-1, 0] → [-15, 0] (braking)
            } else {
                a_long = a_long_action * JERK_LONG[3];   // Positive: [0, 1] → [0, 4] (acceleration)
            }

            // Symmetric scaling for lateral jerk
            a_lat = action[1] * JERK_LAT[2];
        } else { // discrete
            int a_long_idx = (int)action[0];
            int a_lat_idx = (int)action[1];
            a_long = JERK_LONG[a_long_idx];
            a_lat = JERK_LAT[a_lat_idx];
        }

        // Calculate new acceleration
        float a_long_new = agent->a_long + a_long * env->dt;
        float a_lat_new = agent->a_lat + a_lat * env->dt;

        // Make it easy to stop with 0 accel
        if (agent->a_long * a_long_new < 0) {
            a_long_new = 0.0f;
        } else {
            a_long_new = clip(a_long_new, -5.0f, 2.5f);
        }

        if (agent->a_lat * a_lat_new < 0) {
            a_lat_new = 0.0f;
        } else {
            a_lat_new = clip(a_lat_new, -4.0f, 4.0f);
        }

        // Calculate new velocity
        float v_dot_heading = agent->vx * agent->heading_x + agent->vy * agent->heading_y;
        float signed_v = copysignf(sqrtf(agent->vx*agent->vx + agent->vy*agent->vy), v_dot_heading);
        float v_new = signed_v + 0.5f * (a_long_new + agent->a_long) * env->dt;

        // Make it easy to stop with 0 vel
        if (signed_v * v_new < 0) {
            v_new = 0.0f;
        } else {
            v_new = clip(v_new, -2.0f, 20.0f);
        }

        // Calculate new steering angle
        float signed_curvature = a_lat_new / fmaxf(v_new * v_new, 1e-5f);
        signed_curvature = copysignf(fmaxf(fabsf(signed_curvature), 1e-5f), signed_curvature);
        float steering_angle = atanf(signed_curvature * agent->wheelbase);
        float delta_steer = clip(steering_angle - agent->steering_angle, -0.6f * env->dt, 0.6f * env->dt);
        float new_steering_angle = clip(agent->steering_angle + delta_steer, -0.55f, 0.55f);

        // Update curvature and accel to account for limited steering
        signed_curvature = tanf(new_steering_angle) / agent->wheelbase;
        a_lat_new = v_new * v_new * signed_curvature;

        // Calculate resulting movement using bicycle dynamics
        float d = 0.5f * (v_new + signed_v) * env->dt;
        float theta = d * signed_curvature;
        float dx_local, dy_local;

        if (fabsf(signed_curvature) < 1e-5f || fabsf(theta) < 1e-5f) {
            dx_local = d;
            dy_local = 0.0f;
        } else {
            dx_local = sinf(theta) / signed_curvature;
            dy_local = (1.0f - cosf(theta)) / signed_curvature;
        }

        float dx = dx_local * agent->heading_x - dy_local * agent->heading_y;
        float dy = dx_local * agent->heading_y + dy_local * agent->heading_x;

        // Update everything
        agent->x += dx;
        agent->y += dy;
        agent->jerk_long = (a_long_new - agent->a_long) / env->dt;
        agent->jerk_lat = (a_lat_new - agent->a_lat) / env->dt;
        agent->a_long = a_long_new;
        agent->a_lat = a_lat_new;
        agent->heading = normalize_heading(agent->heading + theta);
        agent->heading_x = cosf(agent->heading);
        agent->heading_y = sinf(agent->heading);
        agent->vx = v_new * agent->heading_x;
        agent->vy = v_new * agent->heading_y;
        agent->steering_angle = new_steering_angle;
    }

    return;
}

// A separate actor index is rebuilt from current positions for each observation
// update. Road geometry stays in the shared, immutable GridMap.
#define ACTOR_OBSERVATION_CELL_SIZE 50.0f
#define ACTOR_OBSERVATION_BUCKETS 16384  // Power of two; hash collisions are harmless.
#define ACTOR_OBSERVATION_WORDS ((MAX_AGENTS + 63) / 64)

typedef struct {
    int count;  // -1 until this agent's visible partners have been selected.
    int indices[MAX_VISIBLE_AGENTS - 1];
} ActorObservationPartners;

// Zero-initialize before the first build; release the cache after the last use.
typedef struct {
    int heads[ACTOR_OBSERVATION_BUCKETS];
    int next[MAX_AGENTS];
    int usable;
    ActorObservationPartners* partners;
    int partner_capacity;
} ActorObservationGrid;

static inline float actor_detection_distance(const Entity* entity) {
    return entity->object_detection_distance > 0.0f ? entity->object_detection_distance : 50.0f;
}

static inline int observation_actor_index(const Drive* env, int rank) {
    return rank < env->active_agent_count
        ? env->active_agent_indices[rank]
        : env->static_car_indices[rank - env->active_agent_count];
}

static int actor_observation_cell(float position, int* cell) {
    double value = floor((double)position / ACTOR_OBSERVATION_CELL_SIZE);
    // Fall back to the original scan if coordinates cannot be indexed safely.
    if (!isfinite(value) || value <= INT_MIN || value >= INT_MAX) return 0;
    *cell = (int)value;
    return 1;
}

static inline unsigned actor_observation_bucket(int x, int y) {
    return ((uint32_t)x * 73856093u ^ (uint32_t)y * 19349663u)
        & (ACTOR_OBSERVATION_BUCKETS - 1);
}

static void build_actor_observation_grid(const Drive* env, ActorObservationGrid* grid) {
    grid->usable = 0;
    if (grid->partner_capacity < env->active_agent_count) {
        ActorObservationPartners* partners = realloc(grid->partners,
            (size_t)env->active_agent_count * sizeof(ActorObservationPartners));
        if (!partners) {
            fprintf(stderr, "Failed to allocate actor observation cache\n");
            abort();
        }
        grid->partners = partners;
        grid->partner_capacity = env->active_agent_count;
    }
    // The grid and cached partners describe the same snapshot of actor state.
    for (int i = 0; i < env->active_agent_count; i++) grid->partners[i].count = -1;
    // For small scenarios a direct scan costs less than constructing the grid.
    if (env->num_controllable_agents <= 128) return;
    assert(env->num_controllable_agents <= MAX_AGENTS);
    memset(grid->heads, -1, sizeof(grid->heads));
    for (int rank = 0; rank < env->num_controllable_agents; rank++) {
        const Entity* entity = &env->entities[observation_actor_index(env, rank)];
        int x, y;
        if (!actor_observation_cell(entity->x, &x) || !actor_observation_cell(entity->y, &y)) return;
        unsigned bucket = actor_observation_bucket(x, y);
        grid->next[rank] = grid->heads[bucket];
        grid->heads[bucket] = rank;
    }
    grid->usable = 1;
}

static void free_actor_observation_grid(ActorObservationGrid* grid) {
    free(grid->partners);
    grid->partners = NULL;
    grid->partner_capacity = 0;
    grid->usable = 0;
}

static int nearby_observation_actors(const Drive* env, const ActorObservationGrid* grid,
                                    const Entity* ego, int* indices) {
    int count = 0;
    if (ego->respawn_timestep != -1) return 0;
    float radius = actor_detection_distance(ego);
    int min_x, max_x, min_y, max_y;
    if (!grid->usable ||
        !actor_observation_cell(ego->x - radius, &min_x) ||
        !actor_observation_cell(ego->x + radius, &max_x) ||
        !actor_observation_cell(ego->y - radius, &min_y) ||
        !actor_observation_cell(ego->y + radius, &max_y) ||
        // A full scan is cheaper for unusually large detection radii.
        ((double)max_x - min_x + 1) * ((double)max_y - min_y + 1) > env->num_controllable_agents) {
        for (int rank = 0; rank < env->num_controllable_agents; rank++)
            indices[count++] = observation_actor_index(env, rank);
        return count;
    }

    // Bits represent the original active-then-background order. This preserves
    // observation ordering and the 63-partner cutoff, and deduplicates buckets
    // when two neighboring cells hash to the same bucket.
    uint64_t candidates[ACTOR_OBSERVATION_WORDS] = {0};
    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            unsigned bucket = actor_observation_bucket(x, y);
            for (int rank = grid->heads[bucket]; rank != -1; rank = grid->next[rank])
                candidates[rank / 64] |= UINT64_C(1) << (rank % 64);
        }
    }
    int words = (env->num_controllable_agents + 63) / 64;
    for (int word = 0; word < words; word++) {
        uint64_t bits = candidates[word];
        while (bits) {
            int rank = word * 64 + __builtin_ctzll(bits);
            indices[count++] = observation_actor_index(env, rank);
            bits &= bits - 1;
        }
    }
    return count;
}

// Cache the final observation selection, so either consumer can run first.
// Rebuild the grid after changing actor positions, lists, respawns or radii.
static const ActorObservationPartners* visible_observation_actors(
        const Drive* env, ActorObservationGrid* grid, int agent_slot) {
    ActorObservationPartners* partners = &grid->partners[agent_slot];
    if (partners->count >= 0) return partners;
    partners->count = 0;
    int agent_idx = env->active_agent_indices[agent_slot];
    const Entity* ego = &env->entities[agent_idx];
    float radius = actor_detection_distance(ego);
    float max_distance = radius * radius;
    int nearby_actors[MAX_AGENTS];
    int nearby_count = nearby_observation_actors(env, grid, ego, nearby_actors);
    for (int j = 0; j < nearby_count && partners->count < MAX_VISIBLE_AGENTS - 1; j++) {
        int index = nearby_actors[j];
        const Entity* other = &env->entities[index];
        if (other->type > 3) break;
        if (index == agent_idx || other->respawn_timestep != -1) continue;
        float dx = other->x - ego->x, dy = other->y - ego->y;
        if (dx * dx + dy * dy > max_distance) continue;
        partners->indices[partners->count++] = index;
    }
    return partners;
}

static void apply_safety_penalties(Drive* env, ActorObservationGrid* actor_grid) {
    for (int agent_slot = 0; agent_slot < env->active_agent_count; agent_slot++) {
        int agent_idx = env->active_agent_indices[agent_slot];
        Entity* ego = &env->entities[agent_idx];
        if (ego->type > 3) break;
        float speed_squared = ego->vx * ego->vx + ego->vy * ego->vy;
        if (ego->type != VEHICLE || ego->x == INVALID_POSITION || ego->respawn_timestep != -1 ||
            ego->collision_state == VEHICLE_COLLISION || speed_squared <= SAFETY_MIN_SPEED * SAFETY_MIN_SPEED) continue;

        const ActorObservationPartners* partners = visible_observation_actors(env, actor_grid, agent_slot);
        int ttc_violation = 0, lateral_violation = 0;
        for (int j = 0; j < partners->count; j++) {
            Entity* other = &env->entities[partners->indices[j]];
            if (other->type < VEHICLE || other->x == INVALID_POSITION) continue;
            float dx = fabsf(other->x - ego->x), dy = fabsf(other->y - ego->y);
            // Conservative world-axis bounds avoid unnecessary detailed geometry.
            float extent = 0.5f * (ego->length + ego->width + other->length + other->width);
            if (dx < extent + SAFETY_CLEARANCE && dy < extent + SAFETY_CLEARANCE &&
                vehicle_clearance_squared(ego, other) < SAFETY_CLEARANCE * SAFETY_CLEARANCE) {
                lateral_violation = 1;
            }
            if (other->type == VEHICLE &&
                dx <= extent + SAFETY_TIME_SECONDS * fabsf(other->vx - ego->vx) &&
                dy <= extent + SAFETY_TIME_SECONDS * fabsf(other->vy - ego->vy) &&
                !vehicle_approaches_from_behind(ego, other) &&
                vehicle_time_to_collision(ego, other) < SAFETY_TIME_SECONDS) {
                ttc_violation = 1;
            }
        }
        float reward = -(ttc_violation + lateral_violation) * SAFETY_COLLISION_FRACTION *
            ego->safety_reward_coefficient * (1.0f + 0.1f * sqrtf(speed_squared));
        env->rewards[agent_slot] += reward;
        record_reward_component(env, agent_slot, REWARD_COMPONENT_SAFETY, reward);
    }
}

// Pure observation conversion: repeated calls never advance history.
static void write_ego_observation(const Entity* e, int dynamics_model, float* obs) {
    const float c = e->heading_x, s = e->heading_y;
    // Safety penalties, comfort, then lane-following preferences.
    obs[0] = e->collision_reward_coefficient;
    obs[1] = e->safety_reward_coefficient;
    obs[2] = e->offroad_reward_coefficient;
    obs[3] = e->stop_line_reward_coefficient;
    obs[4] = e->comfort_reward_coefficient;
    obs[5] = e->lane_center_reward_coefficient;
    obs[6] = e->lane_align_reward_coefficient;
    obs[7] = e->lane_center_bias;
    // Current speed, steering angle, speed limit, and vehicle dimensions.
    obs[8] = hypotf(e->vx, e->vy) / MAX_SPEED;
    obs[9] = e->steering_angle / 0.667f;
    obs[10] = e->max_speed == 0.0f ? 1.0f : fminf(e->max_speed / MAX_SPEED, 1.0f);
    obs[11] = e->width / MAX_VEH_WIDTH;
    obs[12] = e->length / MAX_VEH_LEN;
    // Goal geometry and remaining-goal flags.
    obs[13] = e->goal_radius / 3.0f;
    float dx = e->goal_position_x - e->x, dy = e->goal_position_y - e->y;
    obs[14] = (dx * c + dy * s) * 0.005f;
    obs[15] = (-dx * s + dy * c) * 0.005f;
    int next = e->multi_goal_idx + 1;
    if (next < e->multi_goal_count) {
        dx = e->multi_goal_position_x[next] - e->x;
        dy = e->multi_goal_position_y[next] - e->y;
    }
    obs[16] = (dx * c + dy * s) * 0.005f;
    obs[17] = (-dx * s + dy * c) * 0.005f;
    int remaining = e->reached_final_goal ? 0 : e->multi_goal_count > 0
        ? e->multi_goal_count - e->multi_goal_idx : 1;
    obs[18] = remaining >= 2;
    obs[19] = remaining >= 1;
    // Current lane-relative pose.
    obs[20] = e->lane_heading_angle_diff;
    obs[21] = e->metrics_array[LANE_LATERAL_OFFSET_IDX] > -100.0f
        ? e->metrics_array[LANE_LATERAL_OFFSET_IDX] / 4.0f : 0.0f;
    // Ten past poses (XY and heading cos/sin), speeds and selected commands.
    for (int k = 0; k < 10; k++) {
        int index = (e->ego_history_head + EGO_HISTORY_CAPACITY - 1 - k) % EGO_HISTORY_CAPACITY;
        const float* h = e->ego_history[index];
        dx = h[0] - e->x;
        dy = h[1] - e->y;
        obs[22 + 2*k] = (dx * c + dy * s) * 0.02f;
        obs[22 + 2*k + 1] = (-dx * s + dy * c) * 0.02f;
        obs[42 + 2*k] = h[2] * c + h[3] * s;
        obs[42 + 2*k + 1] = h[3] * c - h[2] * s;
        obs[62 + k] = h[4] / MAX_SPEED;
        obs[77 + 2*k] = h[5];
        obs[77 + 2*k + 1] = h[6];
    }
    // Speeds 20, 30, 40, 50 and 60 steps ago, adjacent to recent speeds.
    for (int k = 0; k < 5; k++) {
        int lag = 20 + 10*k;
        int index = (e->ego_history_head + EGO_HISTORY_CAPACITY - lag) % EGO_HISTORY_CAPACITY;
        obs[72 + k] = e->ego_history[index][4] / MAX_SPEED;
    }
    if (dynamics_model == JERK) {
        obs[97] = e->steering_angle / M_PI;
        obs[97 + 1] = e->a_long < 0 ? e->a_long / (-JERK_LONG[0]) : e->a_long / JERK_LONG[3];
        obs[97 + 2] = e->a_lat / JERK_LAT[2];
    }
}

void compute_observations(Drive* env, ActorObservationGrid* actor_grid) {
    int ego_dim = drive_ego_features(env->dynamics_model);
    int max_obs = ego_dim + 7*(MAX_VISIBLE_AGENTS - 1) + 7*MAX_ROAD_SEGMENT_OBSERVATIONS;
    memset(env->observations, 0, max_obs*env->active_agent_count*sizeof(float));
    float (*observations)[max_obs] = (float(*)[max_obs])env->observations;
    for(int i = 0; i < env->active_agent_count; i++) {
        float* obs = &observations[i][0];
        Entity* ego_entity = &env->entities[env->active_agent_indices[i]];
        if(ego_entity->type > 3) break;

        float cos_heading = ego_entity->heading_x;
        float sin_heading = ego_entity->heading_y;
        write_ego_observation(ego_entity, env->dynamics_model, obs);

        // Relative Pos of other cars
        int obs_idx = ego_dim;
        int cars_seen = 0;
        const ActorObservationPartners* partners = visible_observation_actors(env, actor_grid, i);
        for(int j = 0; j < partners->count; j++) {
            Entity* other_entity = &env->entities[partners->indices[j]];
            // Store original relative positions
            float dx = other_entity->x - ego_entity->x;
            float dy = other_entity->y - ego_entity->y;
            // Rotate to ego vehicle's frame
            float rel_x = dx*cos_heading + dy*sin_heading;
            float rel_y = -dx*sin_heading + dy*cos_heading;
            // Store observations with correct indexing
            obs[obs_idx] = rel_x * 0.02f;
            obs[obs_idx + 1] = rel_y * 0.02f;
            obs[obs_idx + 2] = other_entity->width / MAX_VEH_WIDTH;
            obs[obs_idx + 3] = other_entity->length / MAX_VEH_LEN;
            // relative heading
            float rel_heading_x = other_entity->heading_x * ego_entity->heading_x +
                     other_entity->heading_y * ego_entity->heading_y;  // cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
            float rel_heading_y = other_entity->heading_y * ego_entity->heading_x -
                                other_entity->heading_x * ego_entity->heading_y;  // sin(a-b) = sin(a)cos(b) - cos(a)sin(b)

            obs[obs_idx + 4] = rel_heading_x;
            obs[obs_idx + 5] = rel_heading_y;
            // obs[obs_idx + 4] = cosf(rel_heading) / MAX_ORIENTATION_RAD;
            // obs[obs_idx + 5] = sinf(rel_heading) / MAX_ORIENTATION_RAD;
            // // relative speed
            float other_speed = sqrtf(other_entity->vx*other_entity->vx + other_entity->vy*other_entity->vy);
            obs[obs_idx + 6] = other_speed / MAX_SPEED;
            cars_seen++;
            obs_idx += 7;  // Move to next observation slot
        }
        int remaining_partner_obs = (MAX_VISIBLE_AGENTS - 1 - cars_seen) * 7;
        memset(&obs[obs_idx], 0, remaining_partner_obs * sizeof(float));
        obs_idx += remaining_partner_obs;
        // map observations
        int grid_idx = getGridIndex(env, ego_entity->x, ego_entity->y);

        int cell_x = 0;
        int cell_y = 0;
        if (grid_idx >= 0) {
            cell_x = grid_idx % env->grid_map->grid_cols;
            cell_y = grid_idx / env->grid_map->grid_cols;
        }
        int road_obs_count = 0;
        int neighbor_count = grid_idx >= 0
            ? env->grid_map->vision_range*env->grid_map->vision_range
            : 0;
        for (int j = 0; j < neighbor_count && road_obs_count < MAX_ROAD_SEGMENT_OBSERVATIONS; j++) {
            int x = cell_x + env->neighbor_offsets[j*2];
            int y = cell_y + env->neighbor_offsets[j*2+1];
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows) continue;
            int neighbor_idx = env->grid_map->grid_cols*y + x; 
            int entity_count = env->grid_map->cell_entities_count[neighbor_idx];
            if (entity_count == 0 || env->grid_map->cells[neighbor_idx] == NULL) continue;

            GridMapEntity* neighbor_entities = env->grid_map->cells[neighbor_idx];
            for (int k = 0; k < entity_count && road_obs_count < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {
                int entity_idx = neighbor_entities[k].entity_idx;
                int geometry_idx = neighbor_entities[k].geometry_idx;

                // Validate entity_idx before accessing
                if(entity_idx < 0 || entity_idx >= env->num_roads) {
                    printf("ERROR: Invalid entity_idx %d (max: %d)\n", entity_idx, env->num_roads-1);
                    continue;
                }

                const StaticRoadEntity* entity = &env->static_road_entities[entity_idx];
                int road_type = effective_road_type(env, entity);
                if (road_type == NONE) continue;

                // Validate geometry_idx before accessing
                if(geometry_idx < 0 || geometry_idx >= entity->array_size - 1) {
                    printf("ERROR: Invalid geometry_idx %d for entity %d (max: %d)\n",
                        geometry_idx, entity_idx, entity->array_size-1);
                    continue;
                }
                float start_x = entity->traj_x[geometry_idx];
                float start_y = entity->traj_y[geometry_idx];
                float end_x = entity->traj_x[geometry_idx+1];
                float end_y = entity->traj_y[geometry_idx+1];
                float mid_x = (start_x + end_x) / 2.0f;
                float mid_y = (start_y + end_y) / 2.0f;
                float rel_x = mid_x - ego_entity->x;
                float rel_y = mid_y - ego_entity->y;
                float x_obs = rel_x*cos_heading + rel_y*sin_heading;
                float y_obs = -rel_x*sin_heading + rel_y*cos_heading;
                float length = relative_distance_2d(mid_x, mid_y, end_x, end_y);
                float line_attribute = 0.0f;
                if (road_type == TRAFFIC_LIGHT_STOP_LINE) {
                    line_attribute = traffic_light_observation_value(road_traffic_light_color(env, entity));
                }
                // Calculate angle from ego to midpoint (vector from ego to midpoint)
                float dx = end_x - mid_x;
                float dy = end_y - mid_y;
                float dx_norm = dx;
                float dy_norm = dy;
                float hypot = sqrtf(dx*dx + dy*dy);
                if(hypot > 0) {
                    dx_norm /= hypot;
                    dy_norm /= hypot;
                }
                // Compute sin and cos of relative angle directly without atan2f
                float cos_angle = dx_norm*cos_heading + dy_norm*sin_heading;
                float sin_angle = -dx_norm*sin_heading + dy_norm*cos_heading;
                obs[obs_idx] = x_obs * 0.02f;
                obs[obs_idx + 1] = y_obs * 0.02f;
                obs[obs_idx + 2] = length / MAX_ROAD_SEGMENT_LENGTH;
                obs[obs_idx + 3] = line_attribute;
                obs[obs_idx + 4] = cos_angle;
                obs[obs_idx + 5] = sin_angle;
                obs[obs_idx + 6] = road_type - 4.0f;
                obs_idx += 7;
                road_obs_count++;
            }
        }
        int remaining_obs = (MAX_ROAD_SEGMENT_OBSERVATIONS - road_obs_count) * 7;
        // Set the entire block to 0 at once
        memset(&obs[obs_idx], 0, remaining_obs * sizeof(float));

        if (env->export_replay_enabled && env->export_log.observations &&
            i < env->export_log.num_agents && env->timestep < env->export_log.capacity) {
            float (*export_obs)[env->export_log.capacity][max_obs] = (float (*)[env->export_log.capacity][max_obs]) env->export_log.observations;
            memcpy(&export_obs[i][env->timestep][0], obs, max_obs * sizeof(float));
        }
    }
}

static int find_forward_projection_on_lane(const StaticRoadEntity* lane, Entity* agent, int* out_segment_idx, float* out_fraction) {
    int best_idx = -1;
    float best_dist_sq = 1e30f;

    for (int i = 1; i < lane->array_size; i++) {
        float x0 = lane->traj_x[i - 1];
        float y0 = lane->traj_y[i - 1];
        float x1 = lane->traj_x[i];
        float y1 = lane->traj_y[i];
        float dx = x1 - x0;
        float dy = y1 - y0;
        float seg_len_sq = dx * dx + dy * dy;
        if (seg_len_sq < 1e-6f) continue;

        float to_agent_x = agent->x - x0;
        float to_agent_y = agent->y - y0;
        float t = (to_agent_x * dx + to_agent_y * dy) / seg_len_sq;
        if (t < 0.0f) t = 0.0f;
        else if (t > 1.0f) t = 1.0f;

        float proj_x = x0 + t * dx;
        float proj_y = y0 + t * dy;

        float rel_x = proj_x - agent->x;
        float rel_y = proj_y - agent->y;
        float forward = rel_x * agent->heading_x + rel_y * agent->heading_y;
        if (forward < 0.0f) continue;

        float dist_sq = rel_x * rel_x + rel_y * rel_y;
        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_idx = i;
            *out_fraction = t;
        }
    }

    if (best_idx != -1) {
        *out_segment_idx = best_idx;
        return 1;
    }

    return 0;
}

void compute_new_goal(Drive* env, int agent_idx) {
    Entity* agent = &env->entities[agent_idx];
    int current_lane = agent->current_lane_idx;

    if (current_lane < 0 || current_lane >= env->num_roads) return; // No current lane

    // Target distance: 40m ahead along the lane topology from agent's current position
    float target_distance = 40.0f;
    int current_entity = current_lane;
    const StaticRoadEntity* lane = &env->static_road_entities[current_entity];

    int initial_segment_idx = 1;
    float initial_fraction = 0.0f;
    if (!find_forward_projection_on_lane(lane, agent, &initial_segment_idx, &initial_fraction)) {
        int forward_idx = -1;
        for (int i = 0; i < lane->array_size; i++) {
            float to_point_x = lane->traj_x[i] - agent->x;
            float to_point_y = lane->traj_y[i] - agent->y;
            float dot = to_point_x * agent->heading_x + to_point_y * agent->heading_y;
            if (dot > 0.0f) {
                forward_idx = i;
                break;
            }
        }

        if (forward_idx == -1) {
            agent->goal_position_x = lane->traj_x[lane->array_size - 1];
            agent->goal_position_y = lane->traj_y[lane->array_size - 1];
            agent->sampled_new_goal = 0;
            return;
        }

        initial_segment_idx = forward_idx;
        if (initial_segment_idx == 0) initial_segment_idx = 1;
        initial_fraction = 0.0f;
    }

    float remaining_distance = target_distance;
    int first_lane = 1;

    // Traverse the topology graph starting from the vehicle's position forward
    while (current_entity != -1) {
        lane = &env->static_road_entities[current_entity];

        int start_idx = first_lane ? initial_segment_idx : 1;
        // Ensure start_idx is at least 1 to avoid accessing traj_x[i-1] with i=0
        if (start_idx < 1) start_idx = 1;
        first_lane = 0;

        for (int i = start_idx; i < lane->array_size; i++) {
            float prev_x = lane->traj_x[i - 1];
            float prev_y = lane->traj_y[i - 1];
            float next_x = lane->traj_x[i];
            float next_y = lane->traj_y[i];
            float seg_dx = next_x - prev_x;
            float seg_dy = next_y - prev_y;
            float segment_length = relative_distance_2d(prev_x, prev_y, next_x, next_y);

            if (remaining_distance <= segment_length) {
                agent->goal_position_x = next_x;
                agent->goal_position_y = next_y;
                agent->sampled_new_goal = 0;
                return;
            }

            remaining_distance -= segment_length;
        }

        int connected_lanes[5];
        int num_connected = getNextLanes(env->topology_graph, current_entity, connected_lanes, 5);

        if (num_connected == 0) {
            agent->goal_position_x = lane->traj_x[lane->array_size - 1];
            agent->goal_position_y = lane->traj_y[lane->array_size - 1];
            agent->sampled_new_goal = 0;
            return; // No further lanes to traverse
        }

        int random_idx = agent_idx % num_connected;
        current_entity = connected_lanes[random_idx];
    }
}

int short_term_goals() {
    return getenv("SHORT_TERM_GOALS") && strcmp(getenv("SHORT_TERM_GOALS"), "1") == 0;
}

int subsample_scenario() {
    return getenv("SUBSAMPLE_SCENARIO") && strcmp(getenv("SUBSAMPLE_SCENARIO"), "1") == 0;
}

void c_reset(Drive* env){
    memset(env->rewards, 0, env->active_agent_count * sizeof(float));
    memset(env->terminals, 0, env->active_agent_count * sizeof(unsigned char));
    memset(env->truncations, 0, env->active_agent_count * sizeof(unsigned char));

    if (env->export_replay_enabled && env->export_log.traj_x == NULL) {
        int export_all_agents = getenv("EXPORT_ALL_AGENTS") && strcmp(getenv("EXPORT_ALL_AGENTS"), "1") == 0;
        env->export_log.num_agents = (env->active_agent_count < 64 || export_all_agents)
            ? env->active_agent_count : 64;
        env->export_log.num_actors = env->export_log.num_agents + env->static_car_count;
        env->export_log.actor_indices = malloc((size_t)env->export_log.num_actors * sizeof(int));
        for (int i = 0; i < env->export_log.num_actors; i++)
            env->export_log.actor_indices[i] = i < env->export_log.num_agents
                ? env->active_agent_indices[i] : env->static_car_indices[i - env->export_log.num_agents];
        env->export_log.capacity = (env->scenario_length < MAX_EXPORT_STEPS - 2 || export_all_agents)
            ? env->scenario_length + 2 : MAX_EXPORT_STEPS;
        if (env->export_log.capacity < 1) env->export_log.capacity = 1;
        size_t actor_steps = (size_t)env->export_log.num_actors * env->export_log.capacity;
        env->export_log.traj_x = calloc(actor_steps, sizeof(float));
        env->export_log.traj_y = calloc(actor_steps, sizeof(float));
        env->export_log.traj_z = calloc(actor_steps, sizeof(float));
        env->export_log.traj_vx = calloc(actor_steps, sizeof(float));
        env->export_log.traj_vy = calloc(actor_steps, sizeof(float));
        env->export_log.traj_vz = calloc(actor_steps, sizeof(float));
        env->export_log.heading = calloc(actor_steps, sizeof(float));
        env->export_log.valid = calloc(actor_steps, sizeof(int));
        env->export_log.traffic_light_states =
            calloc((size_t)env->num_traffic_lights * env->export_log.capacity, sizeof(int));

        size_t agent_steps = (size_t)env->export_log.num_agents * env->export_log.capacity;
        int ego_dim = drive_ego_features(env->dynamics_model);
        int max_obs = ego_dim + 7*(MAX_VISIBLE_AGENTS - 1) + 7*MAX_ROAD_SEGMENT_OBSERVATIONS;
        env->export_log.observations = calloc(agent_steps * max_obs, sizeof(float));
        env->export_log.reward_components =
            calloc(agent_steps * REPLAY_REWARD_COMPONENT_COUNT, sizeof(float));
        env->export_log.action_indices =
            malloc(agent_steps * REPLAY_ACTION_COMPONENT_COUNT * sizeof(int));
    }
    if (env->export_replay_enabled && env->export_log.valid)
        memset(env->export_log.valid, 0, (size_t)env->export_log.num_actors * env->export_log.capacity * sizeof(int));
    if (env->export_replay_enabled && env->export_log.traffic_light_states)
        memset(env->export_log.traffic_light_states, 0, (size_t)env->num_traffic_lights * env->export_log.capacity * sizeof(int));
    if (env->export_replay_enabled && env->export_log.reward_components != NULL) {
        memset(
            env->export_log.reward_components,
            0,
            (size_t)env->export_log.num_agents * env->export_log.capacity * REPLAY_REWARD_COMPONENT_COUNT * sizeof(float));
    }
    if (env->export_replay_enabled && env->export_log.action_indices != NULL) {
        size_t action_count = (size_t)env->export_log.num_agents * env->export_log.capacity * REPLAY_ACTION_COMPONENT_COUNT;
        for (size_t i = 0; i < action_count; i++) {
            env->export_log.action_indices[i] = -1;
        }
    }
    if (subsample_scenario() && env->scenario_length > 91) {
        env->init_steps = time(NULL) % (env->scenario_length - 91);
    }
    env->timestep = env->init_steps;
    set_start_position(env);
    init_traffic_lights(env);
    record_replay_trajectories(env);
    for(int x = 0;x<env->active_agent_count; x++){
        env->logs[x] = (Log){0};
        int agent_idx = env->active_agent_indices[x];
        env->entities[agent_idx].respawn_timestep = -1;
        env->entities[agent_idx].respawn_count = 0;
        env->entities[agent_idx].collided_before_goal = 0;
        env->entities[agent_idx].reached_goal_this_episode = 0;
        env->entities[agent_idx].reached_final_goal = 0;
        env->entities[agent_idx].sampled_new_goal = 0;
        env->entities[agent_idx].multi_goal_idx = 0;
        env->entities[agent_idx].metrics_array[COLLISION_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[OFFROAD_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[LANE_ALIGNED_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[AVG_DISPLACEMENT_ERROR_IDX] = 0.0f;
        env->entities[agent_idx].cumulative_displacement = 0.0f;
        env->entities[agent_idx].displacement_sample_count = 0;

        if (env->use_goal_generation) {
            env->entities[agent_idx].goal_position_x = env->entities[agent_idx].init_goal_x;
            env->entities[agent_idx].goal_position_y = env->entities[agent_idx].init_goal_y;
        } else if (env->entities[agent_idx].multi_goal_count > 0) {
            env->entities[agent_idx].goal_position_x = env->entities[agent_idx].multi_goal_position_x[0];
            env->entities[agent_idx].goal_position_y = env->entities[agent_idx].multi_goal_position_y[0];
            env->entities[agent_idx].goal_position_z = env->entities[agent_idx].multi_goal_position_z[0];
        }

        if (short_term_goals() || subsample_scenario()) {
            int goal_timestep = fmin(env->timestep + 90, env->entities[agent_idx].array_size - 1);
            env->entities[agent_idx].goal_position_x = env->entities[agent_idx].traj_x[goal_timestep];
            env->entities[agent_idx].goal_position_y = env->entities[agent_idx].traj_y[goal_timestep];
            env->entities[agent_idx].goal_timestep = goal_timestep;
        }

        compute_agent_metrics(env, agent_idx);
    }

    ActorObservationGrid actor_grid = {0};
    build_actor_observation_grid(env, &actor_grid);
    compute_observations(env, &actor_grid);
    free_actor_observation_grid(&actor_grid);
}

void respawn_agent(Drive* env, int agent_idx){
    env->entities[agent_idx].x = env->entities[agent_idx].traj_x[0];
    env->entities[agent_idx].y = env->entities[agent_idx].traj_y[0];
    env->entities[agent_idx].prev_x = env->entities[agent_idx].x;
    env->entities[agent_idx].prev_y = env->entities[agent_idx].y;
    env->entities[agent_idx].heading = env->entities[agent_idx].traj_heading[0];
    env->entities[agent_idx].heading_x = cosf(env->entities[agent_idx].heading);
    env->entities[agent_idx].heading_y = sinf(env->entities[agent_idx].heading);
    env->entities[agent_idx].vx = env->entities[agent_idx].traj_vx[0];
    env->entities[agent_idx].vy = env->entities[agent_idx].traj_vy[0];
    env->entities[agent_idx].metrics_array[COLLISION_IDX] = 0.0f;
    env->entities[agent_idx].metrics_array[OFFROAD_IDX] = 0.0f;
    env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX] = 0.0f;
    env->entities[agent_idx].metrics_array[LANE_ALIGNED_IDX] = 0.0f;
    env->entities[agent_idx].metrics_array[AVG_DISPLACEMENT_ERROR_IDX] = 0.0f;
    env->entities[agent_idx].cumulative_displacement = 0.0f;
    env->entities[agent_idx].displacement_sample_count = 0;
    env->entities[agent_idx].respawn_timestep = env->timestep;
    env->entities[agent_idx].a_long = 0.0f;
    env->entities[agent_idx].a_lat = 0.0f;
    env->entities[agent_idx].jerk_long = 0.0f;
    env->entities[agent_idx].jerk_lat = 0.0f;
    env->entities[agent_idx].steering_angle = 0.0f;
    env->entities[agent_idx].multi_goal_idx = 0;
    clear_action_queue(env, &env->entities[agent_idx]);
    reset_ego_history(&env->entities[agent_idx]);
}

float compute_distance_to_goal(Entity* e) {
    // Compute distance from the goal to the closest point on the vehicle
    // This is needed for really large vehicles like buses that otherwise
    // may have hard time going through the goals.

    float cos_heading = e->heading_x;
    float sin_heading = e->heading_y;

    float goal_x = e->goal_position_x - e->x;
    float goal_y = e->goal_position_y - e->y;

    float rel_goal_x = goal_x*cos_heading + goal_y*sin_heading;
    float rel_goal_y = -goal_x*sin_heading + goal_y*cos_heading;

    float dx = fmaxf(fabsf(rel_goal_x) - 0.5f * e->length, 0.0f);
    float dy = fmaxf(fabsf(rel_goal_y) - 0.5f * e->width, 0.0f);
    return sqrtf(dx * dx + dy * dy);
}

void c_step(Drive* env){
    memset(env->rewards, 0, env->active_agent_count * sizeof(float));
    memset(env->terminals, 0, env->active_agent_count * sizeof(unsigned char));
    memset(env->truncations, 0, env->active_agent_count * sizeof(unsigned char));

    env->timestep++;

    if (env->timestep > env->scenario_length + 1 && env->play_exactly_once) return;

    if (env->timestep == env->scenario_length) {
        add_log(env);
        memset(env->truncations, 1, env->active_agent_count * sizeof(unsigned char));
    }

    if (env->timestep == env->scenario_length + 1 && !env->play_exactly_once) {
        c_reset(env);
        return;
    }

    update_traffic_lights(env);

    // Move statix experts
    for (int i = 0; i < env->expert_static_car_count; i++) {
        int expert_idx = env->expert_static_car_indices[i];
        move_expert(env, env->actions, expert_idx);
    }
    // Process actions for all active agents
    for(int i = 0; i < env->active_agent_count; i++){
        env->logs[i].score = 0.0f;
        env->logs[i].episode_length += 1;
        int agent_idx = env->active_agent_indices[i];
        move_dynamics(env, i, agent_idx);
        record_speed_count(env, &env->entities[agent_idx]);
    }
    if (env->num_eval_traffic_light_lines > 0 && env->timestep < env->scenario_length)
        update_eval_traffic_light_lines(env);
    for(int i = 0; i < env->active_agent_count; i++){
        int agent_idx = env->active_agent_indices[i];
        Entity* e = &env->entities[agent_idx];
        int prev_collision_state = env->entities[agent_idx].collision_state;
        //if(env->entities[agent_idx].respawn_timestep != -1) continue;
        compute_agent_metrics(env, agent_idx);
        int collision_state = env->entities[agent_idx].collision_state;

        if(collision_state > 0){
            if (collision_state == VEHICLE_COLLISION && prev_collision_state != VEHICLE_COLLISION) {
                float speed = sqrtf(e->vx*e->vx + e->vy*e->vy);
                float collision_reward = -e->collision_reward_coefficient * (1 + 0.1 * speed);
                env->rewards[i] = collision_reward;
                record_reward_component(env, i, REWARD_COMPONENT_COLLISION, collision_reward);
                env->logs[i].collision_rate = 1.0f;
                env->logs[i].avg_collisions_per_agent += 1.0f;
                
                // With high coefficient the episode will end and any future
                // reward will not be counted to make it harder for the agent
                // to reward hack.
                if (e->collision_reward_coefficient >= 0.5f) {
                    env->terminals[i] = 1;
                }

                Entity* collided_with = &env->entities[e->collided_with_agent_idx];
                if (collided_with->mark_as_expert) {
                    int coef_bin = clipped_scaled_bin(e->collision_reward_coefficient, 100.0f, COEFFICIENT_COUNT_BINS);
                    env->expert_collision_counts[coef_bin]++;
                } else {
                    float coef1 = fminf(e->collision_reward_coefficient, collided_with->collision_reward_coefficient);
                    float coef2 = fmaxf(e->collision_reward_coefficient, collided_with->collision_reward_coefficient);
                    int coef1_bin = clipped_scaled_bin(coef1, 100.0f, COLLISION_COUNT_BINS);
                    int coef2_bin = clipped_scaled_bin(coef2, 100.0f, COLLISION_COUNT_BINS);
                    int joint_bin_idx = coef1_bin * COLLISION_COUNT_BINS + coef2_bin - coef1_bin * (coef1_bin + 1) / 2;
                    env->collision_counts[joint_bin_idx]++;
                }
            } else if (collision_state == VEHICLE_COLLISION) {
                // Continuing contact costs 1% per step, without another collision event.
                float speed = sqrtf(e->vx*e->vx + e->vy*e->vy);
                float collision_reward = -0.01f * e->collision_reward_coefficient * (1 + 0.1f * speed);
                env->rewards[i] += collision_reward;
                record_reward_component(env, i, REWARD_COMPONENT_COLLISION, collision_reward);
            } else if (collision_state == OFFROAD && prev_collision_state != OFFROAD) {
                float speed = sqrtf(e->vx*e->vx + e->vy*e->vy);
                float offroad_reward = -e->offroad_reward_coefficient * (1 + 0.1 * speed);
                env->rewards[i] = offroad_reward;
                env->logs[i].offroad_rate = 1.0f;
                record_reward_component(env, i, REWARD_COMPONENT_OFFROAD, offroad_reward);
                env->logs[i].avg_offroad_per_agent += 1.0f;
                int offroad_bin = clipped_scaled_bin(e->offroad_reward_coefficient, 100.0f, COEFFICIENT_COUNT_BINS);
                env->offroad_counts[offroad_bin]++;
                if (e->offroad_reward_coefficient >= 0.5f) {
                    env->terminals[i] = 1;
                }
            } else if (collision_state == RED_TRAFFIC_LIGHT_CROSSED) {
                // Geometry already requires new overlap with a specific red line.
                float stop_line_reward = -e->stop_line_reward_coefficient;
                env->rewards[i] = stop_line_reward;
                env->logs[i].stop_line_rate = 1.0f;
                record_reward_component(env, i, REWARD_COMPONENT_STOP_LINE, stop_line_reward);
                env->logs[i].avg_stop_line_per_agent += 1.0f;
                int stop_line_bin = clipped_scaled_bin(e->stop_line_reward_coefficient, 100.0f, COEFFICIENT_COUNT_BINS);
                env->stop_line_counts[stop_line_bin]++;
            }
            if(!env->entities[agent_idx].reached_goal_this_episode){
                env->entities[agent_idx].collided_before_goal = 1;
            }
        }

        int lane_aligned = e->metrics_array[LANE_ALIGNED_IDX];
        float lateral_offset = e->metrics_array[LANE_LATERAL_OFFSET_IDX];
        float lane_heading_angle_diff = e->lane_heading_angle_diff;

        env->logs[i].lane_alignment_rate += lane_aligned;

        if (lateral_offset > -100.0f && lane_heading_angle_diff < M_PI/3.0f) {
            float lane_center_reward = e->lane_center_reward_coefficient * (-fabsf(e->lane_center_bias - lateral_offset) + 0.05f / expf(fabsf(lateral_offset - e->lane_center_bias) - 0.5f));
            env->rewards[i] += lane_center_reward;
            record_reward_component(env, i, REWARD_COMPONENT_LANE_CENTER, lane_center_reward);
            if (e->lane_center_reward_coefficient > 0.0f) {
                record_lane_center_offset_count(env, lateral_offset, e->lane_center_bias);
            }
        }

        float speed = sqrtf(e->vx*e->vx + e->vy*e->vy);

        if (lateral_offset > -100.0f) {
            env->logs[i].lane_lateral_offset += fabsf(lateral_offset);
            env->logs[i].lane_heading_angle_diff += lane_heading_angle_diff;
            env->logs[i].close_to_lane += 1.0f;

            float lane_align_reward = e->lane_align_reward_coefficient * (fminf(cosf(lane_heading_angle_diff), 0.0f) + 0.0025f * (1 - lane_heading_angle_diff / M_PI * 2.0f));
            env->rewards[i] += lane_align_reward;
            record_reward_component(env, i, REWARD_COMPONENT_LANE_ALIGNMENT, lane_align_reward);

            if (speed > 2.5f) {
                float velocity_reward = 2.5e-4f * fmax(cosf(lane_heading_angle_diff), 0.0f);
                env->rewards[i] += velocity_reward;
                record_reward_component(env, i, REWARD_COMPONENT_VELOCITY, velocity_reward);
            }
        }

        float distance_to_goal = compute_distance_to_goal(&env->entities[agent_idx]);

        // Reward agent if it is within X meters of goal
        if (!env->entities[agent_idx].reached_final_goal && distance_to_goal < env->entities[agent_idx].goal_radius) {
            if (short_term_goals()) {
                env->rewards[i] += env->reward_goal;
                record_reward_component(env, i, REWARD_COMPONENT_GOAL, env->reward_goal);
                env->logs[i].num_goals_reached += 1;
                env->entities[agent_idx].reached_goal_this_episode = 1;
                env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX] = 1.0f;
                if (env->entities[agent_idx].goal_timestep < env->entities[agent_idx].array_size - 1) {
                    int t = fmin(env->entities[agent_idx].goal_timestep + 80,
                        env->entities[agent_idx].array_size - 1);
                    env->entities[agent_idx].goal_timestep = t;
                    env->entities[agent_idx].goal_position_x = env->entities[agent_idx].traj_x[t];
                    env->entities[agent_idx].goal_position_y = env->entities[agent_idx].traj_y[t];
                } else {
                    env->entities[agent_idx].reached_final_goal = 1;
                }
            } else if (env->entities[agent_idx].multi_goal_count > 0) {
                // The vehicle must slow down to 1 m/s to obtain the last goal reward
                if (speed < 1 || env->entities[agent_idx].multi_goal_idx < env->entities[agent_idx].multi_goal_count - 1) {
                    float reward = env->reward_goal;
                    env->rewards[i] += reward;
                    record_reward_component(env, i, REWARD_COMPONENT_GOAL, reward);
                    env->entities[agent_idx].sampled_new_goal = 1;
                    env->logs[i].num_goals_reached += 1;
                    env->entities[agent_idx].reached_goal_this_episode = 1;
                    env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX] = 1.0f;

                    int goal_idx = ++env->entities[agent_idx].multi_goal_idx;
                    if (goal_idx < env->entities[agent_idx].multi_goal_count) {
                        env->entities[agent_idx].goal_position_x = env->entities[agent_idx].multi_goal_position_x[goal_idx];
                        env->entities[agent_idx].goal_position_y = env->entities[agent_idx].multi_goal_position_y[goal_idx];
                        env->entities[agent_idx].goal_position_z = env->entities[agent_idx].multi_goal_position_z[goal_idx];
                    } else {
                        env->entities[agent_idx].reached_final_goal = 1;
                    }
                }
            } else {
                env->rewards[i] += env->reward_goal;
                record_reward_component(env, i, REWARD_COMPONENT_GOAL, env->reward_goal);
                env->entities[agent_idx].sampled_new_goal = 1;
                env->logs[i].num_goals_reached += 1;
                env->entities[agent_idx].reached_goal_this_episode = 1;
                env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX] = 1.0f;
                env->terminals[i] = 1;
                env->entities[agent_idx].reached_final_goal = 1;
            }
	    }

        if (env->use_goal_generation && env->entities[agent_idx].sampled_new_goal) {
            compute_new_goal(env, agent_idx);
        }

        // Apply ADE reward
        float current_ade = env->entities[agent_idx].metrics_array[AVG_DISPLACEMENT_ERROR_IDX];
        if(current_ade > 0.0f && env->reward_ade != 0.0f) {
            float ade_reward = env->reward_ade * current_ade;
            env->rewards[i] += ade_reward;
            record_reward_component(env, i, REWARD_COMPONENT_ADE, ade_reward);
        }
        env->logs[i].avg_displacement_error = current_ade;
    }

    if (env->num_exit_orders > 0 && env->timestep < env->scenario_length) update_exit_orders(env);
    record_replay_trajectories(env);

    ActorObservationGrid actor_grid = {0};
    build_actor_observation_grid(env, &actor_grid);
    apply_safety_penalties(env, &actor_grid);
    compute_observations(env, &actor_grid);
    free_actor_observation_grid(&actor_grid);

    if (env->timestep == env->scenario_length && env->export_replay_enabled && env->export_log.traj_x)
        saveReplayToFile(env);
}

const Color STONE_GRAY = (Color){80, 80, 80, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};
const Color LIGHTGREEN = (Color){152, 255, 152, 255};

typedef struct AgentLabel {
    Vector2 screen_pos;
    float speed_kph;
    bool visible;
} AgentLabel;

typedef struct Client Client;
struct Client {
    float width;
    float height;
    Texture2D puffers;
    Vector3 camera_target;
    float camera_zoom;
    Camera3D camera;
    Model cars[6];
    int car_assignments[MAX_AGENTS];  // To keep car model assignments consistent per vehicle
    Vector3 default_camera_position;
    Vector3 default_camera_target;
};

Client* make_client(Drive* env){
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = 1280;
    client->height = 704;
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(client->width, client->height, "PufferLib Ray GPU Drive");
    SetTargetFPS(30);
    client->puffers = LoadTexture("resources/puffers_128.png");
    client->cars[0] = LoadModel("resources/drive/RedCar.glb");
    client->cars[1] = LoadModel("resources/drive/WhiteCar.glb");
    client->cars[2] = LoadModel("resources/drive/BlueCar.glb");
    client->cars[3] = LoadModel("resources/drive/YellowCar.glb");
    client->cars[4] = LoadModel("resources/drive/GreenCar.glb");
    client->cars[5] = LoadModel("resources/drive/GreyCar.glb");
    for (int i = 0; i < MAX_AGENTS; i++) {
        client->car_assignments[i] = (rand() % 4) + 1;
    }
    // Get initial target position from first active agent
    Vector3 target_pos = {
        0,
        0,  // Y is up
        1   // Z is depth
    };

    // Set up camera to look at target from above and behind
    client->default_camera_position = (Vector3){
        0,           // Same X as target
        120.0f,   // 20 units above target
        175.0f    // 20 units behind target
    };
    client->default_camera_target = target_pos;
    client->camera.position = client->default_camera_position;
    client->camera.target = client->default_camera_target;
    client->camera.up = (Vector3){ 0.0f, -1.0f, 0.0f };  // Y is up
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    client->camera_zoom = 1.0f;
    return client;
}

// Camera control functions
void handle_camera_controls(Client* client) {
    static Vector2 prev_mouse_pos = {0};
    static bool is_dragging = false;
    float camera_move_speed = 0.5f;

    // Handle mouse drag for camera movement
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        prev_mouse_pos = GetMousePosition();
        is_dragging = true;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        is_dragging = false;
    }

    if (is_dragging) {
        Vector2 current_mouse_pos = GetMousePosition();
        Vector2 delta = {
            (current_mouse_pos.x - prev_mouse_pos.x) * camera_move_speed,
            -(current_mouse_pos.y - prev_mouse_pos.y) * camera_move_speed
        };

        // Update camera position (only X and Y)
        client->camera.position.x += delta.x;
        client->camera.position.y += delta.y;

        // Update camera target (only X and Y)
        client->camera.target.x += delta.x;
        client->camera.target.y += delta.y;

        prev_mouse_pos = current_mouse_pos;
    }

    // Handle mouse wheel for zoom
    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        float zoom_factor = 1.0f - (wheel * 0.1f);
        // Calculate the current direction vector from target to position
        Vector3 direction = {
            client->camera.position.x - client->camera.target.x,
            client->camera.position.y - client->camera.target.y,
            client->camera.position.z - client->camera.target.z
        };

        // Scale the direction vector by the zoom factor
        direction.x *= zoom_factor;
        direction.y *= zoom_factor;
        direction.z *= zoom_factor;

        // Update the camera position based on the scaled direction
        client->camera.position.x = client->camera.target.x + direction.x;
        client->camera.position.y = client->camera.target.y + direction.y;
        client->camera.position.z = client->camera.target.z + direction.z;
    }
}

void draw_agent_obs(Drive* env, int agent_index, int mode, int obs_only, int lasers){
    // Diamond dimensions
    float diamond_height = 3.0f;    // Total height of diamond
    float diamond_width = 1.5f;     // Width of diamond
    float diamond_z = 8.0f;         // Base Z position

    // Define diamond points
    Vector3 top_point = (Vector3){0.0f, 0.0f, diamond_z + diamond_height/2};     // Top point
    Vector3 bottom_point = (Vector3){0.0f, 0.0f, diamond_z - diamond_height/2};  // Bottom point
    Vector3 front_point = (Vector3){0.0f, diamond_width/2, diamond_z};           // Front point
    Vector3 back_point = (Vector3){0.0f, -diamond_width/2, diamond_z};           // Back point
    Vector3 left_point = (Vector3){-diamond_width/2, 0.0f, diamond_z};           // Left point
    Vector3 right_point = (Vector3){diamond_width/2, 0.0f, diamond_z};           // Right point

    // Draw the diamond faces
    // Top pyramid

    if(mode ==0){
        DrawTriangle3D(top_point, front_point, right_point, PUFF_CYAN);    // Front-right face
        DrawTriangle3D(top_point, right_point, back_point, PUFF_CYAN);     // Back-right face
        DrawTriangle3D(top_point, back_point, left_point, PUFF_CYAN);      // Back-left face
        DrawTriangle3D(top_point, left_point, front_point, PUFF_CYAN);     // Front-left face

        // Bottom pyramid
        DrawTriangle3D(bottom_point, right_point, front_point, PUFF_CYAN); // Front-right face
        DrawTriangle3D(bottom_point, back_point, right_point, PUFF_CYAN);  // Back-right face
        DrawTriangle3D(bottom_point, left_point, back_point, PUFF_CYAN);   // Back-left face
        DrawTriangle3D(bottom_point, front_point, left_point, PUFF_CYAN);  // Front-left face
    }
    if(!IsKeyDown(KEY_LEFT_CONTROL) && obs_only==0){
        return;
    }

    int ego_dim = drive_ego_features(env->dynamics_model);
    int max_obs = ego_dim + 7*(MAX_VISIBLE_AGENTS - 1) + 7*MAX_ROAD_SEGMENT_OBSERVATIONS;
    float (*observations)[max_obs] = (float(*)[max_obs])env->observations;
    float* agent_obs = &observations[agent_index][0];
    // self
    int active_idx = env->active_agent_indices[agent_index];
    float heading_self_x = env->entities[active_idx].heading_x;
    float heading_self_y = env->entities[active_idx].heading_y;
    float px = env->entities[active_idx].x;
    float py = env->entities[active_idx].y;
    // draw goal
    float goal_x = agent_obs[14] * 200;
    float goal_y = agent_obs[15] * 200;
    if(mode == 0 ){
        DrawSphere((Vector3){goal_x, goal_y, 1}, 0.5f, LIGHTGREEN);
        DrawCircle3D((Vector3){goal_x, goal_y, 0.1f}, env->goal_radius, (Vector3){0, 0, 1}, 90.0f, Fade(LIGHTGREEN, 0.3f));
    }

    if (mode == 1){
        float goal_x_world = px + (goal_x * heading_self_x - goal_y*heading_self_y);
        float goal_y_world = py + (goal_x * heading_self_y + goal_y*heading_self_x);
        DrawSphere((Vector3){goal_x_world, goal_y_world, 1}, 0.5f, LIGHTGREEN);
        DrawCircle3D((Vector3){goal_x_world, goal_y_world, 0.1f}, env->goal_radius, (Vector3){0, 0, 1}, 90.0f, Fade(LIGHTGREEN, 0.3f));
    }
    // First draw other agent observations
    int obs_idx = ego_dim;  // Start after ego obs
    for(int j = 0; j < MAX_VISIBLE_AGENTS - 1; j++) {
        if(agent_obs[obs_idx] == 0 || agent_obs[obs_idx + 1] == 0) {
            obs_idx += 7;  // Move to next agent observation
            continue;
        }
        // Draw position of other agents
        float x = agent_obs[obs_idx] * 50;
        float y = agent_obs[obs_idx + 1] * 50;
        if(lasers && mode == 0){
            DrawLine3D(
                (Vector3){0, 0, 0},
                (Vector3){x, y, 1},
                ORANGE
            );
        }

        float partner_x = px + (x*heading_self_x - y*heading_self_y);
        float partner_y = py + (x*heading_self_y + y*heading_self_x);
        if(lasers && mode ==1){
            DrawLine3D(
                (Vector3){px, py, 1},
                (Vector3){partner_x,partner_y,1},
                ORANGE
            );
        }

        float half_width = 0.5*agent_obs[obs_idx + 2]*MAX_VEH_WIDTH;
        float half_len = 0.5*agent_obs[obs_idx + 3]*MAX_VEH_LEN;
        float theta_x = agent_obs[obs_idx + 4];
        float theta_y = agent_obs[obs_idx + 5];
        float partner_angle = atan2f(theta_y, theta_x);
        float cos_heading = cosf(partner_angle);
        float sin_heading = sinf(partner_angle);
        Vector3 corners[4] = {
            (Vector3){
                x + (half_len * cos_heading - half_width * sin_heading),
                y + (half_len * sin_heading + half_width * cos_heading),
                1
            },
            (Vector3){
                x + (half_len * cos_heading + half_width * sin_heading),
                y + (half_len * sin_heading - half_width * cos_heading),
                1
            },
           (Vector3){
                x + (-half_len * cos_heading + half_width * sin_heading),
                y + (-half_len * sin_heading - half_width * cos_heading),
                1
            },
           (Vector3){
                x + (-half_len * cos_heading - half_width * sin_heading),
                y + (-half_len * sin_heading + half_width * cos_heading),
                1
            },
        };

        if(mode ==0){
            for (int j = 0; j < 4; j++) {
                DrawLine3D(corners[j], corners[(j+1)%4], ORANGE);
            }
        }

        if(mode ==1){
            Vector3 world_corners[4];
            for (int j = 0; j < 4; j++) {
                float lx = corners[j].x;
                float ly = corners[j].y;

                world_corners[j].x = px + (lx * heading_self_x - ly * heading_self_y);
                world_corners[j].y = py + (lx * heading_self_y + ly * heading_self_x);
                world_corners[j].z = 1;
            }
            for (int j = 0; j < 4; j++) {
                DrawLine3D(world_corners[j], world_corners[(j+1)%4], ORANGE);
            }
        }

        // draw an arrow above the car pointing in the direction that the partner is going
        float arrow_length = 7.5f;
        float arrow_x = x + arrow_length*cosf(partner_angle);
        float arrow_y = y + arrow_length*sinf(partner_angle);
        float arrow_x_world;
        float arrow_y_world;
        if(mode ==0){
            DrawLine3D((Vector3){x, y, 1}, (Vector3){arrow_x, arrow_y, 1}, PUFF_WHITE);
        }
        if(mode == 1){
            arrow_x_world = px + (arrow_x * heading_self_x - arrow_y*heading_self_y);
            arrow_y_world = py + (arrow_x * heading_self_y + arrow_y*heading_self_x);
            DrawLine3D((Vector3){partner_x, partner_y, 1}, (Vector3){arrow_x_world, arrow_y_world, 1}, PUFF_WHITE);
        }
        // Calculate perpendicular offsets for arrow head
        float arrow_size = 2.0f;  // Size of the arrow head
        float dx = arrow_x - x;
        float dy = arrow_y - y;
        float length = sqrtf(dx*dx + dy*dy);
        if (length > 0) {
            // Normalize direction vector
            dx /= length;
            dy /= length;

            // Calculate perpendicular vector

            float perp_x = -dy * arrow_size;
            float perp_y = dx * arrow_size;

            float arrow_x_end1 = arrow_x - dx*arrow_size + perp_x;
            float arrow_y_end1 = arrow_y - dy*arrow_size + perp_y;
            float arrow_x_end2 = arrow_x - dx*arrow_size - perp_x;
            float arrow_y_end2 = arrow_y - dy*arrow_size - perp_y;

            // Draw the two lines forming the arrow head
            if(mode ==0){
                DrawLine3D(
                    (Vector3){arrow_x, arrow_y, 1},
                    (Vector3){arrow_x_end1, arrow_y_end1, 1},
                    PUFF_WHITE
                );
                DrawLine3D(
                    (Vector3){arrow_x, arrow_y, 1},
                    (Vector3){arrow_x_end2, arrow_y_end2, 1},
                    PUFF_WHITE
                );
            }

            if(mode==1){
                float arrow_x_end1_world = px + (arrow_x_end1 * heading_self_x - arrow_y_end1*heading_self_y);
                float arrow_y_end1_world = py + (arrow_x_end1 * heading_self_y + arrow_y_end1*heading_self_x);
                float arrow_x_end2_world = px + (arrow_x_end2 * heading_self_x - arrow_y_end2*heading_self_y);
                float arrow_y_end2_world = py + (arrow_x_end2 * heading_self_y + arrow_y_end2*heading_self_x);
                DrawLine3D(
                    (Vector3){arrow_x_world, arrow_y_world, 1},
                    (Vector3){arrow_x_end1_world, arrow_y_end1_world, 1},
                    PUFF_WHITE
                );
                DrawLine3D(
                    (Vector3){arrow_x_world, arrow_y_world, 1},
                    (Vector3){arrow_x_end2_world, arrow_y_end2_world, 1},
                    PUFF_WHITE
                );

            }
        }

        obs_idx += 7;  // Move to next agent observation (7 values per agent)
    }
    // Then draw map observations
    int map_start_idx = ego_dim + 7*(MAX_VISIBLE_AGENTS - 1);  // Start after agent observations
    for(int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {  // Loop through potential map entities
        int entity_idx = map_start_idx + k*7;
        if(agent_obs[entity_idx] == 0 && agent_obs[entity_idx + 1] == 0){
            continue;
        }
        Color lineColor = BLUE;  // Default color
        int entity_type = (int)agent_obs[entity_idx + 6];
        // Choose color based on entity type
        if(entity_type+4 != ROAD_EDGE){
            continue;
        }
        lineColor = PUFF_CYAN;
        // For road segments, draw line between start and end points
        float x_middle = agent_obs[entity_idx] * 50;
        float y_middle = agent_obs[entity_idx + 1] * 50;
        float rel_angle_x = (agent_obs[entity_idx + 4]);
        float rel_angle_y = (agent_obs[entity_idx + 5]);
        float rel_angle = atan2f(rel_angle_y, rel_angle_x);
        float segment_length = agent_obs[entity_idx + 2] * MAX_ROAD_SEGMENT_LENGTH;
        // Calculate endpoint using the relative angle directly
        // Calculate endpoint directly
        float x_start = x_middle - segment_length*cosf(rel_angle);
        float y_start = y_middle - segment_length*sinf(rel_angle);
        float x_end = x_middle + segment_length*cosf(rel_angle);
        float y_end = y_middle + segment_length*sinf(rel_angle);


        if(lasers && mode ==0){
            DrawLine3D((Vector3){0,0,0}, (Vector3){x_middle, y_middle, 1}, lineColor);
        }

        if(mode ==1){
            float x_middle_world = px + (x_middle*heading_self_x - y_middle*heading_self_y);
            float y_middle_world = py + (x_middle*heading_self_y + y_middle*heading_self_x);
            float x_start_world = px + (x_start*heading_self_x - y_start*heading_self_y);
            float y_start_world = py + (x_start*heading_self_y + y_start*heading_self_x);
            float x_end_world = px + (x_end*heading_self_x - y_end*heading_self_y);
            float y_end_world = py + (x_end*heading_self_y + y_end*heading_self_x);
            DrawCube((Vector3){x_middle_world, y_middle_world, 1}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3){x_start_world, y_start_world, 1}, (Vector3){x_end_world, y_end_world, 1}, BLUE);
            if(lasers) DrawLine3D((Vector3){px,py,1}, (Vector3){x_middle_world, y_middle_world, 1}, lineColor);
        }
        if(mode ==0){
            DrawCube((Vector3){x_middle, y_middle, 1}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3){x_start, y_start, 1}, (Vector3){x_end, y_end, 1}, BLUE);
        }
    }
}

void draw_road_edge(Drive* env, float start_x, float start_y, float end_x, float end_y){
    Color CURB_TOP = (Color){220, 220, 220, 255};      // Top surface - lightest
    Color CURB_SIDE = (Color){180, 180, 180, 255};     // Side faces - medium
    Color CURB_BOTTOM = (Color){160, 160, 160, 255};
                    // Calculate curb dimensions
    float curb_height = 0.5f;  // Height of the curb
    float curb_width = 0.3f;   // Width/thickness of the curb
    float road_z = 0.2f;       // Ensure z-level for roads is below agents

    // Calculate direction vector between start and end
    Vector3 direction = {
        end_x - start_x,
        end_y - start_y,
        0.0f
    };

    // Calculate length of the segment
    float length = sqrtf(direction.x * direction.x + direction.y * direction.y);

    // Normalize direction vector
    Vector3 normalized_dir = {
        direction.x / length,
        direction.y / length,
        0.0f
    };

    // Calculate perpendicular vector for width
    Vector3 perpendicular = {
        -normalized_dir.y,
        normalized_dir.x,
        0.0f
    };

    // Calculate the four bottom corners of the curb
    Vector3 b1 = {
        start_x - perpendicular.x * curb_width/2,
        start_y - perpendicular.y * curb_width/2,
        road_z
    };
    Vector3 b2 = {
        start_x + perpendicular.x * curb_width/2,
        start_y + perpendicular.y * curb_width/2,
        road_z
    };
    Vector3 b3 = {
        end_x + perpendicular.x * curb_width/2,
        end_y + perpendicular.y * curb_width/2,
        road_z
    };
    Vector3 b4 = {
        end_x - perpendicular.x * curb_width/2,
        end_y - perpendicular.y * curb_width/2,
        road_z
    };

    // Draw the curb faces
    // Bottom face
    DrawTriangle3D(b1, b2, b3, CURB_BOTTOM);
    DrawTriangle3D(b1, b3, b4, CURB_BOTTOM);

    // Top face (raised by curb_height)
    Vector3 t1 = {b1.x, b1.y, b1.z + curb_height};
    Vector3 t2 = {b2.x, b2.y, b2.z + curb_height};
    Vector3 t3 = {b3.x, b3.y, b3.z + curb_height};
    Vector3 t4 = {b4.x, b4.y, b4.z + curb_height};
    DrawTriangle3D(t1, t3, t2, CURB_TOP);
    DrawTriangle3D(t1, t4, t3, CURB_TOP);

    // Side faces
    DrawTriangle3D(b1, t1, b2, CURB_SIDE);
    DrawTriangle3D(t1, t2, b2, CURB_SIDE);
    DrawTriangle3D(b2, t2, b3, CURB_SIDE);
    DrawTriangle3D(t2, t3, b3, CURB_SIDE);
    DrawTriangle3D(b3, t3, b4, CURB_SIDE);
    DrawTriangle3D(t3, t4, b4, CURB_SIDE);
    DrawTriangle3D(b4, t4, b1, CURB_SIDE);
    DrawTriangle3D(t4, t1, b1, CURB_SIDE);
}

void draw_scene(Drive* env, Client* client, int mode, int obs_only, int lasers, int show_grid, Camera3D camera, AgentLabel* labels, int max_labels, int* label_count){
   // Draw a grid to help with orientation
    // DrawGrid(20, 1.0f);
    DrawLine3D((Vector3){env->grid_map->top_left_x, env->grid_map->top_left_y, 0}, (Vector3){env->grid_map->bottom_right_x, env->grid_map->top_left_y, 0}, PUFF_CYAN);
    DrawLine3D((Vector3){env->grid_map->top_left_x, env->grid_map->bottom_right_y, 0}, (Vector3){env->grid_map->top_left_x, env->grid_map->top_left_y, 0}, PUFF_CYAN);
    DrawLine3D((Vector3){env->grid_map->bottom_right_x, env->grid_map->bottom_right_y, 0}, (Vector3){env->grid_map->bottom_right_x, env->grid_map->top_left_y, 0}, PUFF_CYAN);
    DrawLine3D((Vector3){env->grid_map->top_left_x, env->grid_map->bottom_right_y, 0}, (Vector3){env->grid_map->bottom_right_x, env->grid_map->bottom_right_y, 0}, PUFF_CYAN);
    for(int i = 0; i < env->num_objects; i++) {
        // Draw objects
        if(env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN || env->entities[i].type == CYCLIST) {
            // Check if this vehicle is an active agent
            bool is_active_agent = false;
            bool is_static_car = false;
            int agent_index = -1;
            for(int j = 0; j < env->active_agent_count; j++) {
                if(env->active_agent_indices[j] == i) {
                    is_active_agent = true;
                    agent_index = j;
                    break;
                }
            }
            for(int j = 0; j < env->static_car_count; j++) {
                if(env->static_car_indices[j] == i) {
                    is_static_car = true;
                    break;
                }
            }

            Vector3 size = {
                env->entities[i].length,
                env->entities[i].width,
                env->entities[i].height
            };

            Vector3 position;
            float heading;
            position = (Vector3){
                env->entities[i].x,
                env->entities[i].y,
                1
            };
            heading = env->entities[i].heading;


            const char* show_ghost_car = getenv("SHOW_GHOST_CAR");
            if (is_active_agent && (!show_ghost_car || strcmp(show_ghost_car, "1") == 0)) {
                int traj_idx = env->timestep;
                if (traj_idx < 0) traj_idx = 0;
                if (traj_idx >= env->entities[i].array_size) {
                    traj_idx = env->entities[i].array_size - 1;
                }

                float ghost_heading = env->entities[i].traj_heading[traj_idx];
                float gx = env->entities[i].traj_x[traj_idx];
                float gy = env->entities[i].traj_y[traj_idx];
                Vector3 ghost_position = (Vector3){gx, gy, position.z};

                float ghost_cos = cosf(ghost_heading);
                float ghost_sin = sinf(ghost_heading);
                float ghost_half_len = env->entities[i].length * 0.5f;
                float ghost_half_width = env->entities[i].width * 0.5f;
                Vector3 ghost_corners[4] = {
                    (Vector3){
                        ghost_position.x + (ghost_half_len * ghost_cos - ghost_half_width * ghost_sin),
                        ghost_position.y + (ghost_half_len * ghost_sin + ghost_half_width * ghost_cos),
                        ghost_position.z
                    },
                    (Vector3){
                        ghost_position.x + (ghost_half_len * ghost_cos + ghost_half_width * ghost_sin),
                        ghost_position.y + (ghost_half_len * ghost_sin - ghost_half_width * ghost_cos),
                        ghost_position.z
                    },
                    (Vector3){
                        ghost_position.x + (-ghost_half_len * ghost_cos + ghost_half_width * ghost_sin),
                        ghost_position.y + (-ghost_half_len * ghost_sin - ghost_half_width * ghost_cos),
                        ghost_position.z
                    },
                    (Vector3){
                        ghost_position.x + (-ghost_half_len * ghost_cos - ghost_half_width * ghost_sin),
                        ghost_position.y + (-ghost_half_len * ghost_sin + ghost_half_width * ghost_cos),
                        ghost_position.z
                    },
                };
                Color ghost_color = Fade(PUFF_WHITE, 0.45f);
                for (int j = 0; j < 4; j++) {
                    DrawLine3D(ghost_corners[j], ghost_corners[(j+1)%4], ghost_color);
                }
                Vector3 ghost_arrow_end = {
                    ghost_position.x + ghost_cos * ghost_half_len * 1.5f,
                    ghost_position.y + ghost_sin * ghost_half_len * 1.5f,
                    ghost_position.z
                };
                DrawLine3D(ghost_position, ghost_arrow_end, ghost_color);
                DrawSphere(ghost_arrow_end, 0.15f, ghost_color);

                BoundingBox bounds = GetModelBoundingBox(client->cars[5]);
                Vector3 model_size = {
                    bounds.max.x - bounds.min.x,
                    bounds.max.y - bounds.min.y,
                    bounds.max.z - bounds.min.z
                };
                Vector3 scale = {
                    size.x / model_size.x,
                    size.y / model_size.y,
                    size.z / model_size.z
                };

                rlPushMatrix();
                rlTranslatef(ghost_position.x, ghost_position.y, ghost_position.z);
                rlRotatef(ghost_heading*RAD2DEG, 0.0f, 0.0f, 1.0f);
                Color ghost_tint = Fade(PUFF_WHITE, 0.35f);
                DrawModelEx(client->cars[5], (Vector3){0, 0, 0}, (Vector3){1, 0, 0}, 90.0f, scale, ghost_tint);
                rlPopMatrix();
            }

            // HIDE CARS ON RESPAWN - IMPORTANT TO KNOW VISUAL SETTING
            if((!is_active_agent && !is_static_car)){
                continue;
            }

            bool is_expert = (!is_active_agent) && (env->entities[i].mark_as_expert == 1);

            // Save current transform
            if(mode==1){
                float cos_heading = env->entities[i].heading_x;
                float sin_heading = env->entities[i].heading_y;

                // Calculate half dimensions
                float half_len = env->entities[i].length * 0.5f;
                float half_width = env->entities[i].width * 0.5f;

                // Calculate the four corners of the collision box
                Vector3 corners[4] = {
                    (Vector3){
                        position.x + (half_len * cos_heading - half_width * sin_heading),
                        position.y + (half_len * sin_heading + half_width * cos_heading),
                        position.z
                    },


                    (Vector3){
                        position.x + (half_len * cos_heading + half_width * sin_heading),
                        position.y + (half_len * sin_heading - half_width * cos_heading),
                        position.z
                    },
                   (Vector3){
                        position.x + (-half_len * cos_heading + half_width * sin_heading),
                        position.y + (-half_len * sin_heading - half_width * cos_heading),
                        position.z
                    },
                   (Vector3){
                        position.x + (-half_len * cos_heading - half_width * sin_heading),
                        position.y + (-half_len * sin_heading + half_width * cos_heading),
                        position.z
                    },


                };

                if(agent_index == env->human_agent_idx) {
                    draw_agent_obs(env, agent_index, mode, obs_only, lasers);
                }
                if((obs_only ||  IsKeyDown(KEY_LEFT_CONTROL)) && agent_index != env->human_agent_idx){
                    continue;
                }

                // --- Draw the car  ---
                Color car_color = GRAY;              // default for static
                if (is_expert) car_color = GOLD;      // expert replay
                if (is_active_agent) car_color = BLUE; // policy-controlled
                if (is_active_agent && env->entities[i].collision_state > 0) car_color = RED;
                rlSetLineWidth(3.0f);
                for (int j = 0; j < 4; j++) {
                    DrawLine3D(corners[j], corners[(j+1)%4], car_color);
                }
                // --- Draw a heading arrow pointing forward ---
                Vector3 arrowStart = position;
                Vector3 arrowEnd = {
                    position.x + cos_heading * half_len * 1.5f, // extend arrow beyond car
                    position.y + sin_heading * half_len * 1.5f,
                    position.z
                };

                DrawLine3D(arrowStart, arrowEnd, car_color);
                DrawSphere(arrowEnd, 0.2f, car_color);  // arrow tip
            }
            else {
                rlPushMatrix();
                // Translate to position, rotate around Y axis, then draw
                rlTranslatef(position.x, position.y, position.z);
                rlRotatef(heading*RAD2DEG, 0.0f, 0.0f, 1.0f);  // Convert radians to degrees
                // Determine color based on status
                Color object_color = PUFF_BACKGROUND2;  // fill color unused for model tint
                Color outline_color = PUFF_CYAN;        // not used for model tint
                Model car_model = client->cars[5];
                if(is_active_agent){
                    car_model = client->cars[client->car_assignments[i %64]];
                }
                if(agent_index == env->human_agent_idx){
                    object_color = PUFF_CYAN;
                    outline_color = PUFF_WHITE;
                }
                if(is_active_agent && env->entities[i].collision_state > 0) {
                    car_model = client->cars[0];  // Collided agent
                }
                // Draw obs for human selected agent
                if(agent_index == env->human_agent_idx && !env->entities[agent_index].metrics_array[REACHED_GOAL_IDX]) {
                    draw_agent_obs(env, agent_index, mode, obs_only, lasers);
                }
                // Draw cube for cars static and active
                // Calculate scale factors based on desired size and model dimensions

                BoundingBox bounds = GetModelBoundingBox(car_model);
                Vector3 model_size = {
                    bounds.max.x - bounds.min.x,
                    bounds.max.y - bounds.min.y,
                    bounds.max.z - bounds.min.z
                };
                Vector3 scale = {
                    size.x / model_size.x,
                    size.y / model_size.y,
                    size.z / model_size.z
                };
                if((obs_only ||  IsKeyDown(KEY_LEFT_CONTROL)) && agent_index != env->human_agent_idx){
                    rlPopMatrix();
                    continue;
                }

                if (is_expert && size.x * size.y <= 0.300001f && size.z < 1.0f) {
                    // Small stationary objects use their real box, without a vehicle model.
                    Vector3 center = {0, 0, size.z / 2.0f - position.z};
                    DrawCube(center, size.x, size.y, size.z, GRAY);
                    DrawCubeWires(center, size.x, size.y, size.z, DARKGRAY);
                    rlPopMatrix();
                    continue;
                }
                DrawModelEx(car_model, (Vector3){0, 0, 0}, (Vector3){1, 0, 0}, 90.0f, scale, WHITE);
                {
                    float cos_heading = env->entities[i].heading_x;
                    float sin_heading = env->entities[i].heading_y;
                    float half_len = env->entities[i].length * 0.5f;
                    float half_width = env->entities[i].width * 0.5f;
                    Vector3 corners[4] = {
                        (Vector3){ 0 + ( half_len * cos_heading - half_width * sin_heading), 0 + ( half_len * sin_heading + half_width * cos_heading), 0 },
                        (Vector3){ 0 + ( half_len * cos_heading + half_width * sin_heading), 0 + ( half_len * sin_heading - half_width * cos_heading), 0 },
                        (Vector3){ 0 + (-half_len * cos_heading + half_width * sin_heading), 0 + (-half_len * sin_heading - half_width * cos_heading), 0 },
                        (Vector3){ 0 + (-half_len * cos_heading - half_width * sin_heading), 0 + (-half_len * sin_heading + half_width * cos_heading), 0 },
                    };
                    Color wire_color = GRAY;                 // static
                    if (!is_active_agent && env->entities[i].mark_as_expert == 1) wire_color = GOLD;  // expert replay
                    if (is_active_agent) wire_color = BLUE;   // policy
                    if (is_active_agent && env->entities[i].collision_state > 0) wire_color = RED;
                    rlSetLineWidth(2.0f);
                    for (int j = 0; j < 4; j++) {
                        DrawLine3D(corners[j], corners[(j+1)%4], wire_color);
                    }
                }
                rlPopMatrix();
            }

            // FPV Camera Control
            if(IsKeyDown(KEY_SPACE) && env->human_agent_idx== agent_index){
                if(env->entities[agent_index].metrics_array[REACHED_GOAL_IDX]){
                    env->human_agent_idx = rand() % env->active_agent_count;
                }
                Vector3 camera_position = (Vector3){
                        position.x - (25.0f * cosf(heading)),
                        position.y - (25.0f * sinf(heading)),
                        position.z + 15
                };

                Vector3 camera_target = (Vector3){
                    position.x + 40.0f * cosf(heading),
                    position.y + 40.0f * sinf(heading),
                    position.z - 5.0f
                };
                client->camera.position = camera_position;
                client->camera.target = camera_target;
                client->camera.up = (Vector3){0, 0, 1};
            }
            if(IsKeyReleased(KEY_SPACE)){
                client->camera.position = client->default_camera_position;
                client->camera.target = client->default_camera_target;
                client->camera.up = (Vector3){0, 0, 1};
            }
            // Draw goal position for active agents

            if(!is_active_agent || env->entities[i].valid == 0) {
                continue;
            }
            if(!IsKeyDown(KEY_LEFT_CONTROL) && obs_only==0 && !env->entities[i].reached_final_goal){
                DrawSphere((Vector3){
                    env->entities[i].goal_position_x,
                    env->entities[i].goal_position_y,
                    1
                }, 0.5f, DARKGREEN);

                DrawCircle3D((Vector3){
                    env->entities[i].goal_position_x,
                    env->entities[i].goal_position_y,
                    0.1f
                }, env->goal_radius, (Vector3){0, 0, 1}, 90.0f, Fade(LIGHTGREEN, 0.3f));
            }

            int show_speed = getenv("SHOW_SPEED") == NULL || strcmp(getenv("SHOW_SPEED"), "1") == 0;

            if (show_speed && !env->entities[i].reached_final_goal && labels && label_count && (*label_count) < max_labels) {
                Vector3 label_world = {
                    position.x,
                    position.y,
                    position.z + size.z + 0.5f
                };
                Vector2 screen_pos = GetWorldToScreen(label_world, camera);
                float viewport_width = (client && client->width > 0) ? client->width : (float)GetScreenWidth();
                float viewport_height = (client && client->height > 0) ? client->height : (float)GetScreenHeight();
                if (screen_pos.x >= 0 && screen_pos.x <= viewport_width &&
                    screen_pos.y >= 0 && screen_pos.y <= viewport_height) {
                    float speed_mps = sqrtf(env->entities[i].vx * env->entities[i].vx +
                                            env->entities[i].vy * env->entities[i].vy);
                    labels[*label_count].screen_pos = screen_pos;
                    labels[*label_count].speed_kph = speed_mps * 3.6f;
                    labels[*label_count].visible = true;
                    (*label_count)++;
                }
            }
        }
    }
    // Draw immutable geometry with this environment's traffic-light state.
    for (int i = 0; i < env->num_roads; i++) {
        int road_type = effective_road_type(env, &env->static_road_entities[i]);
        if (road_type == NONE) continue;
        for(int j = 0; j < env->static_road_entities[i].array_size - 1; j++) {
            Vector3 start = {
                env->static_road_entities[i].traj_x[j],
                env->static_road_entities[i].traj_y[j],
                1
            };
            Vector3 end = {
                env->static_road_entities[i].traj_x[j + 1],
                env->static_road_entities[i].traj_y[j + 1],
                1
            };
            if(road_type == ROAD_EDGE){
                draw_road_edge(env, start.x, start.y, end.x, end.y);
            } else if (road_type == STOP_SIGN) {
                DrawLine3D(start, end, RED);
            } else if (road_type == RIGHT_HAND_STOP_LINE) {
                DrawLine3D(start, end, (Color){255, 182, 193, 255});
            } else if (road_type == CROSSWALK) {
                DrawLine3D(start, end, YELLOW);
            } else if (road_type == TRAFFIC_LIGHT_STOP_LINE) {
                DrawLine3D(start, end, traffic_light_render_color(road_traffic_light_color(env, &env->static_road_entities[i])));
            } else if (road_type == ROAD_LANE) {
                DrawLine3D(start, end, GRAY);
            }
        }
    }
    if(show_grid) {
    // Draw grid cells using the stored bounds
    float grid_start_x = env->grid_map->top_left_x;
    float grid_start_y = env->grid_map->bottom_right_y;
    for(int i = 0; i < env->grid_map->grid_cols; i++) {
        for(int j = 0; j < env->grid_map->grid_rows; j++) {
            float x = grid_start_x + i*GRID_CELL_SIZE;
            float y = grid_start_y + j*GRID_CELL_SIZE;
            DrawCubeWires(
                (Vector3){x + GRID_CELL_SIZE/2, y + GRID_CELL_SIZE/2, 1},
                GRID_CELL_SIZE, GRID_CELL_SIZE, 0.1f, PUFF_BACKGROUND2);
        }
        }
    }
}

void saveTopDownImage(Drive* env, Client* client, const char *filename, RenderTexture2D target, int map_height, int obs, int lasers, int trajectories, int frame_count, float* path, int log_trajectories, int show_grid){
    // Top-down orthographic camera
    Camera3D camera = {0};
    camera.position = (Vector3){ 0.0f, 0.0f, 500.0f };  // above the scene
    camera.target   = (Vector3){ 0.0f, 0.0f, 0.0f };  // look at origin
    camera.up       = (Vector3){ 0.0f, -1.0f, 0.0f };
    camera.fovy     = map_height;
    camera.projection = CAMERA_ORTHOGRAPHIC;
    Color road = (Color){35, 35, 37, 255};

    BeginTextureMode(target);
        ClearBackground(road);
        BeginMode3D(camera);
            rlEnableDepthTest();

            // Draw log trajectories FIRST (in background at lower Z-level)
            if(log_trajectories){
                for(int i=0; i<env->active_agent_count;i++){
                    int idx = env->active_agent_indices[i];
                    for(int j=0; j<env->entities[idx].array_size;j++){
                        float x = env->entities[idx].traj_x[j];
                        float y = env->entities[idx].traj_y[j];
                        float valid = env->entities[idx].traj_valid[j];
                        if(!valid) continue;
                        DrawSphere((Vector3){x,y,0.5f}, 0.3f, Fade(LIGHTGREEN, 0.6f));
                    }
                }
            }

            // Draw current path trajectories SECOND (slightly higher than log trajectories)
            if(trajectories){
                for(int i=0; i<frame_count; i++){
                    DrawSphere((Vector3){path[i*2], path[i*2 +1], 0.8f}, 0.5f, YELLOW);
                }
            }

            // Draw main scene LAST (on top)
            draw_scene(env, client, 1, obs, lasers, show_grid, camera, NULL, 0, NULL);

        EndMode3D();
    EndTextureMode();

    // save to file
    Image img = LoadImageFromTexture(target.texture);
    ImageFlipVertical(&img);
    ExportImage(img, filename);
    UnloadImage(img);
}


void c_render(Drive* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    Client* client = env->client;
    AgentLabel labels[MAX_AGENTS] = {0};
    int label_count = 0;
    BeginDrawing();
    Color road = (Color){35, 35, 37, 255};
    ClearBackground(road);
    BeginMode3D(client->camera);
    handle_camera_controls(env->client);
    draw_scene(env, client, 0, 0, 0, 0, client->camera, labels, MAX_AGENTS, &label_count);
    EndMode3D();
    for (int i = 0; i < label_count; i++) {
        if (!labels[i].visible) continue;
        DrawText(
            TextFormat("%.0f km/h", labels[i].speed_kph),
            (int)labels[i].screen_pos.x,
            (int)labels[i].screen_pos.y,
            20,
            RED
        );
    }
    // Draw debug info
    DrawText(TextFormat("Camera Position: (%.2f, %.2f, %.2f)",
        client->camera.position.x,
        client->camera.position.y,
        client->camera.position.z), 10, 10, 20, PUFF_WHITE);
    DrawText(TextFormat("Camera Target: (%.2f, %.2f, %.2f)",
        client->camera.target.x,
        client->camera.target.y,
        client->camera.target.z), 10, 30, 20, PUFF_WHITE);
    DrawText(TextFormat("Timestep: %d", env->timestep), 10, 50, 20, PUFF_WHITE);
    // acceleration & steering
    int human_idx = env->active_agent_indices[env->human_agent_idx];
    DrawText(TextFormat("Controlling Agent: %d", env->human_agent_idx), 10, 70, 20, PUFF_WHITE);
    DrawText(TextFormat("Agent Index: %d", human_idx), 10, 90, 20, PUFF_WHITE);
    // Controls help
    DrawText("Controls: W/S - Accelerate/Brake, A/D - Steer, 1-4 - Switch Agent",
             10, client->height - 30, 20, PUFF_WHITE);
    // acceleration & steering
    if (env->action_type == 1) { // continuous (float)
        float (*action_array_f)[2] = (float(*)[2])env->actions;
        DrawText(TextFormat("Acceleration: %.2f", action_array_f[env->human_agent_idx][0]), 10, 110, 20, PUFF_WHITE);
        DrawText(TextFormat("Steering: %.2f", action_array_f[env->human_agent_idx][1]), 10, 130, 20, PUFF_WHITE);
    } else { // discrete (int)
        int (*action_array)[2] = (int(*)[2])env->actions;
        DrawText(TextFormat("Acceleration: %d", action_array[env->human_agent_idx][0]), 10, 110, 20, PUFF_WHITE);
        DrawText(TextFormat("Steering: %d", action_array[env->human_agent_idx][1]), 10, 130, 20, PUFF_WHITE);
    }
    DrawText(TextFormat("Grid Rows: %d", env->grid_map->grid_rows), 10, 150, 20, PUFF_WHITE);
    DrawText(TextFormat("Grid Cols: %d", env->grid_map->grid_cols), 10, 170, 20, PUFF_WHITE);
    EndDrawing();
}

void close_client(Client* client){
    for (int i = 0; i < 6; i++) {
        UnloadModel(client->cars[i]);
    }
    UnloadTexture(client->puffers);
    CloseWindow();
    free(client);
}
