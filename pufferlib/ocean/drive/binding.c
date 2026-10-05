#include "drive.h"
#include <Python.h>

static PyObject* vec_active_agent_positions(PyObject* self, PyObject* args);
static PyObject* make_static_road_map(PyObject* self, PyObject* args);

#define Env Drive
#define MY_SHARED
#define MY_PUT
#define MY_METHODS \
    {"vec_active_agent_positions", vec_active_agent_positions, METH_VARARGS, \
     "Return world XY positions, environment indices, and entity indices for active agents"}, \
    {"create_static_road_map", make_static_road_map, METH_VARARGS, \
     "Take ownership of a StaticRoadEntity array and build a reusable road map"}
#include "../env_binding.h"

#define STATIC_ROAD_MAP_CAPSULE "puffer_drive.StaticRoadMap"

static void destroy_static_road_map_capsule(PyObject* capsule) {
    StaticRoadMap* map = PyCapsule_GetPointer(capsule, STATIC_ROAD_MAP_CAPSULE);
    if (map) release_static_road_map(map);
}

static PyObject* make_static_road_map(PyObject* self, PyObject* args) {
    PyObject* address;
    int count;
    if (!PyArg_ParseTuple(args, "Oi", &address, &count)) return NULL;
    StaticRoadEntity* roads = PyLong_AsVoidPtr(address);
    if (PyErr_Occurred()) return NULL;
    if (count < 0 || (count > 0 && roads == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Expected a nonnegative road count and a non-null road array");
        return NULL;
    }
    for (int i = 0; i < count; i++) {
        if (!grid_map_entity_type(roads[i].type) || roads[i].array_size < 0 ||
            (roads[i].array_size > 0 && (!roads[i].traj_x || !roads[i].traj_y || !roads[i].traj_z)) ||
            (roads[i].type == TRAFFIC_LIGHT_STOP_LINE &&
             (roads[i].traffic_light_idx < 0 ||
              (roads[i].traffic_light_off_type != NONE && roads[i].traffic_light_off_type != STOP_SIGN)))) {
            PyErr_SetString(PyExc_ValueError, "Invalid static road geometry or traffic-light index");
            return NULL;
        }
    }
    StaticRoadMap* map = create_static_road_map(roads, count);
    PyObject* capsule = PyCapsule_New(map, STATIC_ROAD_MAP_CAPSULE, destroy_static_road_map_capsule);
    if (!capsule) release_static_road_map(map);
    return capsule;
}

static PyObject* vec_active_agent_positions(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "vec_active_agent_positions requires 1 argument");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    npy_intp agent_count = 0;
    for (int env_idx = 0; env_idx < vec->num_envs; env_idx++) {
        Drive* env = vec->envs[env_idx];
        if (env->active_agent_count < 0 || env->active_agent_count > MAX_AGENTS) {
            PyErr_SetString(PyExc_RuntimeError, "Invalid active agent count");
            return NULL;
        }
        agent_count += env->active_agent_count;
    }

    npy_intp position_dims[2] = {agent_count, 2};
    npy_intp index_dims[1] = {agent_count};
    PyObject* positions = PyArray_SimpleNew(2, position_dims, NPY_FLOAT32);
    PyObject* env_indices = PyArray_SimpleNew(1, index_dims, NPY_INT32);
    PyObject* entity_indices = PyArray_SimpleNew(1, index_dims, NPY_INT32);
    if (!positions || !env_indices || !entity_indices) {
        Py_XDECREF(positions);
        Py_XDECREF(env_indices);
        Py_XDECREF(entity_indices);
        return NULL;
    }

    float* position_data = (float*)PyArray_DATA((PyArrayObject*)positions);
    int32_t* env_index_data = (int32_t*)PyArray_DATA((PyArrayObject*)env_indices);
    int32_t* entity_index_data = (int32_t*)PyArray_DATA((PyArrayObject*)entity_indices);
    npy_intp row = 0;
    for (int env_idx = 0; env_idx < vec->num_envs; env_idx++) {
        Drive* env = vec->envs[env_idx];
        for (int active_idx = 0; active_idx < env->active_agent_count; active_idx++) {
            int entity_idx = env->active_agent_indices[active_idx];
            if (entity_idx < 0 || entity_idx >= env->num_objects) {
                Py_DECREF(positions);
                Py_DECREF(env_indices);
                Py_DECREF(entity_indices);
                PyErr_SetString(PyExc_RuntimeError, "Invalid active entity index");
                return NULL;
            }
            Entity* entity = &env->entities[entity_idx];
            position_data[2 * row] = entity->x;
            position_data[2 * row + 1] = entity->y;
            env_index_data[row] = env_idx;
            entity_index_data[row] = entity_idx;
            row++;
        }
    }

    PyObject* result = PyTuple_New(3);
    if (!result) {
        Py_DECREF(positions);
        Py_DECREF(env_indices);
        Py_DECREF(entity_indices);
        return NULL;
    }
    PyTuple_SET_ITEM(result, 0, positions);
    PyTuple_SET_ITEM(result, 1, env_indices);
    PyTuple_SET_ITEM(result, 2, entity_indices);
    return result;
}

static int my_put(Env* env, PyObject* args, PyObject* kwargs) {
    PyObject* obs = PyDict_GetItemString(kwargs, "observations");
    if (!PyObject_TypeCheck(obs, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return 1;
    }
    PyArrayObject* observations = (PyArrayObject*)obs;
    if (!PyArray_ISCONTIGUOUS(observations)) {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return 1;
    }
    env->observations = PyArray_DATA(observations);

    PyObject* act = PyDict_GetItemString(kwargs, "actions");
    if (!PyObject_TypeCheck(act, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return 1;
    }
    PyArrayObject* actions = (PyArrayObject*)act;
    if (!PyArray_ISCONTIGUOUS(actions)) {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return 1;
    }
    env->actions = PyArray_DATA(actions);
    if (PyArray_ITEMSIZE(actions) == sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Action tensor passed as float64 (pass np.float32 buffer)");
        return 1;
    }

    PyObject* rew = PyDict_GetItemString(kwargs, "rewards");
    if (!PyObject_TypeCheck(rew, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return 1;
    }
    PyArrayObject* rewards = (PyArrayObject*)rew;
    if (!PyArray_ISCONTIGUOUS(rewards)) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return 1;
    }
    env->rewards = PyArray_DATA(rewards);

    PyObject* term = PyDict_GetItemString(kwargs, "terminals");
    if (!PyObject_TypeCheck(term, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return 1;
    }
    PyArrayObject* terminals = (PyArrayObject*)term;
    if (!PyArray_ISCONTIGUOUS(terminals)) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(terminals) != 1) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be 1D");
        return 1;
    }
    env->terminals = PyArray_DATA(terminals);

    PyObject* trunc = PyDict_GetItemString(kwargs, "truncations");
    if (!PyObject_TypeCheck(trunc, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Truncations must be a NumPy array");
        return 1;
    }
    PyArrayObject* truncations = (PyArrayObject*)trunc;
    if (!PyArray_ISCONTIGUOUS(truncations)) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(truncations) != 1) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be 1D");
        return 1;
    }
    env->truncations = PyArray_DATA(truncations);
    return 0;
}

static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs) {
    int num_agents = unpack(kwargs, "num_agents");
    int num_maps = unpack(kwargs, "num_maps");

    PyObject* map_dir = kwargs ? PyDict_GetItemString(kwargs, "map_dir") : NULL;
    if (!map_dir) {
        PyErr_SetString(PyExc_TypeError, "map_dir is not set");
        return NULL;
    }

    if (!PyUnicode_Check(map_dir)) {
        PyErr_SetString(PyExc_TypeError, "map_dir is not string");
        return NULL;
    }

    const char* map_dir_str = PyUnicode_AsUTF8(map_dir);

    PyObject* all_maps_exactly_once_obj = PyDict_GetItemString(kwargs, "all_maps_exactly_once");
    int all_maps_exactly_once = all_maps_exactly_once_obj && Py_IsTrue(all_maps_exactly_once_obj);

    clock_gettime(CLOCK_REALTIME, &ts);
    srand(ts.tv_nsec);
    int total_agent_count = 0;
    int env_count = 0;
    int max_envs = all_maps_exactly_once ? num_maps : num_agents;
    PyObject* agent_offsets = PyList_New(max_envs+1);
    PyObject* map_ids = PyList_New(max_envs);
    // getting env count
    while(total_agent_count < num_agents && env_count < max_envs){
        char map_file[256];
        int map_id = all_maps_exactly_once ? env_count : rand() % num_maps;
        Drive* env = calloc(1, sizeof(Drive));
        sprintf(map_file, "%s/map_%03d.bin", map_dir_str, map_id);
        env->entities = load_map_binary(map_file, env);
        PyObject* obj = NULL;
        obj = kwargs ? PyDict_GetItemString(kwargs, "num_policy_controlled_agents") : NULL;
        if (obj && PyLong_Check(obj)) {
            env->policy_agents_per_env = (int)PyLong_AsLong(obj);
        } else {
            env->policy_agents_per_env = -1;
        }
        obj = kwargs ? PyDict_GetItemString(kwargs, "control_all_agents") : NULL;
        if (obj && PyLong_Check(obj)) {
            env->control_all_agents = (int)PyLong_AsLong(obj);
        } else {
            env->control_all_agents = 0;
        }
        obj = kwargs ? PyDict_GetItemString(kwargs, "deterministic_agent_selection") : NULL;
        if (obj && PyLong_Check(obj)) {
            env->deterministic_agent_selection = (int)PyLong_AsLong(obj);
        } else {
            env->deterministic_agent_selection = 0;
        }
        set_active_agents(env);
        // Store map_id
        PyObject* map_id_obj = PyLong_FromLong(map_id);
        PyList_SetItem(map_ids, env_count, map_id_obj);
        // Store agent offset
        PyObject* offset = PyLong_FromLong(total_agent_count);
        PyList_SetItem(agent_offsets, env_count, offset);
        total_agent_count += env->active_agent_count;
        env_count++;
        c_close(env);
        free(env);
    }
    if(total_agent_count >= num_agents){
        total_agent_count = num_agents;
    }
    PyObject* final_total_agent_count = PyLong_FromLong(total_agent_count);
    PyList_SetItem(agent_offsets, env_count, final_total_agent_count);
    PyObject* final_env_count = PyLong_FromLong(env_count);
    // resize lists
    PyObject* resized_agent_offsets = PyList_GetSlice(agent_offsets, 0, env_count + 1);
    PyObject* resized_map_ids = PyList_GetSlice(map_ids, 0, env_count);
    //
    //Py_DECREF(agent_offsets);
    //Py_DECREF(map_ids);
    // create a tuple
    PyObject* tuple = PyTuple_New(3);
    PyTuple_SetItem(tuple, 0, resized_agent_offsets);
    PyTuple_SetItem(tuple, 1, resized_map_ids);
    PyTuple_SetItem(tuple, 2, final_env_count);
    return tuple;
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->human_agent_idx = unpack(kwargs, "human_agent_idx");
    env->ini_file = unpack_str(kwargs, "ini_file");
    env_init_config conf = {0};
    if(ini_parse(env->ini_file, handler, &conf) < 0) {
        printf("Error while loading %s", env->ini_file);
    }
    if (kwargs && PyDict_GetItemString(kwargs, "scenario_length") && !Py_IsNone(PyDict_GetItemString(kwargs, "scenario_length"))) {
        conf.scenario_length = (int)unpack(kwargs, "scenario_length");
    }
    if (conf.scenario_length <= 0) {
        PyErr_SetString(PyExc_ValueError, "scenario_length must be > 0 (set in INI or kwargs)");
        return -1;
    }
    env->action_type = conf.action_type;
    env->dynamics_model = conf.dynamics_model;
    env->reward_vehicle_collision = conf.reward_vehicle_collision;
    env->reward_offroad_collision = conf.reward_offroad_collision;
    env->reward_goal = conf.reward_goal;
    env->reward_goal_post_respawn = conf.reward_goal_post_respawn;
    env->reward_ade = conf.reward_ade;
    env->goal_radius = conf.goal_radius;
    env->scenario_length = conf.scenario_length;
    env->use_goal_generation = conf.use_goal_generation;
    env->policy_agents_per_env = unpack(kwargs, "num_policy_controlled_agents");
    env->control_all_agents = unpack(kwargs, "control_all_agents");
    env->deterministic_agent_selection = unpack(kwargs, "deterministic_agent_selection");
    env->dt = conf.dt;
    env->control_non_vehicles = (int)unpack(kwargs, "control_non_vehicles");
    
    if (kwargs && Py_IsTrue(PyDict_GetItemString(kwargs, "play_exactly_once"))) {
        env->play_exactly_once = 1;
    }

    int max_agents = unpack(kwargs, "max_agents");
    env->num_agents = max_agents;

    PyObject* map_dir = kwargs ? PyDict_GetItemString(kwargs, "map_dir") : NULL;
    if (!map_dir) {
        PyErr_SetString(PyExc_TypeError, "map_dir is not set");
        return 1;
    }

    if (!PyUnicode_Check(map_dir)) {
        PyErr_SetString(PyExc_TypeError, "map_dir is not string");
        return 1;
    }

    const char* map_dir_str = PyUnicode_AsUTF8(map_dir);

    PyObject* dynamic = PyDict_GetItemString(kwargs, "dynamic_map_generator");
    if (dynamic && !Py_IsNone(dynamic)) {
        env->map_name = "(dynamically-generated)";
        env->dynamic_map_generation = 1;

        PyObject* map = PyObject_CallNoArgs(dynamic);
        if (map == NULL) return 1;
        // 3 items: legacy combined Entity allocation (ownership transferred).
        // 4 items: separate actor and road allocations (ownership transferred).
        // 5 items: separate actors plus a shared road-map capsule. Only the
        // actors transfer ownership; retain a map reference for this environment.
        if (!PyTuple_Check(map) || PyTuple_Size(map) < 3 || PyTuple_Size(map) > 5) {
            PyErr_SetString(PyExc_TypeError, "Expected (num_objects, num_roads, entities_ptr[, roads_ptr[, static_road_map]])");
            Py_DECREF(map);
            return 1;
        }
        long num_objects = PyLong_AsLong(PyTuple_GetItem(map, 0));
        long num_roads = PyLong_AsLong(PyTuple_GetItem(map, 1));
        Entity* entities = PyLong_AsVoidPtr(PyTuple_GetItem(map, 2));
        if (PyErr_Occurred()) {
            Py_DECREF(map);
            return 1;
        }
        if (num_objects < 1 || num_objects > MAX_AGENTS || num_roads < 0 || num_roads > INT_MAX || !entities) {
            PyErr_SetString(PyExc_ValueError, "Invalid object/road count or null entity pointer");
            Py_DECREF(map);
            return 1;
        }
        StaticRoadEntity* roads = NULL;
        StaticRoadMap* shared_map = NULL;
        if (PyTuple_Size(map) >= 4) {
            roads = PyLong_AsVoidPtr(PyTuple_GetItem(map, 3));
            if (PyErr_Occurred() || (num_roads > 0 && !roads)) {
                if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "Static road entity pointer is null");
                Py_DECREF(map);
                return 1;
            }
        }
        if (PyTuple_Size(map) == 5) {
            shared_map = PyCapsule_GetPointer(PyTuple_GetItem(map, 4), STATIC_ROAD_MAP_CAPSULE);
            if (!shared_map) {
                Py_DECREF(map);
                return 1;
            }
            if (shared_map->entities != roads || shared_map->num_roads != num_roads) {
                PyErr_SetString(PyExc_ValueError, "Road array and count do not match the shared map");
                Py_DECREF(map);
                return 1;
            }
        }
        env->num_objects = (int)num_objects;
        env->num_roads = (int)num_roads;
        env->entities = entities;
        env->static_road_entities = roads;
        if (shared_map) acquire_static_road_map(env, shared_map);
        else if (PyTuple_Size(map) == 3) split_legacy_entities(env);
        Py_DECREF(map);
    } else {
        env->dynamic_map_generation = 0;
        int map_id = unpack(kwargs, "map_id");
        char map_file[256];
        sprintf(map_file, "%s/map_%03d.bin", map_dir_str, map_id);
        env->map_name = strdup(map_file);
    }

    PyObject* export_replay = PyDict_GetItemString(kwargs, "export_replay");
    if (export_replay && Py_IsTrue(export_replay)) {
        env->export_replay_enabled = 1;
        env->replay_dir = unpack_str(kwargs, "replay_dir");
    }

    int init_steps = unpack(kwargs, "init_steps");
    env->init_steps = init_steps;
    env->timestep = init_steps;
    init(env);
    return 0;
}

static int assign_int_array_to_dict(PyObject* dict, const char* key, int* values, npy_intp size) {
    PyObject* arr = PyArray_ZEROS(1, &size, NPY_INT, 0);
    if (arr == NULL) {
        return 1;
    }

    memcpy(PyArray_DATA((PyArrayObject*)arr), values, sizeof(int)*size);

    if (PyDict_SetItemString(dict, key, arr) == -1) {
        Py_DECREF(arr);
        PyErr_Format(PyExc_RuntimeError, "failed to set %s in dictionary", key);
        return 1;
    }

    Py_DECREF(arr);
    return 0;
}

static int my_log(
    PyObject* dict,
    Log* log,
    int* collision_counts,
    int* expert_collision_counts,
    int* offroad_counts,
    int* stop_line_counts,
    int* comfort_violation_counts,
    int* lane_center_offset_counts,
    int* acceleration_action_counts,
    int* steering_action_counts,
    int* speed_counts) {
    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "offroad_rate", log->offroad_rate);
    assign_to_dict(dict, "stop_line_rate", log->stop_line_rate);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "priority_violations", log->priority_violations);
    assign_to_dict(dict, "traffic_light_violations", log->traffic_light_violations);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "goal_reward", log->goal_reward);
    assign_to_dict(dict, "lane_center_reward", log->lane_center_reward);
    assign_to_dict(dict, "comfort_reward", log->comfort_reward);
    assign_to_dict(dict, "lane_alignment_reward", log->lane_alignment_reward);
    assign_to_dict(dict, "collision_reward", log->collision_reward);
    assign_to_dict(dict, "offroad_reward", log->offroad_reward);
    assign_to_dict(dict, "stop_line_reward", log->stop_line_reward);
    assign_to_dict(dict, "velocity_reward", log->velocity_reward);
    assign_to_dict(dict, "ade_reward", log->ade_reward);
    assign_to_dict(dict, "dnf_rate", log->dnf_rate);
    assign_to_dict(dict, "avg_displacement_error", log->avg_displacement_error);
    //assign_to_dict(dict, "num_goals_reached", log->num_goals_reached);
    assign_to_dict(dict, "completion_rate", log->completion_rate);
    assign_to_dict(dict, "lane_alignment_rate", log->lane_alignment_rate);
    assign_to_dict(dict, "score", log->score);
    // assign_to_dict(dict, "active_agent_count", log->active_agent_count);
    // assign_to_dict(dict, "expert_static_car_count", log->expert_static_car_count);
    // assign_to_dict(dict, "static_car_count", log->static_car_count);
    assign_to_dict(dict, "avg_offroad_per_agent", log->avg_offroad_per_agent);
    assign_to_dict(dict, "avg_stop_line_per_agent", log->avg_stop_line_per_agent);
    assign_to_dict(dict, "avg_collisions_per_agent", log->avg_collisions_per_agent);
    assign_to_dict(dict, "close_to_lane", log->close_to_lane);
    assign_to_dict(dict, "lane_heading_angle_diff", log->lane_heading_angle_diff);
    assign_to_dict(dict, "lane_lateral_offset", log->lane_lateral_offset);
    assign_to_dict(dict, "avg_distance_driven_per_agent", log->avg_distance_driven_per_agent);

    if (assign_int_array_to_dict(dict, "collision_counts", collision_counts, COLLISION_COUNT_BUCKETS)) {
        return 1;
    }
    if (assign_int_array_to_dict(dict, "expert_collision_counts", expert_collision_counts, COEFFICIENT_COUNT_BINS)) {
        return 1;
    }
    if (assign_int_array_to_dict(dict, "offroad_counts", offroad_counts, COEFFICIENT_COUNT_BINS)) {
        return 1;
    }
    if (assign_int_array_to_dict(dict, "stop_line_counts", stop_line_counts, COEFFICIENT_COUNT_BINS)) {
        return 1;
    }
    if (assign_int_array_to_dict(dict, "comfort_violation_counts", comfort_violation_counts, COEFFICIENT_COUNT_BINS)) {
        return 1;
    }
    if (assign_int_array_to_dict(dict, "lane_center_offset_counts", lane_center_offset_counts, LANE_CENTER_OFFSET_BUCKETS)) {
        return 1;
    }
    if (assign_int_array_to_dict(dict, "acceleration_action_counts", acceleration_action_counts, CLASSIC_ACCELERATION_ACTION_BINS)) {
        return 1;
    }
    if (assign_int_array_to_dict(dict, "steering_action_counts", steering_action_counts, CLASSIC_STEERING_ACTION_BINS)) {
        return 1;
    }
    if (assign_int_array_to_dict(dict, "speed_counts", speed_counts, SPEED_COUNT_BINS)) {
        return 1;
    }

    return 0;
}
