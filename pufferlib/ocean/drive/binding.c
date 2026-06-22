#include "drive.h"
#define Env Drive
#define MY_SHARED
#define MY_PUT
#include "../env_binding.h"

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
        for(int j=0;j<env->num_entities;j++) {
            free_entity(&env->entities[j]);
        }
        free(env->entities);
        free(env->active_agent_indices);
        free(env->static_car_indices);
        free(env->expert_static_car_indices);
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

    //Py_DECREF(num);
    /*
    for(int i = 0;i<num_envs; i++) {
        for(int j=0;j<temp_envs[i].num_entities;j++) {
            free_entity(&temp_envs[i].entities[j]);
        }
        free(temp_envs[i].entities);
        free(temp_envs[i].active_agent_indices);
        free(temp_envs[i].static_car_indices);
    }
    free(temp_envs);
    */
    // return agent_offsets;
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
        PyObject* object_count = PyTuple_GetItem(map, 0);
        if (object_count == NULL) {
            PyErr_SetString(PyExc_TypeError, "object_count is not present");
            return 1;
        }
        env->num_objects = PyLong_AsLong(object_count);
        if (env->num_objects < 1) {
            PyErr_SetString(PyExc_TypeError, "expected at least one object");
            return 1;
        }

        PyObject* road_count = PyTuple_GetItem(map, 1);
        if (road_count == NULL) {
            PyErr_SetString(PyExc_TypeError, "road_count is not present");
            return 1;
        }
        env->num_roads = PyLong_AsLong(road_count);
        if (env->num_roads < 1) {
            PyErr_SetString(PyExc_TypeError, "expected at least one road");
            return 1;
        }

        env->num_entities = env->num_objects + env->num_roads;

        PyObject* entity_ptr = PyTuple_GetItem(map, 2);
        if (entity_ptr == NULL) {
            PyErr_SetString(PyExc_TypeError, "entity_ptr is not present");
            return 1;
        }
        env->entities = (Entity*)PyLong_AsVoidPtr(entity_ptr);
        if (env->entities == NULL) {
            PyErr_SetString(PyExc_TypeError, "entity pointer is null");
            return 1;
        }
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
    int* blind_vehicle_collision_counts,
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

    if (assign_int_array_to_dict(dict, "collision_counts", collision_counts, COLLISION_COUNT_BUCKETS)) {
        return 1;
    }
    if (assign_int_array_to_dict(dict, "expert_collision_counts", expert_collision_counts, COEFFICIENT_COUNT_BINS)) {
        return 1;
    }
    if (assign_int_array_to_dict(dict, "blind_vehicle_collision_counts", blind_vehicle_collision_counts, COEFFICIENT_COUNT_BINS)) {
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
