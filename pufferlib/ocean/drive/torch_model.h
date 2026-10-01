#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TorchModel TorchModel;

#define TORCH_MODEL_WAYPOINT_COUNT 5

TorchModel* torch_model_load(
    const char* model_path,
    const char* model_name,
    int num_agents,
    int dynamics_model);

void torch_model_forward(TorchModel* net, float* observations, int* actions);

// Latest means in ego-frame meters, laid out as [agent][waypoint][x, y].
// Valid until the next forward call or free; NULL if the model has no waypoint head.
const float* torch_model_waypoint_means(const TorchModel* net);

void torch_model_free(TorchModel* net);

#ifdef __cplusplus
}
#endif
