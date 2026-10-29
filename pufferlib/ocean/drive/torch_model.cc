#include "torch_model.h"
#include "observation.h"

#include <torch/cuda.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr int kJerkDynamics = 1;
constexpr int kVehicles = 63;
constexpr int kVehicleDim = 7;
constexpr int kRoadObjects = 200;
constexpr int kObjectDim = 7;

using StateDict = std::unordered_map<std::string, torch::Tensor>;

struct ModelOutput {
    std::vector<torch::Tensor> logits;
    torch::Tensor value;
    torch::Tensor waypoint_means;
};

torch::Device select_device() {
    const char* override = std::getenv("PUFFER_TORCH_DEVICE");
    if (override != nullptr && override[0] != '\0') {
        return torch::Device(override);
    }

    if (torch::cuda::is_available()) {
        return torch::Device(torch::kCUDA, 0);
    }

    return torch::Device(torch::kCPU);
}

std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open model file: " + path);
    }

    return std::vector<char>(
        std::istreambuf_iterator<char>(in),
        std::istreambuf_iterator<char>());
}

std::string python_executable() {
    const char* override = std::getenv("PUFFER_TORCH_PYTHON");
    if (override != nullptr && override[0] != '\0') {
        return override;
    }

    if (access(".venv/bin/python", X_OK) == 0) {
        return ".venv/bin/python";
    }

    return "python3";
}

std::string convert_ordered_dict_checkpoint(const std::string& path) {
    char tmp_template[] = "/tmp/puffer_torch_state_XXXXXX";
    int fd = mkstemp(tmp_template);
    if (fd == -1) {
        throw std::runtime_error("failed to create temporary checkpoint file");
    }
    close(fd);

    const std::string tmp_path(tmp_template);
    const std::string code =
        "import sys, torch\n"
        "sd = torch.load(sys.argv[1], map_location='cpu', weights_only=True)\n"
        "torch.save(dict(sd), sys.argv[2])\n";

    const std::string cmd =
        shell_quote(python_executable()) + " -c " + shell_quote(code) + " " +
        shell_quote(path) + " " + shell_quote(tmp_path);

    int rc = std::system(cmd.c_str());
    if (rc == -1 || !WIFEXITED(rc) || WEXITSTATUS(rc) != 0) {
        std::remove(tmp_path.c_str());
        throw std::runtime_error(
            "failed to convert OrderedDict checkpoint with Python: " + path);
    }

    return tmp_path;
}

StateDict ivalue_to_state_dict(const torch::IValue& value) {
    if (!value.isGenericDict()) {
        throw std::runtime_error("checkpoint did not deserialize to a dict");
    }

    StateDict state;
    auto dict = value.toGenericDict();
    for (const auto& item : dict) {
        if (!item.key().isString() || !item.value().isTensor()) {
            continue;
        }

        state.emplace(
            item.key().toStringRef(),
            item.value().toTensor().detach().to(torch::kCPU).contiguous());
    }

    return state;
}

StateDict load_state_dict(const std::string& path) {
    try {
        return ivalue_to_state_dict(torch::pickle_load(read_file(path)));
    } catch (const c10::Error&) {
        std::string converted = convert_ordered_dict_checkpoint(path);
        try {
            StateDict state =
                ivalue_to_state_dict(torch::pickle_load(read_file(converted)));
            std::remove(converted.c_str());
            return state;
        } catch (...) {
            std::remove(converted.c_str());
            throw;
        }
    }
}

const torch::Tensor& require_tensor(const StateDict& state, const std::string& key) {
    auto it = state.find(key);
    if (it == state.end()) {
        throw std::runtime_error("checkpoint missing tensor: " + key);
    }
    return it->second;
}

int64_t dim0(const StateDict& state, const std::string& key) {
    return require_tensor(state, key).size(0);
}

int64_t dim1(const StateDict& state, const std::string& key) {
    return require_tensor(state, key).size(1);
}

int count_indexed_prefix(const StateDict& state, const std::string& prefix) {
    int count = 0;
    while (state.find(prefix + std::to_string(count) + ".attn.in_proj_weight") != state.end()) {
        count++;
    }
    return count;
}

void load_module_parameters(torch::nn::Module& module, const StateDict& state) {
    torch::NoGradGuard no_grad;
    std::vector<std::string> missing;

    for (auto& item : module.named_parameters(/*recurse=*/true)) {
        auto it = state.find(item.key());
        if (it == state.end()) {
            missing.push_back(item.key());
            continue;
        }

        torch::Tensor dst = item.value();
        torch::Tensor src = it->second.to(dst.device(), dst.scalar_type());
        if (!dst.sizes().equals(src.sizes())) {
            throw std::runtime_error(
                "shape mismatch for " + item.key() + ": checkpoint " +
                c10::str(src.sizes()) + ", model " + c10::str(dst.sizes()));
        }
        dst.copy_(src);
    }

    for (auto& item : module.named_buffers(/*recurse=*/true)) {
        auto it = state.find(item.key());
        if (it == state.end()) {
            continue;
        }

        torch::Tensor dst = item.value();
        torch::Tensor src = it->second.to(dst.device(), dst.scalar_type());
        if (!dst.sizes().equals(src.sizes())) {
            throw std::runtime_error(
                "shape mismatch for buffer " + item.key() + ": checkpoint " +
                c10::str(src.sizes()) + ", model " + c10::str(dst.sizes()));
        }
        dst.copy_(src);
    }

    if (!missing.empty()) {
        std::string message = "checkpoint missing model parameters:";
        for (const auto& key : missing) {
            message += " " + key;
        }
        throw std::runtime_error(message);
    }
}

std::vector<int64_t> action_dims_from_size(int64_t action_size, int dynamics_model) {
    if (dynamics_model == kJerkDynamics || action_size == 7) {
        return {4, 3};
    }
    return {7, action_size - 7};
}

torch::Tensor select_columns(const torch::Tensor& x, const std::vector<int64_t>& indices) {
    torch::Tensor index = torch::tensor(
        indices,
        torch::TensorOptions().dtype(torch::kLong).device(x.device()));
    return x.index_select(1, index);
}

torch::Tensor road_one_hot(const torch::Tensor& road_objects) {
    torch::Tensor road_continuous = road_objects.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(0, 6)});
    torch::Tensor road_categorical = road_objects.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        6});
    torch::Tensor road_onehot =
        torch::one_hot(road_categorical.to(torch::kLong), 8).to(torch::kFloat32);
    return torch::cat({road_continuous, road_onehot}, 2);
}

struct DrivePolicyImpl : torch::nn::Module {
    DrivePolicyImpl(
        int64_t ego_dim_,
        int64_t input_size_,
        int64_t hidden_size_,
        const std::vector<int64_t>& action_dims_)
        : ego_dim(ego_dim_),
          input_size(input_size_),
          hidden_size(hidden_size_),
          action_dims(action_dims_) {
        ego_encoder = register_module(
            "ego_encoder",
            torch::nn::Sequential(
                torch::nn::Linear(ego_dim, input_size),
                torch::nn::LayerNorm(torch::nn::LayerNormOptions({input_size})),
                torch::nn::Linear(input_size, input_size)));
        road_encoder = register_module(
            "road_encoder",
            torch::nn::Sequential(
                torch::nn::Linear(14, input_size),
                torch::nn::LayerNorm(torch::nn::LayerNormOptions({input_size})),
                torch::nn::Linear(input_size, input_size)));
        partner_encoder = register_module(
            "partner_encoder",
            torch::nn::Sequential(
                torch::nn::Linear(kVehicleDim, input_size),
                torch::nn::LayerNorm(torch::nn::LayerNormOptions({input_size})),
                torch::nn::Linear(input_size, input_size)));
        shared_embedding = register_module(
            "shared_embedding",
            torch::nn::Sequential(
                torch::nn::GELU(),
                torch::nn::Linear(3 * input_size, hidden_size)));
        actor = register_module(
            "actor",
            torch::nn::Linear(hidden_size, action_dims[0] + action_dims[1]));
        value_fn = register_module("value_fn", torch::nn::Linear(hidden_size, 1));
    }

    torch::Tensor encode_observations(const torch::Tensor& observations) {
        int64_t batch = observations.size(0);
        int64_t partner_dim = kVehicles * kVehicleDim;
        int64_t road_dim = kRoadObjects * kObjectDim;

        torch::Tensor ego_obs = observations.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(0, ego_dim)});
        torch::Tensor partner_obs = observations.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(ego_dim, ego_dim + partner_dim)});
        torch::Tensor road_obs = observations.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(ego_dim + partner_dim, ego_dim + partner_dim + road_dim)});

        torch::Tensor partner_objects = partner_obs.view({batch, kVehicles, kVehicleDim});
        torch::Tensor road_objects = road_obs.view({batch, kRoadObjects, kObjectDim});
        road_objects = road_one_hot(road_objects);

        torch::Tensor ego_features = ego_encoder->forward(ego_obs);
        torch::Tensor partner_features =
            std::get<0>(partner_encoder->forward(partner_objects).max(1));
        torch::Tensor road_features =
            std::get<0>(road_encoder->forward(road_objects).max(1));

        torch::Tensor concat_features =
            torch::cat({ego_features, road_features, partner_features}, 1);
        return torch::relu(shared_embedding->forward(concat_features));
    }

    ModelOutput decode_actions(const torch::Tensor& hidden) {
        torch::Tensor action = actor->forward(hidden);
        ModelOutput output;
        output.logits = action.split(action_dims, 1);
        output.value = value_fn->forward(hidden);
        return output;
    }

    int64_t ego_dim;
    int64_t input_size;
    int64_t hidden_size;
    std::vector<int64_t> action_dims;
    torch::nn::Sequential ego_encoder{nullptr};
    torch::nn::Sequential road_encoder{nullptr};
    torch::nn::Sequential partner_encoder{nullptr};
    torch::nn::Sequential shared_embedding{nullptr};
    torch::nn::Linear actor{nullptr};
    torch::nn::Linear value_fn{nullptr};
};

TORCH_MODULE(DrivePolicy);

struct DriveRecurrentImpl : torch::nn::Module {
    DriveRecurrentImpl(
        int64_t ego_dim_,
        int64_t input_size_,
        int64_t hidden_size_,
        const std::vector<int64_t>& action_dims_,
        int64_t num_agents)
        : ego_dim(ego_dim_),
          input_size(input_size_),
          hidden_size(hidden_size_),
          action_dims(action_dims_) {
        policy = register_module(
            "policy",
            DrivePolicy(ego_dim, input_size, hidden_size, action_dims));
        cell = register_module(
            "cell",
            torch::nn::LSTMCell(torch::nn::LSTMCellOptions(hidden_size, hidden_size)));
        reset_state(num_agents, torch::Device(torch::kCPU));
    }

    void reset_state(int64_t num_agents, const torch::Device& device) {
        lstm_h = torch::zeros({num_agents, hidden_size}, torch::TensorOptions().device(device));
        lstm_c = torch::zeros({num_agents, hidden_size}, torch::TensorOptions().device(device));
    }

    ModelOutput forward_eval(const torch::Tensor& observations) {
        torch::Tensor hidden = policy->encode_observations(observations);
        auto next = cell->forward(hidden, std::make_tuple(lstm_h, lstm_c));
        lstm_h = std::get<0>(next);
        lstm_c = std::get<1>(next);
        return policy->decode_actions(lstm_h);
    }

    int64_t ego_dim;
    int64_t input_size;
    int64_t hidden_size;
    std::vector<int64_t> action_dims;
    DrivePolicy policy{nullptr};
    torch::nn::LSTMCell cell{nullptr};
    torch::Tensor lstm_h;
    torch::Tensor lstm_c;
};

TORCH_MODULE(DriveRecurrent);

struct SceneEncoderBlockImpl : torch::nn::Module {
    explicit SceneEncoderBlockImpl(int64_t hidden_dim) {
        attn = register_module(
            "attn",
            torch::nn::MultiheadAttention(
                torch::nn::MultiheadAttentionOptions(hidden_dim, 4)));
        layer_norm1 = register_module(
            "layer_norm1",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
        mlp = register_module(
            "mlp",
            torch::nn::Sequential(
                torch::nn::Linear(hidden_dim, 4 * hidden_dim),
                torch::nn::ReLU(),
                torch::nn::Linear(4 * hidden_dim, hidden_dim)));
        layer_norm2 = register_module(
            "layer_norm2",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
    }

    torch::Tensor forward(const torch::Tensor& x_in) {
        torch::Tensor x = x_in;
        torch::Tensor norm_x = layer_norm1->forward(x);
        torch::Tensor norm_seq = norm_x.transpose(0, 1);
        auto attn_result = attn->forward(norm_seq, norm_seq, norm_seq, {}, false);
        torch::Tensor attn_out = std::get<0>(attn_result).transpose(0, 1);
        x = x + attn_out;

        torch::Tensor mlp_out = mlp->forward(layer_norm2->forward(x));
        return x + mlp_out;
    }

    torch::nn::MultiheadAttention attn{nullptr};
    torch::nn::LayerNorm layer_norm1{nullptr};
    torch::nn::Sequential mlp{nullptr};
    torch::nn::LayerNorm layer_norm2{nullptr};
};

TORCH_MODULE(SceneEncoderBlock);

struct SceneEncoderImpl : torch::nn::Module {
    SceneEncoderImpl(int64_t ego_dim_, int64_t hidden_dim_, int num_blocks)
        : ego_dim(ego_dim_), hidden_dim(hidden_dim_) {
        for (int i = 8; i < 14; i++) ego_state_indices.push_back(i);
        for (int i = 20; i < ego_dim; i++) ego_state_indices.push_back(i);
        goal_indices = {14, 15, 18, 19};
        next_goal_indices = {16, 17, 18, 19};
        for (int i = 0; i < 8; i++) reward_pref_indices.push_back(i);

        map_proj = register_module(
            "map_proj",
            torch::nn::Sequential(
                torch::nn::Linear(14, hidden_dim),
                torch::nn::ReLU()));
        vehicle_proj = register_module(
            "vehicle_proj",
            torch::nn::Sequential(
                torch::nn::Linear(kVehicleDim, hidden_dim),
                torch::nn::ReLU()));
        ego_state_proj = register_module(
            "ego_state_proj",
            torch::nn::Sequential(
                torch::nn::Linear(static_cast<int64_t>(ego_state_indices.size()), hidden_dim),
                torch::nn::ReLU()));
        goal_proj = register_module(
            "goal_proj",
            torch::nn::Sequential(
                torch::nn::Linear(static_cast<int64_t>(goal_indices.size()), hidden_dim),
                torch::nn::ReLU()));
        next_goal_proj = register_module(
            "next_goal_proj",
            torch::nn::Sequential(
                torch::nn::Linear(static_cast<int64_t>(next_goal_indices.size()), hidden_dim),
                torch::nn::ReLU()));
        reward_pref_proj = register_module(
            "reward_pref_proj",
            torch::nn::Sequential(
                torch::nn::Linear(static_cast<int64_t>(reward_pref_indices.size()), hidden_dim),
                torch::nn::ReLU()));
        reward_film = register_module(
            "reward_film",
            torch::nn::Sequential(
                torch::nn::Linear(static_cast<int64_t>(reward_pref_indices.size()), hidden_dim),
                torch::nn::ReLU(),
                torch::nn::Linear(hidden_dim, 2 * hidden_dim)));
        blocks = register_module("blocks", torch::nn::ModuleList());
        for (int i = 0; i < num_blocks; i++) {
            blocks->push_back(SceneEncoderBlock(hidden_dim));
        }
    }

    torch::Tensor forward(const torch::Tensor& observations) {
        int64_t batch = observations.size(0);
        int64_t partner_dim = kVehicles * kVehicleDim;
        int64_t road_dim = kRoadObjects * kObjectDim;

        torch::Tensor ego_data = observations.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(0, ego_dim)});
        torch::Tensor vehicle_data = observations.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(ego_dim, ego_dim + partner_dim)})
            .view({batch, kVehicles, kVehicleDim});
        torch::Tensor road_obs = observations.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(ego_dim + partner_dim, ego_dim + partner_dim + road_dim)})
            .view({batch, kRoadObjects, kObjectDim});

        torch::Tensor map_data = road_one_hot(road_obs);
        torch::Tensor map_tokens =
            map_proj->forward(map_data.view({batch * kRoadObjects, -1}))
                .view({batch, kRoadObjects, -1});
        torch::Tensor vehicle_tokens =
            vehicle_proj->forward(vehicle_data.reshape({batch * kVehicles, -1}))
                .view({batch, kVehicles, -1});

        torch::Tensor ego_state_token =
            ego_state_proj->forward(select_columns(ego_data, ego_state_indices))
                .view({batch, 1, -1});
        torch::Tensor goal_token =
            goal_proj->forward(select_columns(ego_data, goal_indices))
                .view({batch, 1, -1});
        torch::Tensor next_goal_token =
            next_goal_proj->forward(select_columns(ego_data, next_goal_indices))
                .view({batch, 1, -1});
        torch::Tensor reward_prefs = select_columns(ego_data, reward_pref_indices);
        torch::Tensor reward_pref_token =
            reward_pref_proj->forward(reward_prefs).view({batch, 1, -1});

        torch::Tensor ego_tokens = torch::cat(
            {ego_state_token, goal_token, next_goal_token, reward_pref_token}, 1);
        torch::Tensor x = torch::cat({map_tokens, vehicle_tokens, ego_tokens}, 1);

        std::vector<torch::Tensor> film = reward_film->forward(reward_prefs).chunk(2, -1);
        torch::Tensor gamma = film[0];
        torch::Tensor beta = film[1];
        x = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1);

        for (const auto& block : *blocks) {
            x = block->as<SceneEncoderBlock>()->forward(x);
        }
        return x;
    }

    int64_t ego_dim;
    int64_t hidden_dim;
    std::vector<int64_t> ego_state_indices;
    std::vector<int64_t> goal_indices;
    std::vector<int64_t> next_goal_indices;
    std::vector<int64_t> reward_pref_indices;
    torch::nn::Sequential map_proj{nullptr};
    torch::nn::Sequential vehicle_proj{nullptr};
    torch::nn::Sequential ego_state_proj{nullptr};
    torch::nn::Sequential goal_proj{nullptr};
    torch::nn::Sequential next_goal_proj{nullptr};
    torch::nn::Sequential reward_pref_proj{nullptr};
    torch::nn::Sequential reward_film{nullptr};
    torch::nn::ModuleList blocks{nullptr};
};

TORCH_MODULE(SceneEncoder);

struct TrajectoryDecoderBlockImpl : torch::nn::Module {
    explicit TrajectoryDecoderBlockImpl(int64_t hidden_dim) {
        attn = register_module(
            "attn",
            torch::nn::MultiheadAttention(
                torch::nn::MultiheadAttentionOptions(hidden_dim, 4)));
        layer_norm1 = register_module(
            "layer_norm1",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
        mlp = register_module(
            "mlp",
            torch::nn::Sequential(
                torch::nn::Linear(hidden_dim, 4 * hidden_dim),
                torch::nn::ReLU(),
                torch::nn::Linear(4 * hidden_dim, hidden_dim)));
        layer_norm2 = register_module(
            "layer_norm2",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
    }

    torch::Tensor forward(const torch::Tensor& x_in, const torch::Tensor& encoder_outputs) {
        torch::Tensor x = x_in;
        torch::Tensor norm_x = layer_norm1->forward(x);
        auto attn_result = attn->forward(
            norm_x.transpose(0, 1),
            encoder_outputs.transpose(0, 1),
            encoder_outputs.transpose(0, 1),
            {},
            false);
        torch::Tensor attn_out = std::get<0>(attn_result).transpose(0, 1);
        x = x + attn_out;

        torch::Tensor mlp_out = mlp->forward(layer_norm2->forward(x));
        return x + mlp_out;
    }

    torch::nn::MultiheadAttention attn{nullptr};
    torch::nn::LayerNorm layer_norm1{nullptr};
    torch::nn::Sequential mlp{nullptr};
    torch::nn::LayerNorm layer_norm2{nullptr};
};

TORCH_MODULE(TrajectoryDecoderBlock);

struct TrajectoryDecoderImpl : torch::nn::Module {
    TrajectoryDecoderImpl(
        const std::vector<int64_t>& action_dims_,
        int64_t hidden_dim_,
        int num_blocks,
        bool predict_waypoints)
        : action_dims(action_dims_), hidden_dim(hidden_dim_) {
        seed = register_parameter(
            "seed",
            torch::randn({1, 1, hidden_dim}) * 0.02);
        blocks = register_module("blocks", torch::nn::ModuleList());
        for (int i = 0; i < num_blocks; i++) {
            blocks->push_back(TrajectoryDecoderBlock(hidden_dim));
        }
        actor = register_module(
            "actor",
            torch::nn::Linear(hidden_dim, action_dims[0] + action_dims[1]));
        value_fn = register_module("value_fn", torch::nn::Linear(hidden_dim, 1));
        if (predict_waypoints) {
            waypoints = register_module(
                "waypoints",
                torch::nn::Linear(hidden_dim, TORCH_MODEL_WAYPOINT_COUNT * 2 * 2));
        }
    }

    ModelOutput forward(const torch::Tensor& encoder_outputs) {
        torch::Tensor x = seed.repeat({encoder_outputs.size(0), 1, 1});
        for (const auto& block : *blocks) {
            x = block->as<TrajectoryDecoderBlock>()->forward(x, encoder_outputs);
        }

        torch::Tensor flat = x.reshape({-1, hidden_dim});
        torch::Tensor action = actor->forward(flat);
        ModelOutput output;
        output.logits = action.split(action_dims, 1);
        output.value = value_fn->forward(flat);
        if (waypoints) {
            // Python's [..., 0] selects the mean, not the predicted scale.
            output.waypoint_means = waypoints->forward(flat)
                .view({-1, TORCH_MODEL_WAYPOINT_COUNT, 2, 2})
                .select(3, 0);
        }
        return output;
    }

    std::vector<int64_t> action_dims;
    int64_t hidden_dim;
    torch::Tensor seed;
    torch::nn::ModuleList blocks{nullptr};
    torch::nn::Linear actor{nullptr};
    torch::nn::Linear value_fn{nullptr};
    torch::nn::Linear waypoints{nullptr};
};

TORCH_MODULE(TrajectoryDecoder);

struct WayformerImpl : torch::nn::Module {
    WayformerImpl(
        int64_t ego_dim_,
        int64_t hidden_dim_,
        int encoder_blocks,
        int decoder_blocks,
        const std::vector<int64_t>& action_dims_,
        bool predict_waypoints)
        : ego_dim(ego_dim_),
          hidden_dim(hidden_dim_),
          action_dims(action_dims_) {
        encoder = register_module(
            "encoder",
            SceneEncoder(ego_dim, hidden_dim, encoder_blocks));
        decoder = register_module(
            "decoder",
            TrajectoryDecoder(action_dims, hidden_dim, decoder_blocks, predict_waypoints));
    }

    ModelOutput forward_eval(const torch::Tensor& observations) {
        return decoder->forward(encoder->forward(observations));
    }

    int64_t ego_dim;
    int64_t hidden_dim;
    std::vector<int64_t> action_dims;
    SceneEncoder encoder{nullptr};
    TrajectoryDecoder decoder{nullptr};
};

TORCH_MODULE(Wayformer);

enum class ModelKind {
    Drive,
    Wayformer,
};

} // namespace

struct TorchModel {
    ModelKind kind;
    torch::Device device{torch::kCPU};
    int64_t num_agents;
    int64_t obs_dim;
    std::vector<int64_t> action_dims;
    DriveRecurrent drive{nullptr};
    Wayformer wayformer{nullptr};
    torch::Tensor waypoint_means;
};

extern "C" TorchModel* torch_model_load(
    const char* model_path,
    const char* model_name,
    int num_agents,
    int dynamics_model) {
    try {
        if (model_path == nullptr || model_name == nullptr) {
            throw std::runtime_error("model path and model name are required");
        }

        StateDict state = load_state_dict(model_path);
        std::unique_ptr<TorchModel> net(new TorchModel());
        net->device = select_device();
        net->num_agents = num_agents;
        std::printf("Torch model device: %s\n", net->device.str().c_str());

        std::string name(model_name);
        if (name == "Drive") {
            int64_t ego_dim = dim1(state, "policy.ego_encoder.0.weight");
            if (ego_dim != drive_ego_features(dynamics_model)) {
                throw std::runtime_error("Drive checkpoint uses an incompatible observation layout; retrain for ego history");
            }
            int64_t input_size = dim0(state, "policy.ego_encoder.0.weight");
            int64_t hidden_size = dim0(state, "policy.shared_embedding.1.weight");
            int64_t action_size = dim0(state, "policy.actor.weight");
            net->action_dims = action_dims_from_size(action_size, dynamics_model);
            net->obs_dim = ego_dim + kVehicles * kVehicleDim + kRoadObjects * kObjectDim;
            net->kind = ModelKind::Drive;
            net->drive = DriveRecurrent(
                ego_dim,
                input_size,
                hidden_size,
                net->action_dims,
                num_agents);
            load_module_parameters(*net->drive, state);
            net->drive->to(net->device);
            net->drive->reset_state(num_agents, net->device);
            net->drive->eval();
        } else if (name == "Wayformer") {
            int64_t hidden_dim = dim0(state, "encoder.map_proj.0.weight");
            int64_t ego_state_dim = dim1(state, "encoder.ego_state_proj.0.weight");
            int64_t ego_dim = drive_ego_features(dynamics_model);
            // Seven preferences, four goal features and two next-goal features
            // have their existing tokens; everything else is in the ego token.
            if (ego_state_dim != ego_dim - 13) {
                throw std::runtime_error("Wayformer checkpoint uses an incompatible observation layout; retrain for ego history");
            }
            int64_t action_size = dim0(state, "decoder.actor.weight");
            int encoder_blocks = count_indexed_prefix(state, "encoder.blocks.");
            int decoder_blocks = count_indexed_prefix(state, "decoder.blocks.");
            net->action_dims = action_dims_from_size(action_size, dynamics_model);
            net->obs_dim = ego_dim + kVehicles * kVehicleDim + kRoadObjects * kObjectDim;
            net->kind = ModelKind::Wayformer;
            net->wayformer = Wayformer(
                ego_dim,
                hidden_dim,
                encoder_blocks,
                decoder_blocks,
                net->action_dims,
                state.count("decoder.waypoints.weight") != 0);
            load_module_parameters(*net->wayformer, state);
            net->wayformer->to(net->device);
            net->wayformer->eval();
        } else {
            throw std::runtime_error(
                "unknown model name '" + name + "'; expected Drive or Wayformer");
        }

        return net.release();
    } catch (const c10::Error& e) {
        std::fprintf(stderr, "torch_model_load failed: %s\n", e.what_without_backtrace());
        return nullptr;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "torch_model_load failed: %s\n", e.what());
        return nullptr;
    }
}

extern "C" void torch_model_forward(TorchModel* net, float* observations, int* actions) {
    if (net == nullptr) {
        return;
    }
    net->waypoint_means = torch::Tensor();
    if (observations == nullptr || actions == nullptr) {
        return;
    }

    try {
        torch::NoGradGuard no_grad;
        torch::Tensor obs = torch::from_blob(
            observations,
            {net->num_agents, net->obs_dim},
            torch::TensorOptions().dtype(torch::kFloat32));
        obs = obs.to(net->device);

        ModelOutput output;
        if (net->kind == ModelKind::Drive) {
            output = net->drive->forward_eval(obs);
        } else {
            output = net->wayformer->forward_eval(obs);
        }

        int64_t action_heads = static_cast<int64_t>(output.logits.size());
        for (int64_t head = 0; head < action_heads; head++) {
            torch::Tensor selected = output.logits[head].argmax(1).to(torch::kCPU);
            const int64_t* selected_ptr = selected.data_ptr<int64_t>();
            for (int64_t agent = 0; agent < net->num_agents; agent++) {
                actions[agent * action_heads + head] =
                    static_cast<int>(selected_ptr[agent]);
            }
        }
        if (output.waypoint_means.defined()) {
            net->waypoint_means = output.waypoint_means.to(torch::kCPU).contiguous();
        }
    } catch (const c10::Error& e) {
        std::fprintf(stderr, "torch_model_forward failed: %s\n", e.what_without_backtrace());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "torch_model_forward failed: %s\n", e.what());
    }
}

extern "C" const float* torch_model_waypoint_means(const TorchModel* net) {
    if (net == nullptr || !net->waypoint_means.defined()) {
        return nullptr;
    }
    return net->waypoint_means.data_ptr<float>();
}

extern "C" void torch_model_free(TorchModel* net) {
    delete net;
}
