// Compile:
// gcc agentpov.c -O3 -Iraylib-5.5_linux_amd64/include ./raylib-5.5_linux_amd64/lib/libraylib.a -lm -llz4 -o agentpov

// Run:
// xvfb-run -s "-screen 0 1280x720x24" ./agentpov experiments/../agent_00.bin

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <math.h>
#include <raylib.h>
#include <rlgl.h>
#include <lz4.h>

#define DISPLAY_REWARD_COMPONENT_COUNT 6
#define DISPLAY_ACTION_COMPONENT_COUNT 2

static const char *DISPLAY_REWARD_LABELS[DISPLAY_REWARD_COMPONENT_COUNT] = {
    "Goal",
    "Lane center",
    "Comfort",
    "Lane alignment",
    "Collision",
    "Offroad",
};

static const float DISPLAY_ACCELERATION_VALUES[7] = {
    -4.0000f, -2.6670f, -1.3330f, -0.0000f, 1.3330f, 2.6670f, 4.0000f,
};

static const float DISPLAY_STEERING_VALUES[13] = {
    -1.000f, -0.833f, -0.667f, -0.500f, -0.333f, -0.167f, 0.000f,
    0.167f, 0.333f, 0.500f, 0.667f, 0.833f, 1.000f,
};

static int write_all(int fd, const unsigned char *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t written = write(fd, buf + total, len - total);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (written == 0) {
            return -1;
        }
        total += (size_t)written;
    }
    return 0;
}

static void print_usage(const char *prog) {
    fprintf(stderr, "usage: %s <path to file> [path to output] [--zoom <value>]\n", prog);
}

static int parse_positive_float(const char *arg, float *value) {
    char *end = NULL;
    errno = 0;
    float parsed = strtof(arg, &end);
    if (errno != 0 || end == arg || *end != '\0' || !isfinite(parsed) || parsed <= 0.0f) {
        return -1;
    }

    *value = parsed;
    return 0;
}

int main(int argc, char *argv[]) {
    const char* filepath = NULL;
    const char* output_path = "pov.mp4";
    int output_path_set = 0;
    float camera_zoom = 8.0f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--zoom") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for --zoom\n");
                print_usage(argv[0]);
                return 1;
            }

            if (parse_positive_float(argv[i + 1], &camera_zoom) != 0) {
                fprintf(stderr, "invalid --zoom value: %s\n", argv[i + 1]);
                return 1;
            }
            i++;
            continue;
        }

        if (filepath == NULL) {
            filepath = argv[i];
        } else if (!output_path_set) {
            output_path = argv[i];
            output_path_set = 1;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (filepath == NULL) {
        print_usage(argv[0]);
        return 1;
    }

    FILE* file = fopen(filepath, "rb");
    if (file == NULL) {
        fprintf(stderr, "failed to open file\n");
        return 1;
    }

    int max_obs = 0;
    int exported_steps = 0;
    int compressed_size = 0;
    int reward_component_count = 0;
    int reward_compressed_size = 0;
    int action_component_count = 0;
    char* compressed = NULL;
    float* obs_buf = NULL;
    char* reward_compressed = NULL;
    float* reward_values = NULL;
    float* reward_cumulative = NULL;
    int* action_values = NULL;

    if (fread(&max_obs, sizeof(max_obs), 1, file) != 1) {
        fprintf(stderr, "failed to read max_obs\n");
        fclose(file);
        return 1;
    }
    if (fread(&exported_steps, sizeof(exported_steps), 1, file) != 1) {
        fprintf(stderr, "failed to read exported_steps\n");
        fclose(file);
        return 1;
    }
    if (fread(&compressed_size, sizeof(compressed_size), 1, file) != 1) {
        fprintf(stderr, "failed to read compressed_size\n");
        fclose(file);
        return 1;
    }
    if (max_obs <= 0 || exported_steps <= 0 || compressed_size <= 0) {
        fprintf(stderr,
                "invalid header values: max_obs=%d exported_steps=%d compressed_size=%d\n",
                max_obs, exported_steps, compressed_size);
        fclose(file);
        return 1;
    }
    fprintf(stderr, "max obs: %d, exported steps: %d\n", max_obs, exported_steps);

    size_t total_obs = (size_t)exported_steps * (size_t)max_obs;
    if (total_obs > (SIZE_MAX / sizeof(float))) {
        fprintf(stderr, "obs buffer too large\n");
        fclose(file);
        return 1;
    }
    size_t uncompressed_bytes = total_obs * sizeof(float);
    if (uncompressed_bytes > INT_MAX) {
        fprintf(stderr, "obs buffer too large for LZ4\n");
        fclose(file);
        return 1;
    }

    compressed = (char*)malloc((size_t)compressed_size);
    if (compressed == NULL) {
        fprintf(stderr, "failed to allocate compressed buffer\n");
        fclose(file);
        return 1;
    }
    if (fread(compressed, 1, (size_t)compressed_size, file) != (size_t)compressed_size) {
        fprintf(stderr, "failed to read compressed obs data\n");
        fclose(file);
        free(compressed);
        return 1;
    }

    obs_buf = (float*)malloc(uncompressed_bytes);
    if (obs_buf == NULL) {
        fprintf(stderr, "failed to allocate obs buffer\n");
        fclose(file);
        free(compressed);
        return 1;
    }
    int decoded = LZ4_decompress_safe(compressed,
                                      (char*)obs_buf,
                                      compressed_size,
                                      (int)uncompressed_bytes);
    free(compressed);
    if (decoded != (int)uncompressed_bytes) {
        fprintf(stderr, "failed to decompress obs data\n");
        fclose(file);
        free(obs_buf);
        return 1;
    }

    size_t total_display_rewards = (size_t)exported_steps * DISPLAY_REWARD_COMPONENT_COUNT;
    reward_values = (float*)malloc(total_display_rewards * sizeof(float));
    reward_cumulative = (float*)calloc(total_display_rewards, sizeof(float));
    if (reward_values == NULL || reward_cumulative == NULL) {
        fprintf(stderr, "failed to allocate reward buffers\n");
        fclose(file);
        free(obs_buf);
        free(reward_values);
        free(reward_cumulative);
        return 1;
    }

    if (fread(&reward_component_count, sizeof(reward_component_count), 1, file) != 1) {
        fprintf(stderr, "failed to read reward_component_count\n");
        fclose(file);
        free(obs_buf);
        free(reward_values);
        free(reward_cumulative);
        return 1;
    }

    if (reward_component_count != DISPLAY_REWARD_COMPONENT_COUNT) {
        fprintf(stderr, "unexpected reward_component_count=%d\n", reward_component_count);
        fclose(file);
        free(obs_buf);
        free(reward_values);
        free(reward_cumulative);
        return 1;
    }

    if (fread(&reward_compressed_size, sizeof(reward_compressed_size), 1, file) != 1) {
        fprintf(stderr, "failed to read reward_compressed_size\n");
        fclose(file);
        free(obs_buf);
        free(reward_values);
        free(reward_cumulative);
        return 1;
    }
    if (reward_compressed_size <= 0) {
        fprintf(stderr, "invalid reward_compressed_size=%d\n", reward_compressed_size);
        fclose(file);
        free(obs_buf);
        free(reward_values);
        free(reward_cumulative);
        return 1;
    }

    size_t reward_uncompressed_bytes = total_display_rewards * sizeof(float);
    if (reward_uncompressed_bytes > INT_MAX) {
        fprintf(stderr, "reward buffer too large for LZ4\n");
        fclose(file);
        free(obs_buf);
        free(reward_values);
        free(reward_cumulative);
        return 1;
    }

    reward_compressed = (char*)malloc((size_t)reward_compressed_size);
    if (reward_compressed == NULL) {
        fprintf(stderr, "failed to allocate compressed reward buffer\n");
        fclose(file);
        free(obs_buf);
        free(reward_values);
        free(reward_cumulative);
        return 1;
    }
    if (fread(reward_compressed, 1, (size_t)reward_compressed_size, file) != (size_t)reward_compressed_size) {
        fprintf(stderr, "failed to read compressed reward data\n");
        fclose(file);
        free(obs_buf);
        free(reward_compressed);
        free(reward_values);
        free(reward_cumulative);
        return 1;
    }

    int decoded_rewards = LZ4_decompress_safe(
        reward_compressed,
        (char*)reward_values,
        reward_compressed_size,
        (int)reward_uncompressed_bytes);
    free(reward_compressed);
    reward_compressed = NULL;
    if (decoded_rewards != (int)reward_uncompressed_bytes) {
        fprintf(stderr, "failed to decompress reward data\n");
        fclose(file);
        free(obs_buf);
        free(reward_values);
        free(reward_cumulative);
        return 1;
    }

    if (fread(&action_component_count, sizeof(action_component_count), 1, file) != 1) {
        fprintf(stderr, "failed to read action_component_count\n");
        fclose(file);
        free(obs_buf);
        free(reward_values);
        free(reward_cumulative);
        return 1;
    }

    if (action_component_count != DISPLAY_ACTION_COMPONENT_COUNT) {
        fprintf(stderr, "unexpected action_component_count=%d\n", action_component_count);
        fclose(file);
        free(obs_buf);
        free(reward_values);
        free(reward_cumulative);
        return 1;
    }

    size_t total_actions = (size_t)exported_steps * DISPLAY_ACTION_COMPONENT_COUNT;
    if (total_actions > (SIZE_MAX / sizeof(int))) {
        fprintf(stderr, "action buffer too large\n");
        fclose(file);
        free(obs_buf);
        free(reward_values);
        free(reward_cumulative);
        return 1;
    }

    action_values = (int*)malloc(total_actions * sizeof(int));
    if (action_values == NULL) {
        fprintf(stderr, "failed to allocate action buffer\n");
        fclose(file);
        free(obs_buf);
        free(reward_values);
        free(reward_cumulative);
        return 1;
    }

    if (fread(action_values, sizeof(int), total_actions, file) != total_actions) {
        fprintf(stderr, "failed to read action data\n");
        fclose(file);
        free(obs_buf);
        free(reward_values);
        free(reward_cumulative);
        free(action_values);
        return 1;
    }

    fclose(file);

    for (int t = 0; t < exported_steps; t++) {
        for (int reward_idx = 0; reward_idx < DISPLAY_REWARD_COMPONENT_COUNT; reward_idx++) {
            float previous = (t > 0)
                ? reward_cumulative[(t - 1) * DISPLAY_REWARD_COMPONENT_COUNT + reward_idx]
                : 0.0f;
            reward_cumulative[t * DISPLAY_REWARD_COMPONENT_COUNT + reward_idx] =
                previous + reward_values[t * DISPLAY_REWARD_COMPONENT_COUNT + reward_idx];
        }
    }

    float (*obs)[max_obs] = (float (*)[max_obs]) obs_buf;
    const int other_agent_count = 63;
    const int road_segment_count = 200;
    int ego_dim = max_obs - other_agent_count * 7 - road_segment_count * 7;
    if (ego_dim < 19) {
        fprintf(stderr, "invalid replay layout: max_obs=%d inferred ego_dim=%d\n", max_obs, ego_dim);
        free(obs_buf);
        free(reward_values);
        free(reward_cumulative);
        free(action_values);
        return 1;
    }

    int pipefd[2];
    if (pipe(pipefd)) {
        fprintf(stderr, "failed to create pipe\n");
        free(obs_buf);
        free(reward_values);
        free(reward_cumulative);
        free(action_values);
        return 1;
    }

    __pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "failed to fork\n");
        close(pipefd[0]);
        close(pipefd[1]);
        free(obs_buf);
        free(reward_values);
        free(reward_cumulative);
        free(action_values);
        return 1;
    }
    if (pid == 0) {
        close(pipefd[1]);
        dup2(pipefd[0], STDIN_FILENO);
        close(pipefd[0]);
        execlp("ffmpeg", "ffmpeg",
               "-y",
               "-f", "rawvideo",
               "-pix_fmt", "rgba",
               "-s", "800x800",
               "-r", "10",
               "-i", "-",
               "-c:v", "libx264",
               "-pix_fmt", "yuv420p",
               "-preset", "ultrafast",
               "-crf", "23",
               "-loglevel", "error",
               output_path,
               NULL);
        return 1;
    }
    close(pipefd[0]);

    SetConfigFlags(FLAG_WINDOW_HIDDEN);
    InitWindow(800, 800, "Agent PoV");
    SetTargetFPS(6000);

    Camera2D camera = { 0 };
    camera.offset = (Vector2){ 400.0f, 400.0f };
    camera.zoom = camera_zoom;

    const float base_line_thickness = 0.2f;
    const float base_zoom = 8.0f;

    int status = 0;
    for (int t = 0; t < exported_steps; t++) {
        float line_thickness = base_line_thickness * (base_zoom / camera.zoom);

        BeginDrawing();
        Color road = (Color){35, 35, 37, 255};
        ClearBackground(road);
        BeginMode2D(camera);
        rlMatrixMode(RL_MODELVIEW);
        rlDisableBackfaceCulling();
        rlPushMatrix();
        rlScalef(-1, 1, 1);
        float goal_x = obs[t][0] / 0.005f;
        float goal_y = obs[t][1] / 0.005f;
        float speed = obs[t][2] * 27.7f;
        float width = obs[t][3] * 15.0f;
        float length = obs[t][4] * 30.0f;
        float lane_heading_angle_diff = obs[t][13];
        float lane_lateral_offset = obs[t][14] * 4.0f;
        float next_goal_x = obs[t][16] / 0.005f;
        float next_goal_y = obs[t][17] / 0.005f;
        float goal_radius = obs[t][18] * 3.0f;

        DrawCircleLines(goal_x, goal_y, goal_radius, DARKGREEN);
        DrawCircleLines(next_goal_x, next_goal_y, goal_radius, LIME);
        DrawLineEx((Vector2){-length / 2, -width / 2}, (Vector2){length / 2, -width / 2}, line_thickness, RED);
        DrawLineEx((Vector2){-length / 2, width / 2}, (Vector2){length / 2, width / 2}, line_thickness, RED);
        DrawLineEx((Vector2){-length / 2, -width / 2}, (Vector2){-length / 2, width / 2}, line_thickness, RED);
        DrawLineEx((Vector2){length / 2, -width / 2}, (Vector2){length / 2, width / 2}, line_thickness, RED);
        DrawLineEx((Vector2){0, 0}, (Vector2){length * 0.625f, 0}, line_thickness, RED);

        for (int car_idx = 0; car_idx < other_agent_count; car_idx++) {
            float* other_car_obs = &obs[t][ego_dim + car_idx * 7];
            
            float rel_x = other_car_obs[0] / 0.02f;
            float rel_y = other_car_obs[1] / 0.02f;
            if (rel_x == 0 && rel_y == 0) continue;

            float other_width = other_car_obs[2] * 15.0f;
            float other_length = other_car_obs[3] * 30.0f;
            float rel_heading_x = other_car_obs[4];
            float rel_heading_y = other_car_obs[5];
            // float other_speed = other_car_obs[6] * 27.7f;

            Vector2 corners[4] = {
                {rel_x + other_length/2*rel_heading_x-other_width/2*rel_heading_y, rel_y + other_length/2*rel_heading_y+other_width/2*rel_heading_x},
                {rel_x + other_length/2*rel_heading_x+other_width/2*rel_heading_y, rel_y + other_length/2*rel_heading_y-other_width/2*rel_heading_x},
                {rel_x - other_length/2*rel_heading_x+other_width/2*rel_heading_y, rel_y - other_length/2*rel_heading_y-other_width/2*rel_heading_x},
                    {rel_x - other_length/2*rel_heading_x-other_width/2*rel_heading_y, rel_y - other_length/2*rel_heading_y+other_width/2*rel_heading_x}
            };

            DrawLineEx(corners[0], corners[1], line_thickness, YELLOW);
            DrawLineEx(corners[1], corners[2], line_thickness, YELLOW);
            DrawLineEx(corners[2], corners[3], line_thickness, YELLOW);
            DrawLineEx(corners[3], corners[0], line_thickness, YELLOW);

            DrawLineEx((Vector2){rel_x, rel_y}, (Vector2){rel_x + rel_heading_x * other_length * 0.625f, rel_y + 0.625f * rel_heading_y * other_length}, line_thickness, YELLOW);
        }

        int valid_road_points = 0;
        for (int road_idx = 0; road_idx < road_segment_count; road_idx++) {
            float *road_obs = &obs[t][ego_dim + other_agent_count * 7 + road_idx * 7];

            float rel_x = road_obs[0] / 0.02f;
            float rel_y = road_obs[1] / 0.02f;

            float road_length = road_obs[2] * 100.0f;
            if (road_length < 1e-5) continue;

            int road_type = roundf(road_obs[6] + 4);
            Color color = WHITE;
            if (road_type == 4) {
                color = GRAY;
            } else if (road_type == 6) {
                color = (Color){220, 220, 220, 255};
            } else if (road_type == 7) {
                color = RED;
            } else if (road_type == 11) {
                color = PINK;
            } else if (road_type == 8) {
                color = YELLOW;
            }

            float rel_heading_x = road_obs[4];
            float rel_heading_y = road_obs[5];

            DrawTriangle(
                (Vector2){rel_x + 0.75f * rel_heading_x, rel_y + 0.75f * rel_heading_y},
                (Vector2){rel_x - 0.75f * rel_heading_x + rel_heading_y * 0.6f, rel_y - 0.75f * rel_heading_y - rel_heading_x * 0.6f},
                (Vector2){rel_x - 0.75f * rel_heading_x - rel_heading_y * 0.6f, rel_y - 0.75f * rel_heading_y + rel_heading_x * 0.6f},
                color);
            DrawLineEx((Vector2){rel_x, rel_y}, (Vector2){rel_x + road_length * rel_heading_x, rel_y + road_length * rel_heading_y}, line_thickness, color);
            DrawLineEx((Vector2){rel_x, rel_y}, (Vector2){rel_x - road_length * rel_heading_x, rel_y - road_length * rel_heading_y}, line_thickness, color);
            valid_road_points++;
        }

        rlPopMatrix();
        EndMode2D();

        DrawText(TextFormat("Lane angle diff: %04.1f degrees", lane_heading_angle_diff / PI * 180.0f), 10, 10, 18, WHITE);
        DrawText(TextFormat("Lane lateral offset: %04.2f m", lane_lateral_offset), 10, 33, 18, WHITE);
        DrawText(TextFormat("Speed: %04.2f km/h", speed * 3.6), 10, 56, 18, WHITE);
        int acceleration_index = -1;
        int steering_index = -1;
        if (action_values != NULL) {
            acceleration_index = action_values[t * action_component_count];
            steering_index = action_values[t * action_component_count + 1];
        }

        if (steering_index >= 0 && steering_index < (int)(sizeof(DISPLAY_STEERING_VALUES) / sizeof(DISPLAY_STEERING_VALUES[0]))) {
            float steering_degrees = DISPLAY_STEERING_VALUES[steering_index] * 180.0f / PI;
            DrawText(TextFormat("Steering action: %+05.1f deg", steering_degrees), 10, 79, 18, WHITE);
        } else {
            DrawText("Steering action: N/A", 10, 79, 18, WHITE);
        }

        if (acceleration_index >= 0 && acceleration_index < (int)(sizeof(DISPLAY_ACCELERATION_VALUES) / sizeof(DISPLAY_ACCELERATION_VALUES[0]))) {
            float acceleration_g = DISPLAY_ACCELERATION_VALUES[acceleration_index] / 9.81f;
            DrawText(TextFormat("Acceleration action: %+0.2f g", acceleration_g), 10, 102, 18, WHITE);
        } else {
            DrawText("Acceleration action: N/A", 10, 102, 18, WHITE);
        }

        DrawText(TextFormat("Road points: %d", valid_road_points), 10, 125, 18, WHITE);

        const int table_x = 420;
        const int table_y = 8;
        const int label_x = table_x + 12;
        const int step_x = table_x + 165;
        const int episode_x = table_x + 255;
        const int row_height = 23;

        DrawText("Reward", label_x, table_y + 4, 18, WHITE);
        DrawText("Step", step_x, table_y + 4, 18, WHITE);
        DrawText("Episode", episode_x, table_y + 4, 18, WHITE);
        DrawLine(table_x + 10, table_y + 26, table_x + 360, table_y + 26, GRAY);

        for (int reward_idx = 0; reward_idx < DISPLAY_REWARD_COMPONENT_COUNT; reward_idx++) {
            int row_y = table_y + 34 + reward_idx * row_height;
            float step_reward = reward_values[t * DISPLAY_REWARD_COMPONENT_COUNT + reward_idx];
            float episode_reward = reward_cumulative[t * DISPLAY_REWARD_COMPONENT_COUNT + reward_idx];
            DrawText(DISPLAY_REWARD_LABELS[reward_idx], label_x, row_y, 18, WHITE);
            DrawText(TextFormat("%+.4f", step_reward), step_x, row_y, 18, WHITE);
            DrawText(TextFormat("%+.4f", episode_reward), episode_x, row_y, 18, WHITE);
        }

        EndDrawing();

        unsigned char *screen_data = rlReadScreenPixels(800, 800);
        size_t frame_bytes = 800 * 800 * 4 * sizeof(unsigned char);
        if (write_all(pipefd[1], screen_data, frame_bytes) != 0) {
            fprintf(stderr, "failed to write frame\n");
            RL_FREE(screen_data);
            status = 1;
            break;
        }
        RL_FREE(screen_data);
    }

    close(pipefd[1]);
    CloseWindow();
    free(obs_buf);
    free(reward_values);
    free(reward_cumulative);
    free(action_values);

    return status;
}
