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
    char* compressed = NULL;
    float* obs_buf = NULL;

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
    fclose(file);

    obs_buf = (float*)malloc(uncompressed_bytes);
    if (obs_buf == NULL) {
        fprintf(stderr, "failed to allocate obs buffer\n");
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
        free(obs_buf);
        return 1;
    }

    float (*obs)[max_obs] = (float (*)[max_obs]) obs_buf;

    int pipefd[2];
    if (pipe(pipefd)) {
        fprintf(stderr, "failed to create pipe\n");
        return 1;
    }

    __pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "failed to fork\n");
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

    int ego_dim = 19;
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
        float lane_center_reward_coefficient = obs[t][7];
        float comfort_reward_coefficient = obs[t][8];
        float lane_align_reward_coefficient = obs[t][9];
        float collision_reward_coefficient = obs[t][10];
        float offroad_reward_coefficient = obs[t][11];
        float lane_center_bias = obs[t][12] * 0.5f;
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

        for (int car_idx = 0; car_idx < 63; car_idx++) {
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
        for (int road_idx = 0; road_idx < 200; road_idx++) {
            float *road_obs = &obs[t][ego_dim + 63 * 7 + road_idx * 7];

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

        DrawText(TextFormat("Road points: %d", valid_road_points), 10, 102, 18, WHITE);

        DrawText(TextFormat("Lane center bias: %04.2f m", lane_center_bias), 550, 10, 18, WHITE);
        DrawText(TextFormat("Lane center reward: %04.2f%%", lane_center_reward_coefficient * 100), 550, 33, 18, WHITE);
        DrawText(TextFormat("Lane align reward: %04.2f%%", lane_align_reward_coefficient * 100), 550, 56, 18, WHITE);
        DrawText(TextFormat("Comfort reward: %04.2f%%", comfort_reward_coefficient * 100), 550, 79, 18, WHITE);
        DrawText(TextFormat("Collision reward: %04.2f%%", collision_reward_coefficient * 100), 550, 102, 18, WHITE);
        DrawText(TextFormat("Offroad reward: %04.2f%%", offroad_reward_coefficient * 100), 550, 125, 18, WHITE);

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

    return status;
}
