#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <math.h>
#include <raylib.h>
#include "rlgl.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "error.h"
#include "drive.h"
#include "drivenet.h"
#include "libgen.h"
#include "../env_config.h"
#define TRAJECTORY_LENGTH_DEFAULT 91

typedef struct {
    int pipefd[2];
    pid_t pid;
} VideoRecorder;

bool OpenVideo(VideoRecorder *recorder, const char *output_filename, int width, int height) {
    if (pipe(recorder->pipefd) == -1) {
        fprintf(stderr, "Failed to create pipe\n");
        return false;
    }

    recorder->pid = fork();
    if (recorder->pid == -1) {
        fprintf(stderr, "Failed to fork\n");
        return false;
    }

    char size_str[64];
    snprintf(size_str, sizeof(size_str), "%dx%d", width, height);

    if (recorder->pid == 0) { // Child process: run ffmpeg
        close(recorder->pipefd[1]);
        dup2(recorder->pipefd[0], STDIN_FILENO);
        close(recorder->pipefd[0]);
        // Close all other file descriptors to prevent leaks
        for (int fd = 3; fd < 256; fd++) {
            close(fd);
        }
        execlp("ffmpeg", "ffmpeg",
               "-y",
               "-f", "rawvideo",
               "-pix_fmt", "rgba",
               "-s", size_str,
               "-r", "10",
               "-i", "-",
               "-c:v", "libx264",
               "-pix_fmt", "yuv420p",
               "-preset", "ultrafast",
               "-crf", "23",
               "-loglevel", "error",
               output_filename,
               NULL);
        TraceLog(LOG_ERROR, "Failed to launch ffmpeg");
        return false;
    }

    close(recorder->pipefd[0]); // Close read end in parent
    return true;
}

void WriteFrame(VideoRecorder *recorder, int width, int height) {
    unsigned char *screen_data = rlReadScreenPixels(width, height);
    write(recorder->pipefd[1], screen_data, width * height * 4 * sizeof(*screen_data));
    RL_FREE(screen_data);
}

void CloseVideo(VideoRecorder *recorder) {
    close(recorder->pipefd[1]);
    waitpid(recorder->pid, NULL, 0);
}

void renderTopDownView(Drive* env, Client* client, int map_height, int obs, int lasers, int trajectories, int frame_count, float* path, int log_trajectories, int show_grid, float* reward_log) {

    BeginDrawing();
    AgentLabel labels[MAX_AGENTS] = {0};
    int label_count = 0;

    // Top-down orthographic camera
    Camera3D camera = {0};
    camera.position = (Vector3){ 0.0f, 0.0f, 500.0f };  // above the scene
    camera.target   = (Vector3){ 0.0f, 0.0f, 0.0f };  // look at origin
    camera.up       = (Vector3){ 0.0f, -1.0f, 0.0f };
    camera.fovy     = map_height;
    camera.projection = CAMERA_ORTHOGRAPHIC;

    const char* follow_agent = getenv("FOLLOW_AGENT");
    if (follow_agent && strcmp(follow_agent, "1") == 0 && env->active_agent_count >= 1) {
        int idx = env->active_agent_indices[0];
        Entity* e = &env->entities[idx];
        if (e->reached_final_goal) {
            camera.position = (Vector3){e->goal_position_x, e->goal_position_y, 500.0f};
            camera.target = (Vector3){e->goal_position_x, e->goal_position_y, 0.0f};
            camera.fovy = 150;
        } else {
            camera.position = (Vector3){e->x, e->y, 500.0f};
            camera.target = (Vector3){e->x, e->y, 0.0f};
            camera.fovy = 150;
        }
    }

    Color road = (Color){35, 35, 37, 255};
    ClearBackground(road);
    BeginMode3D(camera);
    rlEnableDepthTest();

    // Draw human replay trajectories if enabled
    if(log_trajectories){
    for(int i=0; i<env->active_agent_count; i++){
        int idx = env->active_agent_indices[i];
        Vector3 prev_point = {0};
        bool has_prev = false;

        for(int j = 0; j < env->entities[idx].array_size; j++){
            float x = env->entities[idx].traj_x[j];
            float y = env->entities[idx].traj_y[j];
            float valid = env->entities[idx].traj_valid[j];

            if(!valid) {
                has_prev = false;
                continue;
            }

            Vector3 curr_point = {x, y, 0.5f};

            if(has_prev) {
                DrawLine3D(prev_point, curr_point, Fade(LIGHTGREEN, 0.6f));
            }

            prev_point = curr_point;
            has_prev = true;
        }
    }
}

    // Draw agent trajs
    if(trajectories){
        for(int i=0; i<frame_count; i++){
            DrawSphere((Vector3){path[i*2], path[i*2 +1], 0.8f}, 0.5f, YELLOW);
        }
    }

    // Draw scene
    draw_scene(env, client, 1, obs, lasers, show_grid, camera, labels, MAX_AGENTS, &label_count);
    EndMode3D();
    for(int i = 0; i < label_count; i++) {
        if(!labels[i].visible) continue;
        // printf("%f %d %f %f\n", labels[i].speed_kph, i, labels[i].screen_pos.x, labels[i].screen_pos.y);
        DrawText(
            TextFormat("%.0f km/h", labels[i].speed_kph),
            (int)labels[i].screen_pos.x,
            (int)labels[i].screen_pos.y,
            48,
            RED
        );
    }

    const char* visualize_reward = getenv("VISUALIZE_REWARD");
    if (visualize_reward && strcmp(visualize_reward, "1") == 0 && env->active_agent_count >= 1) {
        reward_log[env->timestep] = env->rewards[0];

        for (int i = 0; i <= env->timestep / 10; i++) {
            float reward = 0;
            for (int j = 0; j < 10; j++) {
                reward += reward_log[i * 10 + j];
            }

            if (reward > 0) {
                DrawRectangle(80 + i * 4, fmax(170 - sqrtf(reward) * 100, 0), 4, sqrtf(reward) * 100, GREEN);
            } else if (reward < 0) {
                DrawRectangle(80 + i * 4, 170, 4, sqrtf(-reward) * 100, RED);
            }
        }
        DrawLineEx((Vector2){75, 170}, (Vector2){80 + ceilf(env->timestep / 10.0f) * 4, 170}, 3, WHITE);
        DrawLineEx((Vector2){75, 20}, (Vector2){75, 320}, 3, WHITE);

        DrawLineEx((Vector2){65, 170 - sqrtf(2) * 100}, (Vector2){80, 170 - sqrtf(2) * 100}, 3, WHITE);
        DrawText("+2", 35, 158 - sqrtf(2) * 100, 24, WHITE);

        DrawLineEx((Vector2){65, 70}, (Vector2){80, 70}, 3, WHITE);
        DrawText("+1", 35, 58, 24, WHITE);

        DrawLineEx((Vector2){65, 170 - sqrtf(0.5) * 100}, (Vector2){80, 170 - sqrtf(0.5) * 100}, 3, WHITE);
        DrawText("+0.5", 20, 158 - sqrtf(0.5) * 100, 24, WHITE);

        DrawLineEx((Vector2){65, 170 - sqrtf(0.1) * 100}, (Vector2){80, 170 - sqrtf(0.1) * 100}, 3, WHITE);
        DrawText("+0.1", 20, 158 - sqrtf(0.1) * 100, 24, WHITE);

        DrawLineEx((Vector2){65, 170 + sqrtf(2) * 100}, (Vector2){80, 170 + sqrtf(2) * 100}, 3, WHITE);
        DrawText("-2", 35, 158 + sqrtf(2) * 100, 24, WHITE);

        DrawLineEx((Vector2){65, 270}, (Vector2){80, 270}, 3, WHITE);
        DrawText("-1", 35, 258, 24, WHITE);

        DrawLineEx((Vector2){65, 170 + sqrtf(0.5) * 100}, (Vector2){80, 170 + sqrtf(0.5) * 100}, 3, WHITE);
        DrawText("-0.5", 20, 158 + sqrtf(0.5) * 100, 24, WHITE);

        DrawLineEx((Vector2){65, 170 + sqrtf(0.1) * 100}, (Vector2){80, 170 + sqrtf(0.1) * 100}, 3, WHITE);
        DrawText("-0.1", 20, 158 + sqrtf(0.1) * 100, 24, WHITE);
    }

    EndDrawing();
}

int eval_gif(const char* map_name, const char* policy_name, int show_grid, int obs_only, int lasers, int log_trajectories, int frame_skip, float goal_radius, int control_non_vehicles, int init_steps, int control_all_agents, int policy_agents_per_env, int deterministic_selection, const char* view_mode, const char* output_topdown, const char* output_agent, int num_maps, int scenario_length_override) {

    // Parse configuration from INI file
    env_init_config conf = {0};  // Initialize to zero
    const char* ini_file = "pufferlib/config/ocean/drive.ini";
    if(ini_parse(ini_file, handler, &conf) < 0) {
        fprintf(stderr, "Error: Could not load %s. Cannot determine environment configuration.\n", ini_file);
        return -1;
    }

    char map_buffer[100];
    if (map_name == NULL) {
        srand(time(NULL));
        int random_map = rand() % num_maps;
        sprintf(map_buffer, "resources/drive/binaries/map_%03d.bin", random_map); // random map file
        map_name = map_buffer;
    }

    if (frame_skip <= 0) {
        frame_skip = 1;  // Default: render every frame
    }

    // Check if map file exists
    FILE* map_file = fopen(map_name, "rb");
    if (map_file == NULL) {
        RAISE_FILE_ERROR(map_name);
    }
    fclose(map_file);

    FILE* policy_file = fopen(policy_name, "rb");
    if (policy_file == NULL) {
        RAISE_FILE_ERROR(policy_name);
    }
    fclose(policy_file);

    Drive env = {
        .dynamics_model = conf.dynamics_model,
        .reward_goal = 1,
        .reward_vehicle_collision = conf.reward_vehicle_collision,
        .reward_offroad_collision = conf.reward_offroad_collision,
        .reward_ade = conf.reward_ade,
        .goal_radius = conf.goal_radius,
        .dt = conf.dt,
	    .map_name = (char*)map_name,
        .control_non_vehicles = conf.control_non_vehicles,
        .init_steps = conf.init_steps,
        .control_all_agents = conf.control_all_agents,
        .policy_agents_per_env = conf.num_policy_controlled_agents,
        .deterministic_agent_selection = conf.deterministic_agent_selection,
        .export_replay_enabled = true,
        .replay_dir = strdup("replays")
    };
    allocate(&env);

    for (int i = 0; i < env.active_agent_count; i++) {
        Entity* e = &env.entities[env.active_agent_indices[i]];
        e->max_speed = 90/3.6f;
        e->lane_center_reward_coefficient = 2.5e-5;
        e->comfort_reward_coefficient = 0.00f;
        e->lane_align_reward_coefficient = 2.5e-5f;
        e->collision_reward_coefficient = 1.0f;
        e->offroad_reward_coefficient = 1.0f;
        e->lane_center_bias = 0.0f;
        e->goal_radius = goal_radius;
    }
    
    if (scenario_length_override != -1) {
        env.scenario_length = scenario_length_override;
    }

    printf("scenario length is %d\n", env.scenario_length);

    // Set which vehicle to focus on for obs mode
    env.human_agent_idx = 0;

    c_reset(&env);
    // Make client for rendering
    Client* client = (Client*)calloc(1, sizeof(Client));
    env.client = client;

    float map_width = env.grid_map->bottom_right_x - env.grid_map->top_left_x;
    float map_height = env.grid_map->top_left_y - env.grid_map->bottom_right_y;

    printf("Map size: %.1fx%.1f\n", map_width, map_height);
    float scale = 6.0f; // Can be used to increase the video quality

    // Calculate video width and height; round to nearest even number
    int img_width = (int)roundf(map_width * scale / 2.0f) * 2;
    int img_height = (int)roundf(map_height * scale / 2.0f) * 2;
    InitWindow(img_width, img_height, "Puffer Drive");
    SetTargetFPS(6000);
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_HIDDEN);

    Weights* weights = load_weights(policy_name);
    printf("Active agents in map: %d\n", env.active_agent_count);
    DriveNet* net = init_drivenet(weights, env.active_agent_count, env.dynamics_model);

    int frame_count = env.scenario_length > 0 ? env.scenario_length : TRAJECTORY_LENGTH_DEFAULT;
    frame_count -= env.init_steps;
    frame_count -= 1;
    char filename_topdown[1024];

    if (output_topdown != NULL) {
        strcpy(filename_topdown, output_topdown);
    } else {
        char policy_base[256];
        strcpy(policy_base, policy_name);
        *strrchr(policy_base, '.') = '\0';

        char map[256];
        strcpy(map, basename((char*)map_name));
        *strrchr(map, '.') = '\0';

        // Create video directory if it doesn't exist
        char video_dir[512];
        sprintf(video_dir, "%s/video", policy_base);
        char mkdir_cmd[1024];
        snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p \"%s\"", video_dir);
        system(mkdir_cmd);

        sprintf(filename_topdown, "%s/video/%s_topdown.mp4", policy_base, map);
    }

    bool render_topdown = (strcmp(view_mode, "both") == 0 || strcmp(view_mode, "topdown") == 0);

    int rendered_frames = 0;
    double startTime = GetTime();

    VideoRecorder topdown_recorder;
    float reward_log[65536];
    memset(reward_log, 0, 65536 * sizeof(float));

    if (render_topdown) {
        if (!OpenVideo(&topdown_recorder, filename_topdown, img_width, img_height)) {
            CloseWindow();
            return -1;
        }

        printf("Recording topdown view...\n");
        for (int i = 0; i <= frame_count; i++) {
            if (i % frame_skip == 0) {
                renderTopDownView(&env, client, map_height, 0, 0, 0, frame_count, NULL, log_trajectories, show_grid, reward_log);
                WriteFrame(&topdown_recorder, img_width, img_height);
                rendered_frames++;
            }
            forward(net, env.observations, (int*)env.actions);
            c_step(&env);
        }
    }

    double endTime = GetTime();
    double elapsedTime = endTime - startTime;
    double writeFPS = (elapsedTime > 0) ? rendered_frames / elapsedTime : 0;

    printf("Wrote %d frames in %.2f seconds (%.2f FPS) to %s \n",
           rendered_frames, elapsedTime, writeFPS, filename_topdown);

    if (render_topdown) {
        CloseVideo(&topdown_recorder);
    }
    CloseWindow();

    // Clean up resources
    free(client);
    free_allocated(&env);
    free_drivenet(net);
    free(weights);
    return 0;
}

int main(int argc, char* argv[]) {
    int show_grid = 0;
    int obs_only = 0;
    int lasers = 0;
    int log_trajectories = 1;
    int frame_skip = 1;
    float goal_radius = 2.0f;
    int init_steps = 0;
    const char* map_name = NULL;
    const char* policy_name = "resources/drive/puffer_drive_weights.bin";
    int control_all_agents = 0;
    int deterministic_selection = 0;
    int policy_agents_per_env = -1;
    int control_non_vehicles = 0;
    int num_maps = 100;
    int scenario_length_cli = -1;

    const char* view_mode = "both";  // "both", "topdown", "agent"
    const char* output_topdown = NULL;
    const char* output_agent = NULL;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--show-grid") == 0) {
            show_grid = 1;
        } else if (strcmp(argv[i], "--obs-only") == 0) {
            obs_only = 1;
        } else if (strcmp(argv[i], "--lasers") == 0) {
            lasers = 1;
        } else if (strcmp(argv[i], "--log-trajectories") == 0) {
            log_trajectories = 1;
        } else if (strcmp(argv[i], "--frame-skip") == 0) {
            if (i + 1 < argc) {
                frame_skip = atoi(argv[i + 1]);
                i++; // Skip the next argument since we consumed it
                if (frame_skip <= 0) {
                    frame_skip = 1; // Ensure valid value
                }
            }
        } else if (strcmp(argv[i], "--goal-radius") == 0) {
            if (i + 1 < argc) {
                goal_radius = atof(argv[i + 1]);
                i++;
                if (goal_radius <= 0) {
                    goal_radius = 2.0f; // Ensure valid value
                }
            }
        } else if (strcmp(argv[i], "--map-name") == 0) {
            // Check if there's a next argument for the map path
            if (i + 1 < argc) {
                map_name = argv[i + 1];
                i++; // Skip the next argument since we used it as map path
            } else {
                fprintf(stderr, "Error: --map-name option requires a map file path\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--policy-name") == 0) {
            if (i + 1 < argc) {
                policy_name = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --policy-name option requires a policy file path\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--view") == 0) {
            if (i + 1 < argc) {
                view_mode = argv[i + 1];
                i++;
                if (strcmp(view_mode, "both") != 0 &&
                    strcmp(view_mode, "topdown") != 0 &&
                    strcmp(view_mode, "agent") != 0) {
                    fprintf(stderr, "Error: --view must be 'both', 'topdown', or 'agent'\n");
                    return 1;
                }
            } else {
                fprintf(stderr, "Error: --view option requires a value (both/topdown/agent)\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--output-topdown") == 0) {
            if (i + 1 < argc) {
                output_topdown = argv[i + 1];
                i++;
            }
        } else if (strcmp(argv[i], "--output-agent") == 0) {
            if (i + 1 < argc) {
                output_agent = argv[i + 1];
                i++;
            }
        } else if (strcmp(argv[i], "--init-steps") == 0) {
            if (i + 1 < argc) {
                init_steps = atoi(argv[i + 1]);
                i++;
                if (init_steps < 0) {
                    init_steps = 0;
                }
            }
        } else if (strcmp(argv[i], "--control-non-vehicles") == 0) {
            control_non_vehicles = 1;
        } else if (strcmp(argv[i], "--pure-self-play") == 0) {
            control_all_agents = 1;
        } else if (strcmp(argv[i], "--num-policy-controlled-agents") == 0) {
            if (i + 1 < argc) {
                policy_agents_per_env = atoi(argv[i + 1]);
                i++;
            }
        } else if (strcmp(argv[i], "--deterministic-selection") == 0) {
            deterministic_selection = 1;
        } else if (strcmp(argv[i], "--num-maps") == 0) {
            if (i + 1 < argc) {
                num_maps = atoi(argv[i + 1]);
                i++;
            }
        } else if (strcmp(argv[i], "--scenario-length") == 0) {
            if (i + 1 < argc) {
                scenario_length_cli = atoi(argv[i + 1]);
                i++;
            }
        }
    }

    eval_gif(map_name, policy_name, show_grid, obs_only, lasers, log_trajectories, frame_skip, goal_radius, control_non_vehicles, init_steps, control_all_agents, policy_agents_per_env, deterministic_selection, view_mode, output_topdown, output_agent, num_maps, scenario_length_cli);
    return 0;
}
