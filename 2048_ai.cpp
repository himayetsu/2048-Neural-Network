#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <deque>
#include <random>
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <torch/torch.h>
#include <csignal>
#include <iomanip>
#include <sstream>
#include <fstream>

volatile std::sig_atomic_t force_render_next = 0;
volatile std::sig_atomic_t should_quit = 0;

void signal_handler(int signum) {
    should_quit = 1;
}

using vec_t = std::vector<float>;

// CUDA device (will be set in main)
torch::Device device(torch::kCPU);

// Game constants
const int GRID_SIZE = 4;
const int CELL_SIZE = 100;
const int CELL_PADDING = 10;
const int WINDOW_W = GRID_SIZE * CELL_SIZE + (GRID_SIZE + 1) * CELL_PADDING;
const int WINDOW_H = WINDOW_W + 60; // Extra space for score display

// Neural network constants
const int INPUT_SIZE = 16;  // 4x4 grid flattened
const int HIDDEN_SIZE = 512;
const int HIDDEN_SIZE_2 = 256;
const int OUTPUT_SIZE = 4;  // Up, Down, Left, Right
const float LEARNING_RATE = 0.0001f;   // Faster learning
const float GAMMA = 0.95f;             // Short-term focus
const float EPSILON_START = 1.0f;
const float EPSILON_END = 0.01f;       // Keep 1% exploration
const float EPSILON_DECAY = 0.9999f;   // Slower decay
const int REPLAY_BUFFER_SIZE = 100000; // More memory
const int BATCH_SIZE = 256;            // Larger batches
const int TRAIN_EVERY = 4;
const int TARGET_UPDATE_EVERY = 1000;
const int SAVE_EVERY = 100;  // Save model every N episodes

enum class Dir { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };

// Color palette for tiles (similar to original 2048)
struct TileColor {
    int r, g, b;
};

const TileColor TILE_COLORS[] = {
    {205, 193, 180},   // 0 (empty)
    {238, 228, 218},   // 2
    {237, 224, 200},   // 4
    {242, 177, 121},   // 8
    {245, 149, 99},    // 16
    {246, 124, 95},    // 32
    {246, 94, 59},     // 64
    {237, 207, 114},   // 128
    {237, 204, 97},    // 256
    {237, 200, 80},    // 512
    {237, 197, 63},    // 1024
    {237, 194, 46},    // 2048
    {60, 58, 50},      // 4096+
};

struct Experience {
    vec_t state;
    int action;
    float reward;
    vec_t next_state;
    bool done;
};

// Neural network architecture
struct QNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
    
    QNetImpl() {
        fc1 = register_module("fc1", torch::nn::Linear(INPUT_SIZE, HIDDEN_SIZE));
        fc2 = register_module("fc2", torch::nn::Linear(HIDDEN_SIZE, HIDDEN_SIZE_2));
        fc3 = register_module("fc3", torch::nn::Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_2));
        fc4 = register_module("fc4", torch::nn::Linear(HIDDEN_SIZE_2, OUTPUT_SIZE));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::relu(fc3->forward(x));
        x = fc4->forward(x);
        return x;
    }
};
TORCH_MODULE(QNet);

class Game2048 {
public:
    Game2048(bool render, const std::string& model_path = "");
    ~Game2048();
    bool init();
    void train(int episodes);
    void save_model(const std::string& path);
    bool load_model(const std::string& path);

private:
    void reset();
    float step(int action, bool should_print);
    void render();
    void render_network(const vec_t& state);
    void spawn_tile();
    bool can_move() const;
    bool move(Dir direction);
    vec_t get_state() const;
    int select_action(const vec_t& state);
    void train_step();
    int get_tile_color_index(int value) const;
    void draw_tile(int x, int y, int value);
    void draw_text(const std::string& text, int x, int y, SDL_Color color, bool center = false);
    void draw_text_renderer(SDL_Renderer* rend, const std::string& text, int x, int y, SDL_Color color);
    std::vector<int> compress(std::vector<int>& line);
    std::vector<int> merge(std::vector<int>& line);
    bool poll_events();  // Returns false if should quit
    
    // SDL components
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Window* net_window = nullptr;
    SDL_Renderer* net_renderer = nullptr;
    TTF_Font* font = nullptr;
    TTF_Font* font_large = nullptr;
    TTF_Font* font_small = nullptr;
    
    // Game state
    int grid[GRID_SIZE][GRID_SIZE];
    int score = 0;
    int max_tile = 0;
    bool game_over = false;
    int moves_made = 0;
    int invalid_moves = 0;
    
    // Random number generator
    std::mt19937 rng;
    
    // Neural network components
    QNet q_network;
    QNet target_network;
    std::unique_ptr<torch::optim::Adam> optimizer;
    
    std::deque<Experience> replay_buffer;
    float epsilon = EPSILON_START;
    bool render_mode;
    std::string model_save_path;
    
    int episode_count = 0;
    int total_steps = 0;
    
    // For event polling timing
    std::chrono::steady_clock::time_point last_event_poll;
    
    // Training metrics
    float avg_loss = 0.0f;
    int loss_count = 0;
    int last_episode_score = 0;
};

Game2048::Game2048(bool render, const std::string& model_path)
    : render_mode(render),
      model_save_path(model_path.empty() ? "2048_model.pt" : model_path),
      q_network(QNet()),
      target_network(QNet()),
      optimizer(nullptr) {
    rng.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());
    
    q_network->to(device);
    target_network->to(device);
    
    optimizer = std::make_unique<torch::optim::Adam>(
        q_network->parameters(), torch::optim::AdamOptions(LEARNING_RATE));
    
    // Copy initial weights to target network
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < q_network->parameters().size(); i++) {
        target_network->parameters()[i].copy_(q_network->parameters()[i]);
    }
    
    last_event_poll = std::chrono::steady_clock::now();
}

Game2048::~Game2048() {
    // Auto-save on exit
    save_model(model_save_path);
    std::cout << "\nModel saved to " << model_save_path << "\n";
    
    if (font_small) TTF_CloseFont(font_small);
    if (font) TTF_CloseFont(font);
    if (font_large) TTF_CloseFont(font_large);
    if (net_renderer) SDL_DestroyRenderer(net_renderer);
    if (net_window) SDL_DestroyWindow(net_window);
    if (renderer) SDL_DestroyRenderer(renderer);
    if (window) SDL_DestroyWindow(window);
    if (render_mode) {
        TTF_Quit();
        SDL_Quit();
    }
}

void Game2048::save_model(const std::string& path) {
    try {
        torch::serialize::OutputArchive archive;
        q_network->save(archive);
        archive.save_to(path);
        
        // Also save training state
        std::ofstream state_file(path + ".state");
        if (state_file.is_open()) {
            state_file << epsilon << "\n";
            state_file << total_steps << "\n";
            state_file << episode_count << "\n";
            state_file.close();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << "\n";
    }
}

bool Game2048::load_model(const std::string& path) {
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        q_network->load(archive);
        
        // Move networks to device (GPU if available)
        q_network->to(device);
        target_network->to(device);
        
        // Copy to target network
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < q_network->parameters().size(); i++) {
            target_network->parameters()[i].copy_(q_network->parameters()[i]);
        }
        
        // Load training state if exists
        std::ifstream state_file(path + ".state");
        if (state_file.is_open()) {
            state_file >> epsilon;
            state_file >> total_steps;
            state_file >> episode_count;
            state_file.close();
            std::cout << "Loaded training state: eps=" << epsilon 
                      << ", steps=" << total_steps 
                      << ", episodes=" << episode_count << "\n";
        }
        
        std::cout << "Loaded model from " << path << "\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "No existing model found at " << path << ", starting fresh.\n";
        return false;
    }
}

bool Game2048::poll_events() {
    if (!render_mode) return true;
    
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) {
            return false;
        }
        if (e.type == SDL_KEYDOWN) {
            if (e.key.keysym.sym == SDLK_ESCAPE) {
                return false;
            }
            if (e.key.keysym.sym == SDLK_s) {
                save_model(model_save_path);
                std::cout << "Model saved manually.\n";
            }
        }
    }
    return !should_quit;
}

bool Game2048::init() {
    if (render_mode) {
        if (SDL_Init(SDL_INIT_VIDEO) != 0) {
            std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
            return false;
        }
        
        if (TTF_Init() != 0) {
            std::cerr << "TTF_Init failed: " << TTF_GetError() << "\n";
            return false;
        }
        
        // Create game window
        window = SDL_CreateWindow("2048 AI Training",
                                  SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  WINDOW_W, WINDOW_H, SDL_WINDOW_SHOWN);
        // Disable VSync for maximum speed (SDL_RENDERER_PRESENTVSYNC removed)
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        SDL_RenderSetVSync(renderer, 0);  // Disable vsync
        
        // Create single unified network visualization window
        net_window = SDL_CreateWindow("Neural Network - Live Activity (Press S to Save)",
                                      SDL_WINDOWPOS_CENTERED + WINDOW_W + 50,
                                      SDL_WINDOWPOS_CENTERED,
                                      900, 700, SDL_WINDOW_SHOWN);
        net_renderer = SDL_CreateRenderer(net_window, -1, SDL_RENDERER_ACCELERATED);
        SDL_RenderSetVSync(net_renderer, 0);  // Disable vsync
        
        // Load font (fallback to different paths)
        const char* font_paths[] = {
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"
        };
        
        for (const char* path : font_paths) {
            font = TTF_OpenFont(path, 24);
            if (font) {
                font_large = TTF_OpenFont(path, 36);
                font_small = TTF_OpenFont(path, 14);
                break;
            }
        }
        
        if (!font) {
            std::cerr << "Warning: Could not load font, text will not be displayed\n";
        }
    }
    
    // Try to load existing model
    load_model(model_save_path);
    
    reset();
    return true;
}

void Game2048::reset() {
    // Clear the grid
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            grid[i][j] = 0;
        }
    }
    
    score = 0;
    max_tile = 0;
    game_over = false;
    moves_made = 0;
    invalid_moves = 0;
    
    // Spawn two initial tiles
    spawn_tile();
    spawn_tile();
}

void Game2048::spawn_tile() {
    std::vector<std::pair<int, int>> empty_cells;
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            if (grid[i][j] == 0) {
                empty_cells.push_back({i, j});
            }
        }
    }
    
    if (empty_cells.empty()) return;
    
    std::uniform_int_distribution<int> dist(0, (int)empty_cells.size() - 1);
    auto& cell = empty_cells[dist(rng)];
    
    // 90% chance of 2, 10% chance of 4
    std::uniform_real_distribution<float> val_dist(0, 1);
    grid[cell.first][cell.second] = (val_dist(rng) < 0.9f) ? 2 : 4;
}

std::vector<int> Game2048::compress(std::vector<int>& line) {
    std::vector<int> result;
    for (int val : line) {
        if (val != 0) {
            result.push_back(val);
        }
    }
    while (result.size() < GRID_SIZE) {
        result.push_back(0);
    }
    return result;
}

std::vector<int> Game2048::merge(std::vector<int>& line) {
    for (int i = 0; i < GRID_SIZE - 1; i++) {
        if (line[i] != 0 && line[i] == line[i + 1]) {
            line[i] *= 2;
            line[i + 1] = 0;
            score += line[i];
            if (line[i] > max_tile) max_tile = line[i];
        }
    }
    return compress(line);
}

bool Game2048::move(Dir direction) {
    int old_grid[GRID_SIZE][GRID_SIZE];
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            old_grid[i][j] = grid[i][j];
        }
    }
    
    bool moved = false;
    
    switch (direction) {
        case Dir::LEFT:
            for (int i = 0; i < GRID_SIZE; i++) {
                std::vector<int> row(grid[i], grid[i] + GRID_SIZE);
                row = compress(row);
                row = merge(row);
                for (int j = 0; j < GRID_SIZE; j++) {
                    grid[i][j] = row[j];
                }
            }
            break;
            
        case Dir::RIGHT:
            for (int i = 0; i < GRID_SIZE; i++) {
                std::vector<int> row(grid[i], grid[i] + GRID_SIZE);
                std::reverse(row.begin(), row.end());
                row = compress(row);
                row = merge(row);
                std::reverse(row.begin(), row.end());
                for (int j = 0; j < GRID_SIZE; j++) {
                    grid[i][j] = row[j];
                }
            }
            break;
            
        case Dir::UP:
            for (int j = 0; j < GRID_SIZE; j++) {
                std::vector<int> col;
                for (int i = 0; i < GRID_SIZE; i++) {
                    col.push_back(grid[i][j]);
                }
                col = compress(col);
                col = merge(col);
                for (int i = 0; i < GRID_SIZE; i++) {
                    grid[i][j] = col[i];
                }
            }
            break;
            
        case Dir::DOWN:
            for (int j = 0; j < GRID_SIZE; j++) {
                std::vector<int> col;
                for (int i = 0; i < GRID_SIZE; i++) {
                    col.push_back(grid[i][j]);
                }
                std::reverse(col.begin(), col.end());
                col = compress(col);
                col = merge(col);
                std::reverse(col.begin(), col.end());
                for (int i = 0; i < GRID_SIZE; i++) {
                    grid[i][j] = col[i];
                }
            }
            break;
    }
    
    // Check if anything changed
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            if (grid[i][j] != old_grid[i][j]) {
                moved = true;
                break;
            }
        }
        if (moved) break;
    }
    
    return moved;
}

bool Game2048::can_move() const {
    // Check for empty cells
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            if (grid[i][j] == 0) return true;
        }
    }
    
    // Check for possible merges
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            if (i < GRID_SIZE - 1 && grid[i][j] == grid[i + 1][j]) return true;
            if (j < GRID_SIZE - 1 && grid[i][j] == grid[i][j + 1]) return true;
        }
    }
    
    return false;
}

vec_t Game2048::get_state() const {
    vec_t state(INPUT_SIZE, 0.0f);
    
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            int idx = i * GRID_SIZE + j;
            if (grid[i][j] == 0) {
                state[idx] = 0.0f;
            } else {
                // Normalize using log2 (2^17 = 131072 theoretical max)
                state[idx] = std::log2((float)grid[i][j]) / 17.0f;
            }
        }
    }
    
    return state;
}

int Game2048::select_action(const vec_t& state) {
    std::uniform_real_distribution<float> dist(0, 1);
    
    if (dist(rng) < epsilon) {
        // Random action
        std::uniform_int_distribution<int> action_dist(0, 3);
        return action_dist(rng);
    } else {
        // Greedy action from Q-network
        torch::NoGradGuard no_grad;
        auto state_tensor = torch::tensor(state).reshape({1, INPUT_SIZE}).to(device);
        auto q_values = q_network->forward(state_tensor);
        return q_values.argmax(1).item<int>();
    }
}

float Game2048::step(int action, bool should_print) {
    (void)should_print;
    int old_score = score;
    int old_max = max_tile;
    
    Dir dir = static_cast<Dir>(action);
    bool valid_move = move(dir);
    
    float reward = 0.0f;
    
    if (valid_move) {
        moves_made++;
        spawn_tile();
        
        // Reward merges (score increase)
        int score_gained = score - old_score;
        reward += (float)score_gained / 50.0f;
        
        // Big bonus for new max tile
        if (max_tile > old_max) {
            reward += std::log2((float)max_tile);
        }
        
        // Reward keeping max tile in corner (key 2048 strategy)
        int corners[] = {grid[0][0], grid[0][3], grid[3][0], grid[3][3]};
        for (int c : corners) {
            if (c == max_tile) {
                reward += 0.5f;
                break;
            }
        }
        
        // Reward monotonic rows/columns (tiles in order = good)
        auto is_monotonic = [](int a, int b, int c, int d) {
            return (a >= b && b >= c && c >= d) || (a <= b && b <= c && c <= d);
        };
        for (int i = 0; i < 4; i++) {
            if (is_monotonic(grid[i][0], grid[i][1], grid[i][2], grid[i][3])) reward += 0.1f;
            if (is_monotonic(grid[0][i], grid[1][i], grid[2][i], grid[3][i])) reward += 0.1f;
        }
        
        // Count empty cells - more empty = more options
        int empty = 0;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                if (grid[i][j] == 0) empty++;
        reward += empty * 0.1f;
        
    } else {
        invalid_moves++;
        reward = -1.0f;  // Harsh penalty for invalid moves
    }
    
    if (!can_move()) {
        game_over = true;
        reward -= 2.0f;
        reward += std::log2((float)max_tile) * 0.5f;  // Consolation for high tiles
    }
    
    return reward;
}

int Game2048::get_tile_color_index(int value) const {
    if (value == 0) return 0;
    int idx = (int)std::log2((float)value);
    return std::min(idx, 12);
}

void Game2048::draw_tile(int x, int y, int value) {
    int px = CELL_PADDING + x * (CELL_SIZE + CELL_PADDING);
    int py = CELL_PADDING + y * (CELL_SIZE + CELL_PADDING) + 60;
    
    const TileColor& color = TILE_COLORS[get_tile_color_index(value)];
    SDL_SetRenderDrawColor(renderer, (Uint8)color.r, (Uint8)color.g, (Uint8)color.b, 255);
    SDL_Rect rect = {px, py, CELL_SIZE, CELL_SIZE};
    SDL_RenderFillRect(renderer, &rect);
    
    if (value > 0 && font) {
        std::string text = std::to_string(value);
        SDL_Color text_color = (value <= 4) ? SDL_Color{119, 110, 101, 255} : SDL_Color{255, 255, 255, 255};
        
        TTF_Font* use_font = (value >= 1000) ? font : font_large;
        if (!use_font) use_font = font;
        
        SDL_Surface* surface = TTF_RenderText_Blended(use_font, text.c_str(), text_color);
        if (surface) {
            SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
            SDL_Rect text_rect = {px + (CELL_SIZE - surface->w) / 2, 
                                  py + (CELL_SIZE - surface->h) / 2,
                                  surface->w, surface->h};
            SDL_RenderCopy(renderer, texture, nullptr, &text_rect);
            SDL_DestroyTexture(texture);
            SDL_FreeSurface(surface);
        }
    }
}

void Game2048::draw_text(const std::string& text, int x, int y, SDL_Color color, bool center) {
    if (!font) return;
    
    SDL_Surface* surface = TTF_RenderText_Blended(font, text.c_str(), color);
    if (surface) {
        SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
        SDL_Rect rect = {x, y, surface->w, surface->h};
        if (center) {
            rect.x -= surface->w / 2;
        }
        SDL_RenderCopy(renderer, texture, nullptr, &rect);
        SDL_DestroyTexture(texture);
        SDL_FreeSurface(surface);
    }
}

void Game2048::draw_text_renderer(SDL_Renderer* rend, const std::string& text, int x, int y, SDL_Color color) {
    if (!font_small) return;
    
    SDL_Surface* surface = TTF_RenderText_Blended(font_small, text.c_str(), color);
    if (surface) {
        SDL_Texture* texture = SDL_CreateTextureFromSurface(rend, surface);
        SDL_Rect rect = {x, y, surface->w, surface->h};
        SDL_RenderCopy(rend, texture, nullptr, &rect);
        SDL_DestroyTexture(texture);
        SDL_FreeSurface(surface);
    }
}

void Game2048::render() {
    if (!render_mode) return;
    
    // Background
    SDL_SetRenderDrawColor(renderer, 187, 173, 160, 255);
    SDL_RenderClear(renderer);
    
    // Score display
    std::stringstream ss;
    ss << "Score: " << score << "  Max: " << max_tile << "  Episode: " << episode_count;
    draw_text(ss.str(), 10, 15, {119, 110, 101, 255});
    
    // Draw grid background
    SDL_SetRenderDrawColor(renderer, 205, 193, 180, 255);
    SDL_Rect grid_bg = {CELL_PADDING / 2, 60 + CELL_PADDING / 2, 
                        WINDOW_W - CELL_PADDING, WINDOW_W - CELL_PADDING};
    SDL_RenderFillRect(renderer, &grid_bg);
    
    // Draw tiles
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            draw_tile(j, i, grid[i][j]);
        }
    }
    
    SDL_RenderPresent(renderer);
}

void Game2048::render_network(const vec_t& state) {
    if (!render_mode) return;
    
    // Dark background
    SDL_SetRenderDrawColor(net_renderer, 15, 15, 25, 255);
    SDL_RenderClear(net_renderer);
    
    const int win_w = 900;
    const int win_h = 700;
    const int padding_x = 80;
    const int padding_y = 60;
    const int viz_height = 500;
    
    const int display_sizes[] = {16, 32, 24, 24, 4};
    const int actual_sizes[] = {INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE_2, HIDDEN_SIZE_2, OUTPUT_SIZE};
    const int num_layers = 5;
    const char* layer_names[] = {"Input", "Hidden 1", "Hidden 2", "Hidden 3", "Output"};
    const char* action_names[] = {"UP", "DOWN", "LEFT", "RIGHT"};
    
    std::vector<std::vector<SDL_Point>> neuron_pos(num_layers);
    
    for (int l = 0; l < num_layers; l++) {
        int n = display_sizes[l];
        neuron_pos[l].resize(n);
        for (int i = 0; i < n; i++) {
            neuron_pos[l][i] = {
                padding_x + l * (win_w - 2 * padding_x) / (num_layers - 1),
                padding_y + 30 + i * (viz_height - 60) / std::max(1, n - 1)
            };
        }
    }
    
    // Forward pass to get activations (on GPU, then move to CPU for visualization)
    torch::NoGradGuard no_grad;
    torch::Tensor inp = torch::tensor(state).reshape({1, INPUT_SIZE}).to(device);
    torch::Tensor h1 = torch::relu(q_network->fc1->forward(inp));
    torch::Tensor h2 = torch::relu(q_network->fc2->forward(h1));
    torch::Tensor h3 = torch::relu(q_network->fc3->forward(h2));
    torch::Tensor out = q_network->fc4->forward(h3);
    torch::Tensor out_softmax = torch::softmax(out, 1);
    
    // Move to CPU for visualization
    std::vector<torch::Tensor> activations = {
        inp.flatten().cpu(), h1.flatten().cpu(), h2.flatten().cpu(), h3.flatten().cpu(), out_softmax.flatten().cpu()
    };
    out = out.cpu();
    out_softmax = out_softmax.cpu();
    
    // Draw connections with activation-weighted colors
    auto draw_connections = [&](torch::nn::Linear layer, int l, int sample_in, int sample_out) {
        auto w = layer->weight.detach().cpu();  // Move weights to CPU for visualization
        
        for (int i = 0; i < sample_out && i < (int)neuron_pos[l + 1].size(); i++) {
            for (int j = 0; j < sample_in && j < (int)neuron_pos[l].size(); j++) {
                int actual_i = i * actual_sizes[l + 1] / sample_out;
                int actual_j = j * actual_sizes[l] / sample_in;
                
                float weight = w[actual_i][actual_j].item<float>();
                float activation = activations[l][actual_j].item<float>();
                float signal = weight * activation;
                
                float intensity = std::tanh(std::abs(signal)) * 255.0f;
                Uint8 r = signal > 0 ? (Uint8)intensity : 0;
                Uint8 g = signal < 0 ? (Uint8)intensity : 0;
                Uint8 b = (Uint8)(intensity * 0.3f);
                
                SDL_SetRenderDrawColor(net_renderer, r, g, b, 180);
                SDL_RenderDrawLine(net_renderer,
                                   neuron_pos[l][j].x, neuron_pos[l][j].y,
                                   neuron_pos[l + 1][i].x, neuron_pos[l + 1][i].y);
            }
        }
    };
    
    draw_connections(q_network->fc1, 0, 16, 32);
    draw_connections(q_network->fc2, 1, 32, 24);
    draw_connections(q_network->fc3, 2, 24, 24);
    draw_connections(q_network->fc4, 3, 24, 4);
    
    // Draw neurons with activation intensity
    for (int l = 0; l < num_layers; l++) {
        for (int i = 0; i < (int)neuron_pos[l].size(); i++) {
            int actual_i = i * actual_sizes[l] / display_sizes[l];
            float act = activations[l][actual_i].item<float>();
            
            int intensity;
            if (l == num_layers - 1) {
                intensity = std::min(255, (int)(act * 255));
            } else {
                intensity = std::min(255, std::max(0, (int)(act * 300)));
            }
            
            // Size based on activation
            int size = 4 + (int)(std::min(act, 1.0f) * 6);
            SDL_Rect r = {neuron_pos[l][i].x - size, neuron_pos[l][i].y - size, size * 2, size * 2};
            
            if (l == 0) {
                // Input layer - cyan
                SDL_SetRenderDrawColor(net_renderer, 50, (Uint8)intensity, 255, 255);
            } else if (l == num_layers - 1) {
                // Output layer - orange/yellow
                SDL_SetRenderDrawColor(net_renderer, 255, 200, (Uint8)intensity, 255);
            } else {
                // Hidden layers - purple/blue gradient
                SDL_SetRenderDrawColor(net_renderer, (Uint8)(intensity * 0.7), (Uint8)intensity, 255, 255);
            }
            
            SDL_RenderFillRect(net_renderer, &r);
        }
    }
    
    // Draw layer labels
    SDL_Color label_color = {150, 150, 180, 255};
    for (int l = 0; l < num_layers; l++) {
        draw_text_renderer(net_renderer, layer_names[l], 
                          neuron_pos[l][0].x - 25, padding_y - 5, label_color);
    }
    
    // Draw action probabilities and labels
    int best_action = out.argmax(1).item<int>();
    for (int i = 0; i < 4; i++) {
        float q_val = out[0][i].item<float>();
        float prob = out_softmax[0][i].item<float>();
        
        int bar_x = neuron_pos[4][i].x + 20;
        int bar_y = neuron_pos[4][i].y - 8;
        int bar_w = (int)(prob * 100);
        
        // Highlight best action
        if (i == best_action) {
            SDL_SetRenderDrawColor(net_renderer, 50, 255, 100, 255);
        } else {
            SDL_SetRenderDrawColor(net_renderer, 100, 100, 150, 255);
        }
        SDL_Rect bar = {bar_x, bar_y, bar_w, 16};
        SDL_RenderFillRect(net_renderer, &bar);
        
        // Action label
        std::stringstream ss;
        ss << action_names[i] << " " << std::fixed << std::setprecision(2) << q_val;
        SDL_Color action_color = (i == best_action) ? SDL_Color{50, 255, 100, 255} : SDL_Color{150, 150, 180, 255};
        draw_text_renderer(net_renderer, ss.str(), bar_x + 105, bar_y, action_color);
    }
    
    // Draw stats at bottom
    int stats_y = viz_height + 50;
    SDL_Color stats_color = {200, 200, 220, 255};
    
    std::stringstream ss;
    ss << "Episode: " << episode_count << "  |  Total Steps: " << total_steps 
       << "  |  Epsilon: " << std::fixed << std::setprecision(4) << epsilon
       << "  |  Buffer: " << replay_buffer.size();
    draw_text_renderer(net_renderer, ss.str(), 20, stats_y, stats_color);
    
    ss.str("");
    ss << "Press S to manually save  |  Press ESC to quit  |  Model: " << model_save_path;
    draw_text_renderer(net_renderer, ss.str(), 20, stats_y + 25, {120, 120, 150, 255});
    
    // Draw explanation
    draw_text_renderer(net_renderer, "Red = positive signal  |  Green = negative signal  |  Brightness = strength", 
                      20, stats_y + 50, {100, 100, 130, 255});
    
    SDL_RenderPresent(net_renderer);
}

void Game2048::train_step() {
    if (replay_buffer.size() < BATCH_SIZE) return;
    
    // Sample random batch
    std::vector<int> indices(replay_buffer.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    std::vector<vec_t> states, next_states;
    std::vector<int> actions;
    std::vector<float> rewards;
    std::vector<float> dones;
    
    for (int i = 0; i < BATCH_SIZE; i++) {
        const auto& exp = replay_buffer[indices[i]];
        states.push_back(exp.state);
        actions.push_back(exp.action);
        rewards.push_back(exp.reward);
        next_states.push_back(exp.next_state);
        dones.push_back(exp.done ? 1.0f : 0.0f);
    }
    
    // Convert to tensors (build on CPU first, then move to GPU in one batch)
    auto state_tensor = torch::zeros({BATCH_SIZE, INPUT_SIZE});
    auto next_state_tensor = torch::zeros({BATCH_SIZE, INPUT_SIZE});
    auto action_tensor = torch::zeros({BATCH_SIZE}, torch::kLong);
    auto reward_tensor = torch::zeros({BATCH_SIZE});
    auto done_tensor = torch::zeros({BATCH_SIZE});
    
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            state_tensor[i][j] = states[i][j];
            next_state_tensor[i][j] = next_states[i][j];
        }
        action_tensor[i] = actions[i];
        reward_tensor[i] = rewards[i];
        done_tensor[i] = dones[i];
    }
    
    // Move tensors to GPU
    state_tensor = state_tensor.to(device);
    next_state_tensor = next_state_tensor.to(device);
    action_tensor = action_tensor.to(device);
    reward_tensor = reward_tensor.to(device);
    done_tensor = done_tensor.to(device);
    
    // Compute current Q values
    auto q_values = q_network->forward(state_tensor);
    auto q_values_selected = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1);
    
    // Compute target Q values using TARGET network (this is why we have 2 networks!)
    torch::Tensor next_q_values;
    {
        torch::NoGradGuard no_grad;
        next_q_values = target_network->forward(next_state_tensor);
    }
    auto max_next_q = std::get<0>(next_q_values.max(1));
    auto target_q = reward_tensor + GAMMA * max_next_q * (1 - done_tensor);
    
    // Compute loss and update
    auto loss = torch::mse_loss(q_values_selected, target_q.detach());
    
    // Track loss for display
    float loss_val = loss.item<float>();
    avg_loss = (avg_loss * loss_count + loss_val) / (loss_count + 1);
    loss_count++;
    
    optimizer->zero_grad();
    loss.backward();
    
    // Gradient clipping
    torch::nn::utils::clip_grad_norm_(q_network->parameters(), 1.0);
    
    optimizer->step();
    
    // Update target network periodically (copy Q-network weights to target)
    if (total_steps % TARGET_UPDATE_EVERY == 0) {
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < q_network->parameters().size(); i++) {
            target_network->parameters()[i].copy_(q_network->parameters()[i]);
        }
    }
}

void Game2048::train(int episodes) {
    int total_score = 0;
    int max_score = 0;
    int max_tile_ever = 0;
    int games_reaching_2048 = 0;
    int start_episode = episode_count;
    
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    2048 AI Neural Network Training               ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Network: " << INPUT_SIZE << " -> " << HIDDEN_SIZE << " -> " 
              << HIDDEN_SIZE_2 << " -> " << HIDDEN_SIZE_2 << " -> " << OUTPUT_SIZE 
              << std::setw(22) << "║\n";
    std::cout << "║ Learning Rate: " << LEARNING_RATE << "  Gamma: " << GAMMA 
              << "  Batch: " << BATCH_SIZE << std::setw(17) << "║\n";
    std::cout << "║ Model saves to: " << model_save_path << std::setw(33 - (int)model_save_path.length()) << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";
    
    auto last_event_time = std::chrono::steady_clock::now();
    
    for (int ep = start_episode; ep < start_episode + episodes; ep++) {
        episode_count = ep;
        
        reset();
        auto state = get_state();
        int ep_steps = 0;
        
        while (!game_over && ep_steps < 5000) {
            // Poll events less frequently for speed (every 200ms instead of 50ms)
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_event_time).count() > 200) {
                if (!poll_events()) {
                    std::cout << "\nQuitting...\n";
                    return;
                }
                last_event_time = now;
            }
            
            int action = select_action(state);
            float reward = step(action, false);
            auto next_state = get_state();
            
            replay_buffer.push_back({state, action, reward, next_state, game_over});
            if (replay_buffer.size() > REPLAY_BUFFER_SIZE) {
                replay_buffer.pop_front();
            }
            
            if (total_steps % TRAIN_EVERY == 0 && replay_buffer.size() >= BATCH_SIZE) {
                train_step();
            }
            
            state = next_state;
            ep_steps++;
            total_steps++;
            
            // Render every step - no sleep, no freeze!
            if (render_mode && ep_steps % 2 == 0) {  // Every 2nd step to reduce load
                render();
                render_network(state);
            }
        }
        
        // Store this episode's score
        last_episode_score = score;
        total_score += score;
        max_score = std::max(max_score, score);
        if (max_tile > max_tile_ever) max_tile_ever = max_tile;
        if (max_tile >= 2048) games_reaching_2048++;
        
        epsilon = std::max(EPSILON_END, epsilon * EPSILON_DECAY);
        
        // Auto-save periodically
        if (ep % SAVE_EVERY == 0 && ep > start_episode) {
            save_model(model_save_path);
        }
        
        // Print progress every episode
        std::cout << "Ep " << std::setw(5) << ep
                  << " | Score: " << std::setw(5) << score
                  << " | MaxTile: " << std::setw(4) << max_tile
                  << " | Best: " << std::setw(5) << max_score
                  << " | MaxTileEver: " << std::setw(4) << max_tile_ever
                  << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << " | Eps: " << std::setprecision(3) << epsilon
                  << "\n";
        
        // Reset loss tracking every 10 episodes
        if (ep % 10 == 0) {
            avg_loss = 0.0f;
            loss_count = 0;
        }
    }
    
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                         Training Complete!                        ║\n";
    std::cout << "║ Best Score: " << std::setw(6) << max_score 
              << "  |  Highest Tile: " << std::setw(5) << max_tile_ever 
              << "  |  2048 Games: " << std::setw(4) << games_reaching_2048 << " ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
}

int main(int argc, char** argv) {
    SDL_SetMainReady();
    std::signal(SIGINT, signal_handler);
    
    bool render = true;
    int episodes = 10000;
    std::string model_path = "2048_model.pt";
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--no-render") {
            render = false;
        } else if (arg == "--episodes" && i + 1 < argc) {
            episodes = std::stoi(argv[++i]);
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--help") {
            std::cout << "2048 AI Neural Network\n";
            std::cout << "Usage: 2048_ai [options]\n";
            std::cout << "  --episodes N   Train for N episodes (default: 10000)\n";
            std::cout << "  --model PATH   Model file path (default: 2048_model.pt)\n";
            std::cout << "  --no-render    Disable graphics (faster training)\n";
            std::cout << "  --help         Show this help\n";
            std::cout << "\nDuring training:\n";
            std::cout << "  Press S        Save model manually\n";
            std::cout << "  Press ESC      Quit (auto-saves)\n";
            return 0;
        }
    }
    
    std::cout << "Initializing 2048 AI...\n";
    
    // Check for CUDA GPU
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "CUDA GPU detected! Using GPU for training.\n";
        std::cout << "GPU: " << torch::cuda::device_count() << " device(s) available\n";
    } else {
        std::cout << "No CUDA GPU detected. Using CPU.\n";
    }
    
    Game2048 game(render, model_path);
    if (!game.init()) {
        std::cerr << "Failed to initialize game!\n";
        return 1;
    }
    
    std::cout << "Starting training for " << episodes << " episodes...\n\n";
    game.train(episodes);
    
    return 0;
}
