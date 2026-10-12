#ifndef AI2048_TRAINING_VISUALIZATION_CPP
#define AI2048_TRAINING_VISUALIZATION_CPP

// Declarations and implementation are kept together; include this component once.

#include "training.cpp"

#include <memory>

namespace ai2048 {

struct UiEvents {
    bool quit = false;
    bool save = false;
};

class TrainingVisualization {
public:
    TrainingVisualization();
    ~TrainingVisualization();
    TrainingVisualization(const TrainingVisualization&) = delete;
    TrainingVisualization& operator=(const TrainingVisualization&) = delete;

    bool init();
    UiEvents poll_events();
    void render(const GameEngine& game, const TrainingStatus& status,
                const NetworkSnapshot& network);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ai2048

#define SDL_MAIN_HANDLED

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace ai2048 {
namespace {

constexpr int kCellSize = 100;
constexpr int kCellPadding = 10;
constexpr int kGameWidth = 4 * kCellSize + 5 * kCellPadding;
constexpr int kGameHeight = kGameWidth + 60;
constexpr int kNetworkWidth = 1060;
constexpr int kNetworkHeight = 740;
constexpr std::array<int, 5> kDisplaySizes{16, 32, 24, 24, 4};
constexpr std::array<const char*, 5> kLayerNames{
    "Input", "Hidden 1", "Hidden 2", "Hidden 3", "Q values"};
constexpr std::array<const char*, 4> kActionNames{"UP", "DOWN", "LEFT", "RIGHT"};
constexpr std::array<SDL_Color, 13> kTileColors{{
    {205, 193, 180, 255}, {238, 228, 218, 255}, {237, 224, 200, 255},
    {242, 177, 121, 255}, {245, 149, 99, 255}, {246, 124, 95, 255},
    {246, 94, 59, 255}, {237, 207, 114, 255}, {237, 204, 97, 255},
    {237, 200, 80, 255}, {237, 197, 63, 255}, {237, 194, 46, 255},
    {60, 58, 50, 255},
}};

void set_color(SDL_Renderer* renderer, SDL_Color color) {
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
}

float finite_or_zero(float value) {
    return std::isfinite(value) ? value : 0.0f;
}

float activation(const NetworkSnapshot& snapshot, int layer, int index) {
    if (layer < 0 || layer >= static_cast<int>(snapshot.activations.size()) ||
        index < 0 || index >= static_cast<int>(snapshot.activations[layer].size())) {
        return 0.0f;
    }
    return finite_or_zero(snapshot.activations[layer][index]);
}

}  // namespace

struct TrainingVisualization::Impl {
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Window* network_window = nullptr;
    SDL_Renderer* network_renderer = nullptr;
    TTF_Font* font = nullptr;
    TTF_Font* font_large = nullptr;
    TTF_Font* font_small = nullptr;
    bool sdl_initialized = false;
    bool ttf_initialized = false;

    ~Impl() { cleanup(); }

    void cleanup() {
        if (font_small) TTF_CloseFont(font_small);
        if (font_large) TTF_CloseFont(font_large);
        if (font) TTF_CloseFont(font);
        if (network_renderer) SDL_DestroyRenderer(network_renderer);
        if (network_window) SDL_DestroyWindow(network_window);
        if (renderer) SDL_DestroyRenderer(renderer);
        if (window) SDL_DestroyWindow(window);
        font_small = font_large = font = nullptr;
        network_renderer = renderer = nullptr;
        network_window = window = nullptr;
        if (ttf_initialized) TTF_Quit();
        if (sdl_initialized) SDL_QuitSubSystem(SDL_INIT_VIDEO);
        ttf_initialized = sdl_initialized = false;
    }

    void text(SDL_Renderer* target, TTF_Font* text_font, const std::string& value,
              int x, int y, SDL_Color color, bool center = false) {
        if (!text_font || value.empty()) return;
        SDL_Surface* surface = TTF_RenderUTF8_Blended(text_font, value.c_str(), color);
        if (!surface) return;
        SDL_Texture* texture = SDL_CreateTextureFromSurface(target, surface);
        if (texture) {
            SDL_Rect rect{x - (center ? surface->w / 2 : 0), y, surface->w, surface->h};
            SDL_RenderCopy(target, texture, nullptr, &rect);
            SDL_DestroyTexture(texture);
        }
        SDL_FreeSurface(surface);
    }

    // Limit long model paths to the available width instead of clipping them.
    void fitted_text(SDL_Renderer* target, const std::string& value, int x, int y,
                     int max_width, SDL_Color color) {
        if (!font_small) return;
        std::string shown = value;
        int width = 0;
        if (TTF_SizeUTF8(font_small, shown.c_str(), &width, nullptr) != 0) return;
        if (width > max_width) {
            while (!shown.empty()) {
                shown.pop_back();
                const std::string candidate = shown + "...";
                if (TTF_SizeUTF8(font_small, candidate.c_str(), &width, nullptr) == 0 &&
                    width <= max_width) {
                    shown = candidate;
                    break;
                }
            }
        }
        text(target, font_small, shown, x, y, color);
    }

    void draw_tile(int x, int y, int value) {
        const int px = kCellPadding + x * (kCellSize + kCellPadding);
        const int py = 60 + kCellPadding + y * (kCellSize + kCellPadding);
        const int color_index = value == 0 ? 0 :
            std::clamp(static_cast<int>(std::log2(static_cast<float>(value))), 0, 12);
        set_color(renderer, kTileColors[color_index]);
        SDL_Rect rect{px, py, kCellSize, kCellSize};
        SDL_RenderFillRect(renderer, &rect);
        if (value == 0 || !font) return;

        TTF_Font* tile_font = value >= 1000 || !font_large ? font : font_large;
        const SDL_Color color = value <= 4 ? SDL_Color{119, 110, 101, 255} :
                                                         SDL_Color{255, 255, 255, 255};
        text(renderer, tile_font, std::to_string(value), px + kCellSize / 2,
             py + (kCellSize - TTF_FontHeight(tile_font)) / 2, color, true);
    }

    void render_game(const GameEngine& game, const TrainingStatus& status) {
        set_color(renderer, {187, 173, 160, 255});
        SDL_RenderClear(renderer);
        std::ostringstream score;
        score << "Score: " << game.score() << "   Max: " << game.max_tile();
        text(renderer, font, score.str(), 10, 6, {119, 110, 101, 255});
        text(renderer, font_small,
             std::string(status.evaluation ? "Evaluation" : "Training") +
                 "   Episode: " + std::to_string(status.episodes),
             10, 38, {119, 110, 101, 255});
        set_color(renderer, {205, 193, 180, 255});
        SDL_Rect background{kCellPadding / 2, 60 + kCellPadding / 2,
                            kGameWidth - kCellPadding, kGameWidth - kCellPadding};
        SDL_RenderFillRect(renderer, &background);
        const auto& board = game.board();
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) draw_tile(col, row, board[row][col]);
        }
        SDL_RenderPresent(renderer);
    }

    void render_network(const TrainingStatus& status, const NetworkSnapshot& snapshot) {
        set_color(network_renderer, {15, 15, 25, 255});
        SDL_RenderClear(network_renderer);

        std::array<std::vector<SDL_Point>, 5> positions;
        for (int layer = 0; layer < 5; ++layer) {
            const int count = std::min(kDisplaySizes[layer], snapshot.sizes[layer]);
            for (int index = 0; index < count; ++index) {
                positions[layer].push_back({60 + layer * 160,
                    90 + index * 425 / std::max(1, count - 1)});
            }
        }

        // Sample the same neurons across every connection and activation layer.
        SDL_SetRenderDrawBlendMode(network_renderer, SDL_BLENDMODE_BLEND);
        for (int layer = 0; layer < 4 && layer < static_cast<int>(snapshot.weights.size()); ++layer) {
            const auto& weights = snapshot.weights[layer];
            const int inputs = static_cast<int>(positions[layer].size());
            const int outputs = static_cast<int>(positions[layer + 1].size());
            for (int out = 0; out < outputs; ++out) {
                const int actual_out = out * snapshot.sizes[layer + 1] / outputs;
                for (int in = 0; in < inputs; ++in) {
                    const int actual_in = in * snapshot.sizes[layer] / inputs;
                    const std::size_t offset = static_cast<std::size_t>(actual_out) *
                                              snapshot.sizes[layer] + actual_in;
                    if (offset >= weights.size()) continue;
                    const float signal = finite_or_zero(weights[offset]) *
                                         activation(snapshot, layer, actual_in);
                    const Uint8 intensity = static_cast<Uint8>(
                        std::tanh(std::abs(finite_or_zero(signal))) * 255.0f);
                    set_color(network_renderer, {static_cast<Uint8>(signal > 0 ? intensity : 0),
                                                 static_cast<Uint8>(signal < 0 ? intensity : 0),
                                                 static_cast<Uint8>(intensity * 0.3f), 180});
                    SDL_RenderDrawLine(network_renderer, positions[layer][in].x,
                        positions[layer][in].y, positions[layer + 1][out].x,
                        positions[layer + 1][out].y);
                }
            }
        }

        int greedy_action = -1;
        float best_q = -std::numeric_limits<float>::infinity();
        float q_scale = 1.0f;
        for (int action = 0; action < 4; ++action) {
            const float q = snapshot.q_values[action];
            if (std::isfinite(q)) {
                q_scale = std::max(q_scale, std::abs(q));
                if (snapshot.legal_actions[action] && (greedy_action < 0 || q > best_q)) {
                    greedy_action = action;
                    best_q = q;
                }
            }
        }

        for (int layer = 0; layer < 5; ++layer) {
            const int count = static_cast<int>(positions[layer].size());
            for (int index = 0; index < count; ++index) {
                const int actual_index = index * snapshot.sizes[layer] / count;
                const float act = layer == 4 ? finite_or_zero(snapshot.q_values[index]) :
                                              activation(snapshot, layer, actual_index);
                const float strength = std::tanh(std::abs(act));
                const Uint8 intensity = static_cast<Uint8>(strength * 255.0f);
                const int size = 4 + static_cast<int>(strength * 6.0f);
                const auto& point = positions[layer][index];
                SDL_Rect rect{point.x - size, point.y - size, size * 2, size * 2};
                if (layer == 0) {
                    set_color(network_renderer, {50, intensity, 255, 255});
                } else if (layer == 4) {
                    set_color(network_renderer, snapshot.legal_actions[index] ?
                        SDL_Color{255, 200, intensity, 255} : SDL_Color{65, 65, 80, 255});
                } else {
                    set_color(network_renderer,
                        {static_cast<Uint8>(intensity * 0.7f), intensity, 255, 255});
                }
                SDL_RenderFillRect(network_renderer, &rect);
                if (layer == 4 && index == greedy_action) {
                    SDL_Rect outline{point.x - 13, point.y - 13, 26, 26};
                    set_color(network_renderer, {50, 255, 100, 255});
                    SDL_RenderDrawRect(network_renderer, &outline);
                }
                if (layer == 4 && index == status.action) {
                    SDL_Rect outline{point.x - 16, point.y - 16, 32, 32};
                    set_color(network_renderer, {245, 245, 255, 255});
                    SDL_RenderDrawRect(network_renderer, &outline);
                }
            }
            text(network_renderer, font_small, kLayerNames[layer],
                 60 + layer * 160 - 25, 53, {150, 150, 180, 255});
            text(network_renderer, font_small, std::to_string(snapshot.sizes[layer]) + " units",
                 60 + layer * 160 - 25, 538, {100, 100, 130, 255});
        }

        for (int action = 0; action < static_cast<int>(positions[4].size()); ++action) {
            const bool legal = snapshot.legal_actions[action];
            const int y = positions[4][action].y;
            const SDL_Color color = !legal ? SDL_Color{110, 110, 130, 255} :
                action == greedy_action ? SDL_Color{50, 255, 100, 255} :
                                          SDL_Color{200, 200, 220, 255};
            std::ostringstream label;
            label << kActionNames[action] << "  Q: " << std::fixed << std::setprecision(3)
                  << snapshot.q_values[action] << (legal ? "  legal" : "  blocked");
            fitted_text(network_renderer, label.str(), 735, y - 27, 310, color);

            // Signed bars share a scale and a zero point; numeric labels stay raw Q values.
            const int bar_width = static_cast<int>(
                finite_or_zero(snapshot.q_values[action]) / q_scale * 125.0f);
            set_color(network_renderer, {35, 35, 50, 255});
            SDL_Rect background{737, y, 256, 12};
            SDL_RenderFillRect(network_renderer, &background);
            set_color(network_renderer, legal ? color : SDL_Color{70, 70, 90, 255});
            SDL_Rect bar{865 + std::min(0, bar_width), y, std::abs(bar_width), 12};
            SDL_RenderFillRect(network_renderer, &bar);
            set_color(network_renderer, {140, 140, 165, 255});
            SDL_RenderDrawLine(network_renderer, 865, y - 2, 865, y + 14);
            if (action == status.action) {
                text(network_renderer, font_small,
                     status.exploratory ? "SELECTED: random exploration" : "SELECTED: greedy policy",
                     735, y + 24, {245, 245, 255, 255});
            }
        }

        text(network_renderer, font_small,
             status.evaluation ? "2048 Q network - evaluation (no learning)" :
                                 "2048 Q network - live training",
             20, 14, {220, 220, 240, 255});
        text(network_renderer, font_small, "Q = estimated future reward; highest legal Q is greedy",
             20, 574, {170, 170, 200, 255});

        std::ostringstream stats;
        stats << "Episode: " << status.episodes << "   Steps: " << status.total_steps
              << "   Updates: " << status.updates << "   Replay: " << status.replay_size;
        text(network_renderer, font_small, stats.str(), 20, 599, {200, 200, 220, 255});
        stats.str("");
        stats.clear();
        stats << "Random exploration: " << std::fixed << std::setprecision(1)
              << status.epsilon * 100.0f << "%"
              << "   Latest loss: " << std::setprecision(5) << status.loss;
        text(network_renderer, font_small, stats.str(), 20, 622, {200, 200, 220, 255});
        text(network_renderer, font_small,
             "Green outline = best legal Q   White outline = selected move   Bar center = Q of zero",
             20, 645, {130, 130, 160, 255});
        text(network_renderer, font_small,
             "Connections: red = positive signal, green = negative signal, brightness = strength",
             20, 668, {130, 130, 160, 255});
        const std::string shortcuts = status.evaluation ? "Esc / close: quit   Model: " :
                                                         "S: save   Esc / close: quit   Model: ";
        fitted_text(network_renderer, shortcuts + status.model_path,
                    20, 704, kNetworkWidth - 40, {160, 160, 190, 255});
        SDL_RenderPresent(network_renderer);
    }
};

TrainingVisualization::TrainingVisualization() : impl_(std::make_unique<Impl>()) {}
TrainingVisualization::~TrainingVisualization() = default;

bool TrainingVisualization::init() {
    if (impl_->renderer && impl_->network_renderer) return true;
    impl_->cleanup();
    SDL_SetMainReady();
    if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL video initialization failed: " << SDL_GetError() << '\n';
        return false;
    }
    impl_->sdl_initialized = true;
    if (TTF_Init() != 0) {
        std::cerr << "Font initialization failed: " << TTF_GetError() << '\n';
        impl_->cleanup();
        return false;
    }
    impl_->ttf_initialized = true;

    auto fail = [&](const char* message) {
        std::cerr << message << ": " << SDL_GetError() << '\n';
        impl_->cleanup();
        return false;
    };
    impl_->window = SDL_CreateWindow("2048 AI", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                     kGameWidth, kGameHeight, SDL_WINDOW_SHOWN);
    if (!impl_->window) return fail("Game window creation failed");
    impl_->renderer = SDL_CreateRenderer(impl_->window, -1, SDL_RENDERER_ACCELERATED);
    if (!impl_->renderer) return fail("Game renderer creation failed");
    impl_->network_window = SDL_CreateWindow("Neural Network - Live Activity",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        kNetworkWidth, kNetworkHeight, SDL_WINDOW_SHOWN);
    if (!impl_->network_window) return fail("Network window creation failed");
    impl_->network_renderer = SDL_CreateRenderer(impl_->network_window, -1, SDL_RENDERER_ACCELERATED);
    if (!impl_->network_renderer) return fail("Network renderer creation failed");

    // Keep both views visible on larger displays. On smaller displays, leave
    // the network title bar exposed above the foreground game window.
    SDL_Rect usable;
    const int display = std::max(0, SDL_GetWindowDisplayIndex(impl_->window));
    if (SDL_GetDisplayUsableBounds(display, &usable) == 0) {
        constexpr int gap = 24;
        int game_title_height = 0;
        int network_title_height = 0;
        SDL_GetWindowBordersSize(impl_->window, &game_title_height, nullptr, nullptr, nullptr);
        SDL_GetWindowBordersSize(impl_->network_window, &network_title_height, nullptr, nullptr, nullptr);
        const int total_width = kGameWidth + gap + kNetworkWidth;
        const int game_height = kGameHeight + game_title_height;
        const int network_height = kNetworkHeight + network_title_height;
        if (total_width <= usable.w && std::max(game_height, network_height) <= usable.h) {
            const int left = usable.x + (usable.w - total_width) / 2;
            SDL_SetWindowPosition(impl_->window, left,
                usable.y + (usable.h - game_height) / 2 + game_title_height);
            SDL_SetWindowPosition(impl_->network_window, left + kGameWidth + gap,
                usable.y + (usable.h - network_height) / 2 + network_title_height);
        } else {
            const int network_y = usable.y + network_title_height;
            SDL_SetWindowPosition(impl_->network_window,
                usable.x + std::max(0, (usable.w - kNetworkWidth) / 2), network_y);
            SDL_SetWindowPosition(impl_->window,
                usable.x + std::max(0, std::min(gap, usable.w - kGameWidth)),
                network_y + gap + game_title_height);
        }
    }
    SDL_RaiseWindow(impl_->window);

    // These renderers were created without SDL_RENDERER_PRESENTVSYNC.
    const char* font_paths[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    };
    for (const char* path : font_paths) {
        impl_->font = TTF_OpenFont(path, 24);
        if (!impl_->font) continue;
        impl_->font_large = TTF_OpenFont(path, 36);
        impl_->font_small = TTF_OpenFont(path, 14);
        break;
    }
    if (!impl_->font) {
        std::cerr << "Warning: no supported font found; visualization text is unavailable.\n";
    }
    return true;
}

UiEvents TrainingVisualization::poll_events() {
    UiEvents events;
    SDL_Event event;
    while (impl_->sdl_initialized && SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT ||
            (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE)) {
            events.quit = true;
        }
        if (event.type == SDL_KEYDOWN && event.key.repeat == 0) {
            if (event.key.keysym.sym == SDLK_ESCAPE) events.quit = true;
            if (event.key.keysym.sym == SDLK_s) events.save = true;
        }
    }
    return events;
}

void TrainingVisualization::render(const GameEngine& game, const TrainingStatus& status,
                                   const NetworkSnapshot& network) {
    if (!impl_->renderer || !impl_->network_renderer) return;
    impl_->render_game(game, status);
    impl_->render_network(status, network);
}

}  // namespace ai2048

#endif  // AI2048_TRAINING_VISUALIZATION_CPP
