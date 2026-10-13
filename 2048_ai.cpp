#include "training.cpp"
#include "training_visualization.cpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <thread>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void signal_handler(int) { interrupted = 1; }
struct Options {
    bool render = true, eval = false, random = false, fresh = false, quiet = false;
    bool episodes_set = false, seed_set = false;
    bool efficient = false;
    int episodes = 10000, delay_ms = -1;
    uint32_t seed = 42;
    int64_t exploration_steps = 100000;
    int64_t max_training_steps = 0;
    std::optional<float> gamma;
    std::optional<double> learning_rate;
    std::optional<int> n_step, train_every;
    std::optional<bool> symmetry, lookahead;
    std::string model = "2048_model.pt", csv;
};
int64_t integer(const std::string& text, const std::string& option) {
    int64_t number = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), number);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || number < 0)
        throw std::invalid_argument(option + " requires a nonnegative integer");
    return number;
}
float discount(const std::string& text) {
    size_t consumed = 0;
    const float number = std::stof(text, &consumed);
    if (consumed != text.size() || !std::isfinite(number) || number < 0 || number > 1)
        throw std::invalid_argument("--gamma requires a number between 0 and 1");
    return number;
}
double learning_rate(const std::string& text) {
    size_t consumed = 0;
    double number;
    try {
        number = std::stod(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument("--learning-rate requires a finite positive number");
    }
    if (consumed != text.size() || !std::isfinite(number) || number <= 0)
        throw std::invalid_argument("--learning-rate requires a finite positive number");
    return number;
}
void help() {
    std::cout << "2048 AI Neural Network\nUsage: 2048_ai [options]\n"
        << "  --episodes N          Additional games (default: training 10000, evaluation 20)\n"
        << "  --model PATH          Checkpoint (default 2048_model.pt)\n"
        << "  --no-render           Run without graphics\n"
        << "  --eval                Evaluate saved policy; no learning or model saves\n"
        << "  --lookahead           Plan through moves and random spawns using the network\n"
        << "  --no-lookahead        Select directly from network Q-values\n"
        << "  --random              Random legal-move baseline (requires --eval)\n"
        << "  --fresh               Start new weights instead of loading\n"
        << "  --seed N              Seed for new training (42) / evaluation (1000000)\n"
        << "  --exploration-steps N Decay exploration from 100% to 1% (default 100000; new runs)\n"
        << "  --efficient           Experimental: 3-step returns, symmetry, gamma .995, updates every 8\n"
        << "  --gamma G             Override discount on new or resumed training\n"
        << "  --learning-rate LR    Override Adam learning rate on new or resumed training\n"
        << "  --n-step N            Override reward propagation horizon (1..32)\n"
        << "  --augment-symmetry    Reuse rotated/reflected replay experience\n"
        << "  --no-symmetry         Disable symmetry augmentation\n"
        << "  --train-every N       Override moves between optimizer updates\n"
        << "  --max-training-steps N Stop and save after N additional moves (0: unlimited)\n"
        << "  --delay-ms N          Delay per displayed move (default: training 0, evaluation 80)\n"
        << "  --csv PATH            Write per-game metrics\n"
        << "  --quiet               Print only summaries\n"
        << "  --help                Show this help\n\n"
        << "Esc / Ctrl+C: quit; training saves on exit. S: save during training.\n";
}
}
int main(int argc, char** argv) {
    try {
        Options options;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (++i >= argc) throw std::invalid_argument("Missing value for " + arg);
                return argv[i];
            };
            if (arg == "--help") { help(); return 0; }
            if (arg == "--no-render") options.render = false;
            else if (arg == "--eval") options.eval = true;
            else if (arg == "--random") options.random = true;
            else if (arg == "--fresh") options.fresh = true;
            else if (arg == "--quiet") options.quiet = true;
            else if (arg == "--efficient") options.efficient = true;
            else if (arg == "--augment-symmetry") options.symmetry = true;
            else if (arg == "--no-symmetry") options.symmetry = false;
            else if (arg == "--lookahead") options.lookahead = true;
            else if (arg == "--no-lookahead") options.lookahead = false;
            else if (arg == "--gamma") options.gamma = discount(value());
            else if (arg == "--learning-rate") options.learning_rate = learning_rate(value());
            else if (arg == "--model") options.model = value();
            else if (arg == "--csv") options.csv = value();
            else if (arg == "--episodes" || arg == "--delay-ms" || arg == "--n-step" || arg == "--train-every") {
                const auto n = integer(value(), arg);
                if (n > std::numeric_limits<int>::max()) throw std::invalid_argument(arg + " is too large");
                if (arg == "--episodes") { options.episodes = static_cast<int>(n); options.episodes_set = true; }
                else if (arg == "--delay-ms") options.delay_ms = static_cast<int>(n);
                else if (arg == "--n-step") options.n_step = static_cast<int>(n);
                else options.train_every = static_cast<int>(n);
            } else if (arg == "--seed") {
                const auto n = integer(value(), arg);
                if (n > std::numeric_limits<uint32_t>::max()) throw std::invalid_argument("Seed is too large");
                options.seed = static_cast<uint32_t>(n);
                options.seed_set = true;
            } else if (arg == "--exploration-steps") options.exploration_steps = integer(value(), arg);
            else if (arg == "--max-training-steps") options.max_training_steps = integer(value(), arg);
            else throw std::invalid_argument("Unknown option: " + arg);
        }
        if (options.random && !options.eval) throw std::invalid_argument("--random requires --eval");
        if (options.random && options.lookahead.value_or(false))
            throw std::invalid_argument("--lookahead cannot be combined with --random");
        const bool tuning = options.efficient || options.gamma || options.n_step || options.train_every || options.symmetry;
        if (options.eval && (tuning || options.learning_rate || options.max_training_steps))
            throw std::invalid_argument("Training options cannot be used with --eval");
        if (options.eval && !options.episodes_set) options.episodes = 20;
        if (options.eval && !options.seed_set) options.seed = 1000000;
        if (options.delay_ms < 0) options.delay_ms = options.eval ? 80 : 0;
        if (options.model.empty()) throw std::invalid_argument("Model path cannot be empty");
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
        // Avoid CPU thread oversubscription for this small fully connected network.
        torch::set_num_threads(1);
        const torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        ai2048::TrainingConfig config;
        config.seed = options.seed;
        config.exploration_steps = options.exploration_steps;
        ai2048::DQNTrainer trainer(config, device);
        bool loaded = false;
        if (!options.fresh && !options.random) loaded = trainer.load(options.model);
        if (options.eval && !options.random && !options.fresh && !loaded)
            throw std::runtime_error("Checkpoint not found: " + options.model + ". Train first, or use --fresh --eval.");
        if (tuning) {
            auto tuned = trainer.config();
            if (options.efficient) {
                tuned.gamma = 0.995f;
                tuned.n_step = 3;
                tuned.symmetry_augmentation = true;
                tuned.train_every = 8;
                tuned.target_update_every = 125;
            }
            if (options.gamma) tuned.gamma = *options.gamma;
            if (options.n_step) tuned.n_step = *options.n_step;
            if (options.symmetry) tuned.symmetry_augmentation = *options.symmetry;
            if (options.train_every) {
                if (*options.train_every <= 0) throw std::invalid_argument("--train-every must be positive");
                const int64_t target_moves = static_cast<int64_t>(tuned.train_every) * tuned.target_update_every;
                tuned.train_every = *options.train_every;
                tuned.target_update_every = static_cast<int>(std::max(int64_t{1}, target_moves / tuned.train_every));
            }
            trainer.configure_learning(tuned.gamma, tuned.n_step, tuned.symmetry_augmentation,
                                       tuned.train_every, tuned.target_update_every);
        }
        if (options.learning_rate) trainer.configure_learning_rate(*options.learning_rate);
        if (options.lookahead) trainer.enable_lookahead(*options.lookahead);
        std::cout << "Device: " << device << " | " << (loaded ? "Loaded " : "New policy: ") << options.model << '\n';
        if (loaded && !options.eval)
            std::cout << "Resuming after " << trainer.completed_episodes() << " completed games and " << trainer.updates()
                      << " optimizer updates; replay will warm up again.\n";
        if (!options.eval) {
            const auto& learning = trainer.config();
            std::cout << "Training: learning-rate=" << learning.learning_rate
                      << " | gamma=" << learning.gamma << " | n-step=" << learning.n_step
                      << " | symmetry=" << (learning.symmetry_augmentation ? "on" : "off")
                      << " | lookahead=" << (learning.lookahead ? "on" : "off")
                      << " | update every " << learning.train_every << " moves\n";
        }
        std::ofstream csv;
        if (!options.csv.empty()) {
            csv.open(options.csv);
            if (!csv) throw std::runtime_error("Cannot write metrics: " + options.csv);
            csv << "episode,score,max_tile,steps,truncated,loss,epsilon\n";
        }
        ai2048::TrainingVisualization visualization;
        if (options.render && !visualization.init()) return 1;
        ai2048::TrainingCallbacks callbacks;
        const auto starting_steps = trainer.total_steps();
        callbacks.should_stop = [&] {
            if (interrupted) return true;
            if (!options.eval && options.max_training_steps &&
                trainer.total_steps() - starting_steps >= options.max_training_steps) return true;
            if (options.render) {
                const auto event = visualization.poll_events();
                if (event.save && !options.eval) { trainer.save(options.model); std::cout << "Saved " << options.model << '\n'; }
                if (event.quit) return true;
            }
            return false;
        };
        if (options.render) callbacks.on_frame = [&](const auto& game, auto status, const auto& network) {
            status.model_path = options.model;
            visualization.render(game, status, network);
            if (options.delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(options.delay_ms));
        };
        int run_games = 0, best_score = 0, max_tile_ever = 0, games_reaching_2048 = 0;
        std::array<ai2048::EpisodeMetrics, 100> recent{};
        callbacks.on_episode = [&](const ai2048::EpisodeMetrics& game) {
            recent[run_games % recent.size()] = game;
            ++run_games;
            best_score = std::max(best_score, game.score);
            max_tile_ever = std::max(max_tile_ever, game.max_tile);
            if (game.max_tile >= 2048) ++games_reaching_2048;
            if (!options.quiet) {
                if (options.eval) {
                    std::cout << "Ep " << game.episode << " | Score: " << game.score << " | Max: " << game.max_tile
                              << " | Best: " << best_score << " | MaxTileEver: " << max_tile_ever
                              << " | Loss: " << std::fixed << std::setprecision(4) << game.loss
                              << " | Random moves: " << std::setprecision(1) << 100 * game.epsilon << "%"
                              << (game.truncated ? " | TRUNCATED" : "") << std::endl;
                } else {
                    std::cout << "Ep " << std::setw(5) << game.episode
                              << " | Score: " << std::setw(5) << game.score
                              << " | MaxTile: " << std::setw(4) << game.max_tile
                              << " | Best: " << std::setw(5) << best_score
                              << " | MaxTileEver: " << std::setw(4) << max_tile_ever
                              << " | Loss: " << std::fixed << std::setprecision(4) << game.loss
                              << " | Eps: " << std::setprecision(3) << game.epsilon
                              << (game.truncated ? " | TRUNCATED" : "") << std::endl;
                }
                if (run_games % recent.size() == 0) {
                    double score_sum = 0;
                    int reached_1024 = 0, reached_2048 = 0;
                    for (const auto& item : recent) {
                        score_sum += item.score;
                        reached_1024 += item.max_tile >= 1024;
                        reached_2048 += item.max_tile >= 2048;
                    }
                    std::cout << "Last 100 games | Mean score: " << std::fixed << std::setprecision(1)
                              << score_sum / recent.size() << " | 1024+: " << reached_1024
                              << "/100 | 2048+: " << reached_2048 << "/100" << std::endl;
                }
            }
            if (csv) csv << game.episode << ',' << game.score << ',' << game.max_tile << ',' << game.steps << ','
                         << game.truncated << ',' << std::setprecision(9) << game.loss << ',' << game.epsilon << '\n';
            if (!options.eval && game.episode % 100 == 0) trainer.save(options.model);
        };
        const auto started = std::chrono::steady_clock::now();
        int64_t measured_steps = 0;
        if (options.eval) {
            const auto result = trainer.evaluate(options.episodes, options.seed, options.random, callbacks);
            std::cout << "Evaluation: " << (options.random ? "random legal moves" :
                                           trainer.config().lookahead ? "lookahead with network" : "greedy network")
                      << " | seed=" << options.seed << " | games=" << result.games.size()
                      << " | mean_score=" << std::fixed << std::setprecision(2) << result.mean_score()
                      << " | median_score=" << result.median_score() << " | truncated=" << result.truncated_count() << '\n';
            for (int tile : {128, 256, 512, 1024, 2048})
                std::cout << "Tile >= " << tile << ": " << result.tile_count(tile) << '/' << result.games.size() << '\n';
            for (const auto& game : result.games) measured_steps += game.steps;
        } else {
            trainer.train(options.episodes, callbacks);
            trainer.save(options.model);
            std::cout << '\n' << (run_games == options.episodes ? "Training Complete!" : "Training Stopped")
                      << " | Games: " << run_games << '/' << options.episodes
                      << "\nBest Score: " << best_score << " | Highest Tile: " << max_tile_ever
                      << "\n2048 Games: " << games_reaching_2048 << '\n';
            std::cout << "Saved " << options.model << " | completed_games=" << trainer.completed_episodes()
                      << " | steps=" << trainer.total_steps() << " | updates=" << trainer.updates()
                      << " | epsilon=" << std::setprecision(4) << trainer.epsilon() << '\n';
            measured_steps = trainer.total_steps() - starting_steps;
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cout << "Elapsed: " << std::fixed << std::setprecision(2) << seconds << " s | Moves: " << measured_steps
                  << " | Moves/s: " << (seconds > 0 ? measured_steps / seconds : 0) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
