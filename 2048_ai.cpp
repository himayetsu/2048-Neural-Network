#include "training.cpp"
#include "training_visualization.cpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void signal_handler(int) { interrupted = 1; }
struct Options {
    bool render = true, eval = false, random = false, fresh = false, quiet = false;
    bool episodes_set = false, seed_set = false;
    int episodes = 10000, delay_ms = -1;
    uint32_t seed = 42;
    int64_t exploration_steps = 100000;
    std::string model = "2048_model.pt", csv;
};
int64_t integer(const std::string& text, const std::string& option) {
    int64_t number = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), number);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || number < 0)
        throw std::invalid_argument(option + " requires a nonnegative integer");
    return number;
}
void help() {
    std::cout << "2048 AI Neural Network\nUsage: 2048_ai [options]\n"
        << "  --episodes N          Additional games (default: training 10000, evaluation 20)\n"
        << "  --model PATH          Checkpoint (default 2048_model.pt)\n"
        << "  --no-render           Run without graphics\n"
        << "  --eval                Greedy evaluation; no learning or model saves\n"
        << "  --random              Random legal-move baseline (requires --eval)\n"
        << "  --fresh               Start new weights instead of loading\n"
        << "  --seed N              Seed for new training (42) / evaluation (1000000)\n"
        << "  --exploration-steps N Decay exploration from 100% to 1% (default 100000; new runs)\n"
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
            else if (arg == "--model") options.model = value();
            else if (arg == "--csv") options.csv = value();
            else if (arg == "--episodes" || arg == "--delay-ms") {
                const auto n = integer(value(), arg);
                if (n > std::numeric_limits<int>::max()) throw std::invalid_argument(arg + " is too large");
                if (arg == "--episodes") { options.episodes = static_cast<int>(n); options.episodes_set = true; }
                else options.delay_ms = static_cast<int>(n);
            } else if (arg == "--seed") {
                const auto n = integer(value(), arg);
                if (n > std::numeric_limits<uint32_t>::max()) throw std::invalid_argument("Seed is too large");
                options.seed = static_cast<uint32_t>(n);
                options.seed_set = true;
            } else if (arg == "--exploration-steps") options.exploration_steps = integer(value(), arg);
            else throw std::invalid_argument("Unknown option: " + arg);
        }
        if (options.random && !options.eval) throw std::invalid_argument("--random requires --eval");
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
        std::cout << "Device: " << device << " | " << (loaded ? "Loaded " : "New policy: ") << options.model << '\n';
        if (loaded && !options.eval)
            std::cout << "Resuming after " << trainer.completed_episodes() << " completed games and " << trainer.updates()
                      << " optimizer updates; replay will warm up again.\n";
        std::ofstream csv;
        if (!options.csv.empty()) {
            csv.open(options.csv);
            if (!csv) throw std::runtime_error("Cannot write metrics: " + options.csv);
            csv << "episode,score,max_tile,steps,truncated,loss,epsilon\n";
        }
        ai2048::TrainingVisualization visualization;
        if (options.render && !visualization.init()) return 1;
        ai2048::TrainingCallbacks callbacks;
        callbacks.should_stop = [&] {
            if (interrupted) return true;
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
        callbacks.on_episode = [&](const ai2048::EpisodeMetrics& game) {
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
            }
            if (csv) csv << game.episode << ',' << game.score << ',' << game.max_tile << ',' << game.steps << ','
                         << game.truncated << ',' << std::setprecision(9) << game.loss << ',' << game.epsilon << '\n';
            if (!options.eval && game.episode % 100 == 0) trainer.save(options.model);
        };
        if (options.eval) {
            const auto result = trainer.evaluate(options.episodes, options.seed, options.random, callbacks);
            std::cout << "Evaluation: " << (options.random ? "random legal moves" : "greedy network")
                      << " | seed=" << options.seed << " | games=" << result.games.size()
                      << " | mean_score=" << std::fixed << std::setprecision(2) << result.mean_score()
                      << " | median_score=" << result.median_score() << " | truncated=" << result.truncated_count() << '\n';
            for (int tile : {128, 256, 512, 1024, 2048})
                std::cout << "Tile >= " << tile << ": " << result.tile_count(tile) << '/' << result.games.size() << '\n';
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
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
