#ifndef AI2048_TRAINING_CPP
#define AI2048_TRAINING_CPP

// Declarations and implementation are kept together; include this component once.

#include "game_engine.cpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ai2048 {

// Plain display data shared by training and the UI.
struct TrainingStatus {
    int episodes = 0;
    std::int64_t total_steps = 0;
    std::int64_t updates = 0;
    float epsilon = 1.0f;
    float loss = 0.0f;
    std::size_t replay_size = 0;
    std::string model_path;
    bool evaluation = false;
    int action = -1;
    bool exploratory = false;
};

struct NetworkSnapshot {
    // One vector per layer, including the input and the raw Q outputs.
    std::vector<std::vector<float>> activations;
    // Four weight matrices flattened in row-major [output][input] order.
    std::vector<std::vector<float>> weights;
    std::array<int, 5> sizes{16, 256, 128, 128, 4};
    std::array<float, 4> q_values{};
    ActionMask legal_actions{};
};

}  // namespace ai2048

#include <torch/torch.h>
#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace ai2048 {

struct QNetImpl : torch::nn::Module {
    QNetImpl();
    torch::Tensor forward(torch::Tensor input);
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
};
TORCH_MODULE(QNet);

struct TrainingConfig {
    double learning_rate = 0.0001;
    float gamma = 0.99f;
    float epsilon_start = 1.0f;
    float epsilon_end = 0.01f;
    int64_t exploration_steps = 100000;
    size_t replay_capacity = 100000;
    size_t batch_size = 256;
    int train_every = 4;
    int target_update_every = 250; // Optimizer updates, not episodes.
    int max_episode_steps = 5000;
    uint32_t seed = 42;
};

struct Experience {
    State state;
    int action;
    float reward;
    State next_state;
    ActionMask next_valid;
    bool done;
};

struct ActionSelection { int action; bool exploratory; };
struct EpisodeMetrics {
    int episode = 0;
    int score = 0;
    int max_tile = 0;
    int steps = 0;
    bool truncated = false;
    float loss = 0;
    float epsilon = 0;
};
struct EvaluationResult {
    std::vector<EpisodeMetrics> games;
    double mean_score() const;
    double median_score() const;
    int tile_count(int tile) const;
    int truncated_count() const;
};
struct TrainingCallbacks {
    std::function<bool()> should_stop;
    std::function<void(const GameEngine&, const TrainingStatus&, const NetworkSnapshot&)> on_frame;
    std::function<void(const EpisodeMetrics&)> on_episode;
};

// Double DQN: online net selects the best legal action; target net values it.
// Terminal rows never bootstrap, including rows with no legal actions.
torch::Tensor double_dqn_targets(const torch::Tensor& rewards,
    const torch::Tensor& dones, const torch::Tensor& online_next,
    const torch::Tensor& target_next, const torch::Tensor& legal, float gamma);

class DQNTrainer {
public:
    explicit DQNTrainer(TrainingConfig config = {}, torch::Device device = torch::kCPU);
    ActionSelection select_action(const State&, const ActionMask&, float exploration);
    void remember(const Experience&);
    float optimize(); // Requires at least batch_size transitions.
    void sync_target();
    void train(int episodes, const TrainingCallbacks& callbacks = {});
    EvaluationResult evaluate(int episodes, uint32_t seed, bool random_policy = false,
                              const TrainingCallbacks& callbacks = {});
    void save(const std::string& path);
    bool load(const std::string& path); // Missing is false; corrupt/incompatible throws.
    float epsilon() const;
    TrainingStatus status() const;
    NetworkSnapshot snapshot(const State&, const ActionMask&);
    const TrainingConfig& config() const { return config_; }
    int64_t total_steps() const { return total_steps_; }
    int64_t updates() const { return updates_; }
    int completed_episodes() const { return completed_episodes_; }
    size_t replay_size() const { return replay_.size(); }
    QNet& policy_network() { return policy_; }
    QNet& target_network() { return target_; }

private:
    TrainingConfig config_;
    torch::Device device_;
    QNet policy_{nullptr}, target_{nullptr};
    std::unique_ptr<torch::optim::Adam> optimizer_;
    std::vector<Experience> replay_;
    size_t replay_next_ = 0;
    std::mt19937 rng_;
    int64_t total_steps_ = 0, updates_ = 0;
    int completed_episodes_ = 0;
    float last_loss_ = 0;
    std::map<int, int> milestones_;
    int high_score_ = 0, high_score_episode_ = 0;
};
} // namespace ai2048

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace ai2048 {
namespace {
void validate(const TrainingConfig& c) {
    if (!(c.learning_rate > 0) || !std::isfinite(c.learning_rate) ||
        !(c.gamma >= 0 && c.gamma <= 1) ||
        !(c.epsilon_end >= 0 && c.epsilon_end <= c.epsilon_start && c.epsilon_start <= 1) ||
        c.exploration_steps <= 0 || c.batch_size == 0 || c.replay_capacity < c.batch_size ||
        c.train_every <= 0 || c.target_update_every <= 0 || c.max_episode_steps <= 0)
        throw std::invalid_argument("Invalid training configuration");
}

torch::Tensor state_tensor(const State& state, const torch::Device& device) {
    return torch::from_blob(const_cast<float*>(state.data()), {1, kInputSize},
                           torch::kFloat32).clone().to(device);
}

int greedy_action(const torch::Tensor& q_values, const ActionMask& valid) {
    auto values = q_values.detach().to(torch::kCPU).contiguous().view({kActionCount});
    const auto* q = values.data_ptr<float>();
    int best = -1;
    for (int i = 0; i < kActionCount; ++i) {
        if (!std::isfinite(q[i])) throw std::runtime_error("Non-finite network output");
        if (valid[i] && (best < 0 || q[i] > q[best])) best = i;
    }
    if (best < 0) throw std::invalid_argument("Cannot select an action on a terminal board");
    return best;
}

int random_action(const ActionMask& valid, std::mt19937& rng) {
    std::vector<int> actions;
    for (int i = 0; i < kActionCount; ++i) if (valid[i]) actions.push_back(i);
    if (actions.empty()) throw std::invalid_argument("Cannot select an action on a terminal board");
    return actions[std::uniform_int_distribution<size_t>(0, actions.size() - 1)(rng)];
}

std::vector<float> cpu_values(const torch::Tensor& tensor) {
    auto cpu = tensor.detach().to(torch::kCPU).contiguous();
    return {cpu.data_ptr<float>(), cpu.data_ptr<float>() + cpu.numel()};
}
} // namespace

QNetImpl::QNetImpl() {
    fc1 = register_module("fc1", torch::nn::Linear(kInputSize, 256));
    fc2 = register_module("fc2", torch::nn::Linear(256, 128));
    fc3 = register_module("fc3", torch::nn::Linear(128, 128));
    fc4 = register_module("fc4", torch::nn::Linear(128, kActionCount));
}

torch::Tensor QNetImpl::forward(torch::Tensor input) {
    return fc4->forward(torch::relu(fc3->forward(torch::relu(
        fc2->forward(torch::relu(fc1->forward(input)))))));
}

torch::Tensor double_dqn_targets(const torch::Tensor& rewards,
    const torch::Tensor& dones, const torch::Tensor& online_next,
    const torch::Tensor& target_next, const torch::Tensor& legal, float gamma) {
    torch::NoGradGuard guard;
    const auto mask = legal.to(torch::kBool);
    auto actions = online_next.masked_fill(~mask, -std::numeric_limits<float>::infinity())
                              .argmax(1, true);
    auto next = target_next.gather(1, actions).squeeze(1);
    // Set terminal bootstrap to zero BEFORE multiplying: 0 * inf is NaN.
    next = next.masked_fill(dones.to(torch::kBool) | ~mask.any(1), 0);
    return rewards + gamma * next;
}

DQNTrainer::DQNTrainer(TrainingConfig config, torch::Device device)
    : config_(config), device_(device), rng_(config.seed) {
    validate(config_);
    torch::manual_seed(config_.seed);
    policy_ = QNet();
    target_ = QNet();
    policy_->to(device_);
    target_->to(device_);
    for (auto& parameter : target_->parameters()) parameter.requires_grad_(false);
    target_->eval();
    optimizer_ = std::make_unique<torch::optim::Adam>(policy_->parameters(),
        torch::optim::AdamOptions(config_.learning_rate));
    sync_target();
    replay_.reserve(config_.replay_capacity);
}

float DQNTrainer::epsilon() const {
    const double fraction = std::min(1.0, static_cast<double>(total_steps_) / config_.exploration_steps);
    return static_cast<float>(config_.epsilon_start +
        fraction * (config_.epsilon_end - config_.epsilon_start));
}

ActionSelection DQNTrainer::select_action(const State& state, const ActionMask& valid, float exploration) {
    if (!(exploration >= 0 && exploration <= 1)) throw std::invalid_argument("Invalid epsilon");
    if (std::uniform_real_distribution<float>(0, 1)(rng_) < exploration)
        return {random_action(valid, rng_), true};
    torch::NoGradGuard guard;
    return {greedy_action(policy_->forward(state_tensor(state, device_)), valid), false};
}

void DQNTrainer::remember(const Experience& experience) {
    if (experience.action < 0 || experience.action >= kActionCount || !std::isfinite(experience.reward))
        throw std::invalid_argument("Invalid replay transition");
    if (replay_.size() < config_.replay_capacity) replay_.push_back(experience);
    else replay_[replay_next_] = experience;
    replay_next_ = (replay_next_ + 1) % config_.replay_capacity;
}

float DQNTrainer::optimize() {
    if (replay_.size() < config_.batch_size) throw std::logic_error("Replay buffer is not ready");
    const auto count = static_cast<int64_t>(config_.batch_size);
    auto states = torch::empty({count, kInputSize});
    auto next_states = torch::empty_like(states);
    auto actions = torch::empty({count, 1}, torch::kLong);
    auto rewards = torch::empty({count});
    auto dones = torch::empty({count}, torch::kBool);
    auto masks = torch::empty({count, kActionCount}, torch::kBool);
    std::uniform_int_distribution<size_t> sample(0, replay_.size() - 1);
    for (int64_t row = 0; row < count; ++row) {
        const auto& exp = replay_[sample(rng_)];
        std::copy(exp.state.begin(), exp.state.end(), states.data_ptr<float>() + row * kInputSize);
        std::copy(exp.next_state.begin(), exp.next_state.end(), next_states.data_ptr<float>() + row * kInputSize);
        actions.data_ptr<int64_t>()[row] = exp.action;
        rewards.data_ptr<float>()[row] = exp.reward;
        dones.data_ptr<bool>()[row] = exp.done;
        std::copy(exp.next_valid.begin(), exp.next_valid.end(), masks.data_ptr<bool>() + row * kActionCount);
    }
    states = states.to(device_);
    next_states = next_states.to(device_);
    actions = actions.to(device_);
    rewards = rewards.to(device_);
    dones = dones.to(device_);
    masks = masks.to(device_);
    torch::Tensor targets;
    {
        torch::NoGradGuard guard;
        targets = double_dqn_targets(rewards, dones, policy_->forward(next_states),
                                    target_->forward(next_states), masks, config_.gamma);
    }
    auto predicted = policy_->forward(states).gather(1, actions).squeeze(1);
    auto loss = torch::smooth_l1_loss(predicted, targets);
    last_loss_ = loss.item<float>();
    if (!std::isfinite(last_loss_)) throw std::runtime_error("Non-finite training loss");
    optimizer_->zero_grad();
    loss.backward();
    const double gradient_norm = torch::nn::utils::clip_grad_norm_(policy_->parameters(), 1.0);
    if (!std::isfinite(gradient_norm)) throw std::runtime_error("Non-finite training gradient");
    optimizer_->step();
    ++updates_;
    if (updates_ % config_.target_update_every == 0) sync_target();
    return last_loss_;
}

void DQNTrainer::sync_target() {
    torch::NoGradGuard guard;
    const auto source = policy_->parameters();
    auto destination = target_->parameters();
    for (size_t i = 0; i < source.size(); ++i) destination[i].copy_(source[i]);
}

TrainingStatus DQNTrainer::status() const {
    TrainingStatus status;
    status.episodes = completed_episodes_;
    status.total_steps = total_steps_;
    status.updates = updates_;
    status.epsilon = epsilon();
    status.loss = last_loss_;
    status.replay_size = replay_.size();
    return status;
}

NetworkSnapshot DQNTrainer::snapshot(const State& state, const ActionMask& valid) {
    torch::NoGradGuard guard;
    NetworkSnapshot view;
    auto input = state_tensor(state, device_);
    auto h1 = torch::relu(policy_->fc1->forward(input));
    auto h2 = torch::relu(policy_->fc2->forward(h1));
    auto h3 = torch::relu(policy_->fc3->forward(h2));
    auto output = policy_->fc4->forward(h3);
    for (const auto& tensor : {input, h1, h2, h3, output}) view.activations.push_back(cpu_values(tensor));
    for (const auto& layer : {policy_->fc1, policy_->fc2, policy_->fc3, policy_->fc4})
        view.weights.push_back(cpu_values(layer->weight));
    std::copy(view.activations.back().begin(), view.activations.back().end(), view.q_values.begin());
    view.legal_actions = valid;
    return view;
}

void DQNTrainer::train(int episodes, const TrainingCallbacks& callbacks) {
    if (episodes < 0) throw std::invalid_argument("Episode count cannot be negative");
    policy_->train();
    auto last_frame = std::chrono::steady_clock::time_point::min();
    for (int n = 0; n < episodes; ++n) {
        // Per-episode seeds keep tile randomness independent of replay sampling.
        GameEngine game(config_.seed + static_cast<uint32_t>(completed_episodes_));
        EpisodeMetrics metrics;
        double loss_sum = 0;
        int loss_count = 0;
        while (!game.game_over() && metrics.steps < config_.max_episode_steps) {
            if (callbacks.should_stop && callbacks.should_stop()) return;
            auto state = game.state();
            auto valid = game.valid_actions();
            auto selected = select_action(state, valid, epsilon());
            auto now = std::chrono::steady_clock::now();
            if (callbacks.on_frame && (last_frame == std::chrono::steady_clock::time_point::min() ||
                now - last_frame >= std::chrono::milliseconds(33))) {
                auto view = status();
                view.episodes = completed_episodes_ + 1;
                view.action = selected.action;
                view.exploratory = selected.exploratory;
                callbacks.on_frame(game, view, snapshot(state, valid));
                last_frame = now;
            }
            const auto result = game.step(selected.action);
            // Optimize discounted game score. No per-turn board or terminal bonuses.
            remember({state, selected.action, result.score_gain / 100.0f,
                      game.state(), game.valid_actions(), result.done});
            ++total_steps_;
            ++metrics.steps;
            if (total_steps_ % config_.train_every == 0 && replay_.size() >= config_.batch_size) {
                loss_sum += optimize();
                ++loss_count;
            }
        }
        ++completed_episodes_;
        metrics.episode = completed_episodes_;
        metrics.score = game.score();
        metrics.max_tile = game.max_tile();
        metrics.truncated = !game.game_over();
        metrics.epsilon = epsilon();
        metrics.loss = loss_count ? static_cast<float>(loss_sum / loss_count) : 0;
        for (int tile : {64, 128, 256, 512, 1024, 2048, 4096, 8192})
            if (game.max_tile() >= tile) milestones_.emplace(tile, completed_episodes_);
        if (game.score() > high_score_) {
            high_score_ = game.score();
            high_score_episode_ = completed_episodes_;
        }
        if (callbacks.on_episode) callbacks.on_episode(metrics);
    }
}

EvaluationResult DQNTrainer::evaluate(int episodes, uint32_t seed, bool random_policy,
                                     const TrainingCallbacks& callbacks) {
    if (episodes <= 0) throw std::invalid_argument("Evaluation needs at least one episode");
    torch::NoGradGuard guard;
    EvaluationResult result;
    std::mt19937 evaluation_rng(seed);
    for (int episode = 0; episode < episodes; ++episode) {
        GameEngine game(seed + static_cast<uint32_t>(episode));
        EpisodeMetrics metrics;
        metrics.episode = episode + 1;
        metrics.epsilon = random_policy ? 1.0f : 0.0f;
        while (!game.game_over() && metrics.steps < config_.max_episode_steps) {
            if (callbacks.should_stop && callbacks.should_stop()) return result;
            const auto valid = game.valid_actions();
            const auto state = game.state();
            const int action = random_policy ? random_action(valid, evaluation_rng) :
                greedy_action(policy_->forward(state_tensor(state, device_)), valid);
            if (callbacks.on_frame) {
                auto view = status();
                view.episodes = episode + 1;
                view.evaluation = true;
                view.epsilon = random_policy ? 1 : 0;
                view.action = action;
                view.exploratory = random_policy;
                callbacks.on_frame(game, view, snapshot(state, valid));
            }
            game.step(action);
            ++metrics.steps;
        }
        metrics.score = game.score();
        metrics.max_tile = game.max_tile();
        metrics.truncated = !game.game_over();
        result.games.push_back(metrics);
        if (callbacks.on_episode) callbacks.on_episode(metrics);
    }
    return result;
}

double EvaluationResult::mean_score() const {
    double total = 0;
    for (const auto& game : games) total += game.score;
    return games.empty() ? 0 : total / games.size();
}
double EvaluationResult::median_score() const {
    std::vector<int> scores;
    for (const auto& game : games) scores.push_back(game.score);
    if (scores.empty()) return 0;
    std::sort(scores.begin(), scores.end());
    const auto mid = scores.size() / 2;
    return (static_cast<double>(scores[mid]) + scores[(scores.size() - 1) / 2]) / 2;
}
int EvaluationResult::tile_count(int tile) const {
    return static_cast<int>(std::count_if(games.begin(), games.end(),
        [tile](const auto& game) { return game.max_tile >= tile; }));
}
int EvaluationResult::truncated_count() const {
    return static_cast<int>(std::count_if(games.begin(), games.end(),
        [](const auto& game) { return game.truncated; }));
}

void DQNTrainer::save(const std::string& path) {
    torch::serialize::OutputArchive archive, policy, target, optimizer;
    policy_->save(policy);
    target_->save(target);
    optimizer_->save(optimizer);
    archive.write("version", c10::IValue(int64_t{1}));
    archive.write("policy", policy);
    archive.write("target", target);
    archive.write("optimizer", optimizer);
    archive.write("steps", c10::IValue(total_steps_));
    archive.write("updates", c10::IValue(updates_));
    archive.write("episodes", c10::IValue(int64_t{completed_episodes_}));
    std::ostringstream config, rng, milestones;
    config << std::setprecision(std::numeric_limits<double>::max_digits10)
           << config_.learning_rate << ' ' << config_.gamma << ' ' << config_.epsilon_start << ' '
           << config_.epsilon_end << ' ' << config_.exploration_steps << ' ' << config_.replay_capacity
           << ' ' << config_.batch_size << ' ' << config_.train_every << ' ' << config_.target_update_every
           << ' ' << config_.max_episode_steps << ' ' << config_.seed;
    rng << rng_;
    milestones << "HIGH_SCORE " << high_score_ << ' ' << high_score_episode_ << '\n';
    for (auto [tile, episode] : milestones_) milestones << "MILESTONE " << tile << ' ' << episode << '\n';
    archive.write("config", c10::IValue(config.str()));
    archive.write("rng", c10::IValue(rng.str()));
    archive.write("milestones", c10::IValue(milestones.str()));
    // Save completely before replacing the previous checkpoint.
    const auto temporary = path + ".tmp";
    archive.save_to(temporary);
    std::filesystem::rename(temporary, path);
    // Human-readable companions; the .pt archive is authoritative on reload.
    std::ofstream state(path + ".state");
    state << std::setprecision(9) << epsilon() << '\n' << total_steps_ << '\n' << completed_episodes_ << '\n';
    std::ofstream(path + ".milestones") << milestones.str();
}

bool DQNTrainer::load(const std::string& path) {
    if (!std::filesystem::exists(path)) return false;
    torch::serialize::InputArchive archive;
    archive.load_from(path, device_);
    c10::IValue value;
    if (!archive.try_read("version", value)) {
        DQNTrainer restored(config_, device_);
        restored.policy_->load(archive);
        restored.sync_target();
        *this = std::move(restored);
        std::cout << "Loaded legacy network weights; optimizer and exploration schedule start fresh.\n";
        return true;
    }
    if (value.toInt() != 1) throw std::runtime_error("Unsupported checkpoint version");
    archive.read("config", value);
    std::istringstream config(value.toStringRef());
    TrainingConfig saved;
    if (!(config >> saved.learning_rate >> saved.gamma >> saved.epsilon_start >> saved.epsilon_end
          >> saved.exploration_steps >> saved.replay_capacity >> saved.batch_size >> saved.train_every
          >> saved.target_update_every >> saved.max_episode_steps >> saved.seed))
        throw std::runtime_error("Invalid checkpoint configuration");
    DQNTrainer restored(saved, device_);
    torch::serialize::InputArchive policy, target, optimizer;
    archive.read("policy", policy);
    archive.read("target", target);
    archive.read("optimizer", optimizer);
    restored.policy_->load(policy);
    restored.target_->load(target);
    restored.optimizer_->load(optimizer);
    archive.read("steps", value); restored.total_steps_ = value.toInt();
    archive.read("updates", value); restored.updates_ = value.toInt();
    archive.read("episodes", value);
    if (value.toInt() < 0 || value.toInt() > std::numeric_limits<int>::max() ||
        restored.total_steps_ < 0 || restored.updates_ < 0)
        throw std::runtime_error("Invalid checkpoint counters");
    restored.completed_episodes_ = static_cast<int>(value.toInt());
    archive.read("rng", value);
    std::istringstream rng(value.toStringRef());
    if (!(rng >> restored.rng_)) throw std::runtime_error("Invalid checkpoint random state");
    archive.read("milestones", value);
    std::istringstream milestones(value.toStringRef());
    std::string kind;
    int first, second;
    while (milestones >> kind >> first >> second) {
        if (kind == "HIGH_SCORE") {
            restored.high_score_ = first;
            restored.high_score_episode_ = second;
        } else if (kind == "MILESTONE") restored.milestones_[first] = second;
    }
    *this = std::move(restored);
    return true;
}
} // namespace ai2048

#endif  // AI2048_TRAINING_CPP
