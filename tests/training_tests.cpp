#include "training.cpp"

#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ai2048;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Function>
void require_throws(Function function, const std::string& message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

struct TemporaryCheckpoint {
    explicit TemporaryCheckpoint(const std::string& name) {
        const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        path = (std::filesystem::temp_directory_path() /
                ("2048-training-test-" + name + "-" + std::to_string(stamp) + ".pt")).string();
    }
    ~TemporaryCheckpoint() {
        std::error_code error;
        std::filesystem::remove(path, error);
        std::filesystem::remove(path + ".state", error);
        std::filesystem::remove(path + ".milestones", error);
    }
    std::string path;
};

std::vector<torch::Tensor> parameters(QNet& network) {
    std::vector<torch::Tensor> result;
    for (const auto& parameter : network->parameters()) result.push_back(parameter.detach().clone());
    return result;
}

bool parameters_equal(QNet& network, const std::vector<torch::Tensor>& expected) {
    const auto actual = network->parameters();
    if (actual.size() != expected.size()) return false;
    for (size_t index = 0; index < actual.size(); ++index) {
        if (!torch::equal(actual[index], expected[index])) return false;
    }
    return true;
}

void set_constant_outputs(QNet& network, const std::vector<float>& outputs) {
    torch::NoGradGuard guard;
    for (auto& parameter : network->parameters()) parameter.zero_();
    network->fc4->bias.copy_(torch::tensor(outputs));
}

Experience terminal_experience() {
    State state{};
    state[0] = 1.0f / 17.0f;
    state[5] = 4.0f / 17.0f;
    return {state, 2, 1.0f, State{}, ActionMask{}, true};
}

void fill_terminal_replay(DQNTrainer& trainer) {
    const auto transition = terminal_experience();
    for (size_t index = 0; index < trainer.config().replay_capacity; ++index) trainer.remember(transition);
}

TrainingConfig small_config() {
    TrainingConfig config;
    config.seed = 19;
    config.batch_size = 8;
    config.replay_capacity = 32;
    config.max_episode_steps = 5;
    config.train_every = 1;
    config.target_update_every = 100;
    return config;
}

void double_dqn_uses_online_selection_and_target_values() {
    const auto rewards = torch::tensor({1.0f, 2.0f});
    const auto dones = torch::tensor({false, false}, torch::kBool);
    const auto online = torch::tensor({{1.0f, 4.0f, 3.0f, 99.0f}, {7.0f, 2.0f, 9.0f, 0.0f}});
    const auto target = torch::tensor({{50.0f, 5.0f, 70.0f, 1000.0f}, {3.0f, 100.0f, 40.0f, 80.0f}});
    const auto legal = torch::tensor({{true, true, true, false}, {true, true, false, false}}, torch::kBool);
    const auto result = double_dqn_targets(rewards, dones, online, target, legal, 0.5f);
    require(torch::allclose(result, torch::tensor({3.5f, 3.5f})),
            "Double DQN: online values must select a legal action and target values must price that same action");
}

void terminal_targets_are_finite_and_detached() {
    auto rewards = torch::tensor({2.0f, 3.0f, 4.0f}, torch::requires_grad());
    const auto dones = torch::tensor({true, true, false}, torch::kBool);
    auto online = torch::tensor({{10.0f, 20.0f, 30.0f, 40.0f},
                                 {10.0f, 20.0f, 30.0f, 40.0f},
                                 {10.0f, 20.0f, 30.0f, 40.0f}}, torch::requires_grad());
    const float infinity = std::numeric_limits<float>::infinity();
    auto target = torch::tensor({{infinity, infinity, infinity, infinity},
                                {infinity, infinity, infinity, infinity},
                                {1000.0f, 1000.0f, 1000.0f, 1000.0f}}, torch::requires_grad());
    const auto legal = torch::tensor({{false, false, false, false},
                                     {true, true, true, true},
                                     {false, false, false, false}}, torch::kBool);
    const auto result = double_dqn_targets(rewards, dones, online, target, legal, 0.99f);
    require(torch::isfinite(result).all().item<bool>(),
            "terminal targets: an all-illegal mask must not create infinity-times-zero NaNs");
    require(torch::equal(result, rewards.detach()),
            "terminal targets: terminal and no-legal-action states must contribute no bootstrap value");
    require(!result.requires_grad(), "terminal targets: labels must be detached from both networks and reward tensors");
}

void action_selection_excludes_illegal_directions() {
    DQNTrainer trainer(small_config());
    set_constant_outputs(trainer.policy_network(), {1000.0f, 2.0f, 900.0f, 1.0f});
    const ActionMask legal{false, true, false, true};
    const auto greedy = trainer.select_action(State{}, legal, 0.0f);
    require(greedy.action == 1 && !greedy.exploratory,
            "greedy action mask: illegal high-Q outputs must not beat the best legal direction");
    int seen_down = 0;
    int seen_right = 0;
    for (int sample = 0; sample < 100; ++sample) {
        const auto selection = trainer.select_action(State{}, legal, 1.0f);
        require(selection.exploratory, "exploration: epsilon one must label every selected action exploratory");
        require(selection.action == 1 || selection.action == 3,
                "exploration action mask: random choices must be restricted to legal directions");
        seen_down += selection.action == 1;
        seen_right += selection.action == 3;
    }
    require(seen_down > 0 && seen_right > 0, "exploration: seeded samples must cover both available directions");
}

void optimizer_learns_and_synchronizes_target_on_schedule() {
    auto config = small_config();
    config.learning_rate = 0.03;
    config.target_update_every = 75;
    DQNTrainer trainer(config);
    set_constant_outputs(trainer.policy_network(), {0.0f, 0.0f, 0.0f, 0.0f});
    trainer.sync_target();
    const auto initial_policy = parameters(trainer.policy_network());
    const auto initial_target = parameters(trainer.target_network());
    fill_terminal_replay(trainer);
    const float first_loss = trainer.optimize();
    float final_loss = first_loss;
    for (int update = 1; update < 74; ++update) final_loss = trainer.optimize();
    require(std::isfinite(first_loss) && std::isfinite(final_loss), "optimizer: losses must remain finite");
    require(first_loss > 0.0f && final_loss < first_loss * 0.02f,
            "optimizer: repeated fitting of a fixed terminal minibatch must substantially reduce its error");
    require(!parameters_equal(trainer.policy_network(), initial_policy),
            "optimizer: actual policy parameters must change after backward and Adam steps");
    require(parameters_equal(trainer.target_network(), initial_target),
            "target schedule: target parameters must stay fixed before the configured update interval");
    require(trainer.updates() == 74, "optimizer counter: each successful minibatch fit must count once");
    trainer.optimize();
    require(trainer.updates() == 75, "target schedule: expected the configured synchronization boundary");
    require(parameters_equal(trainer.target_network(), parameters(trainer.policy_network())),
            "target schedule: target must copy the updated policy exactly at the configured interval");
    for (const auto& parameter : trainer.policy_network()->parameters()) {
        require(torch::isfinite(parameter).all().item<bool>(), "optimizer: trained parameters must remain finite");
    }
}

void backward_updates_weights_through_every_layer() {
    DQNTrainer trainer(small_config());
    auto& network = trainer.policy_network();
    {
        torch::NoGradGuard guard;
        // Keep every ReLU active so a detached hidden layer cannot hide behind a bias-only fit.
        for (auto& parameter : network->parameters()) parameter.fill_(0.01f);
    }
    trainer.sync_target();
    const std::vector<torch::nn::Linear> layers{network->fc1, network->fc2, network->fc3, network->fc4};
    std::vector<torch::Tensor> before;
    for (const auto& layer : layers) before.push_back(layer->weight.detach().clone());
    auto transition = terminal_experience();
    transition.state.fill(0.25f);
    for (size_t index = 0; index < trainer.config().batch_size; ++index) trainer.remember(transition);
    require(std::isfinite(trainer.optimize()), "full-network backward: optimizer loss must be finite");
    for (size_t index = 0; index < layers.size(); ++index) {
        const auto& weight = layers[index]->weight;
        const auto gradient = weight.grad();
        const std::string context = "full-network backward fc" + std::to_string(index + 1) + ": ";
        require(gradient.defined(), context + "weight gradients must reach this layer");
        require(torch::isfinite(gradient).all().item<bool>() && gradient.abs().sum().item<float>() > 0,
                context + "weight gradients must be finite and nonzero");
        require(!torch::equal(weight, before[index]), context + "Adam must update this layer's weights");
    }
}

void completed_episode_counts_and_stop_callback() {
    auto config = small_config();
    config.exploration_steps = 20;
    DQNTrainer trainer(config);
    std::vector<int> episode_ids;
    TrainingCallbacks callbacks;
    callbacks.on_episode = [&](const EpisodeMetrics& metrics) { episode_ids.push_back(metrics.episode); };
    trainer.train(2, callbacks);
    require(trainer.completed_episodes() == 2 && trainer.total_steps() == 10,
            "training counters: two five-step episodes must count as two completed episodes and ten transitions");
    require(episode_ids == std::vector<int>({1, 2}),
            "episode numbering: completed episode reports should begin at one without duplicates");
    require(trainer.updates() > 0 && trainer.replay_size() == 10,
            "training integration: played transitions must enter replay and trigger optimizer updates");
    const float expected_epsilon = config.epsilon_start +
        (config.epsilon_end - config.epsilon_start) * 0.5f;
    require(std::abs(trainer.epsilon() - expected_epsilon) < 1e-6f,
            "exploration schedule: epsilon must decay by transitions, not by completed episodes");
    trainer.train(1, callbacks);
    require(trainer.completed_episodes() == 3 && episode_ids.back() == 3,
            "training continuation: another train call must continue numbering instead of repeating the last episode");
    const auto before_steps = trainer.total_steps();
    callbacks.should_stop = [&] { return trainer.total_steps() >= before_steps + 3; };
    trainer.train(4, callbacks);
    require(trainer.total_steps() == before_steps + 3 && trainer.completed_episodes() == 3,
            "headless stop callback: stop promptly and do not count an interrupted episode as complete");
}

void checkpoint_restores_predictions_state_and_adam_continuation() {
    auto config = small_config();
    config.learning_rate = 0.003;
    config.gamma = 0.9f;
    config.epsilon_start = 0.8f;
    config.epsilon_end = 0.05f;
    config.exploration_steps = 250;
    DQNTrainer original(config);
    original.train(2);
    fill_terminal_replay(original);
    for (int update = 0; update < 5; ++update) original.optimize();
    const auto saved_policy = parameters(original.policy_network());
    const auto saved_target = parameters(original.target_network());
    TemporaryCheckpoint checkpoint("resume");
    original.save(checkpoint.path);

    DQNTrainer restored;
    require(restored.load(checkpoint.path), "checkpoint: the freshly saved archive must load successfully");
    require(parameters_equal(restored.policy_network(), saved_policy), "checkpoint: policy weights must round-trip exactly");
    require(parameters_equal(restored.target_network(), saved_target),
            "checkpoint: the lagged target must be restored rather than replaced with the policy");
    require(restored.total_steps() == original.total_steps() && restored.updates() == original.updates() &&
            restored.completed_episodes() == original.completed_episodes() && restored.epsilon() == original.epsilon(),
            "checkpoint: transition, optimizer, episode, and exploration progress must be preserved");
    const auto& loaded = restored.config();
    require(loaded.learning_rate == config.learning_rate && loaded.gamma == config.gamma &&
            loaded.epsilon_start == config.epsilon_start && loaded.epsilon_end == config.epsilon_end &&
            loaded.exploration_steps == config.exploration_steps && loaded.replay_capacity == config.replay_capacity &&
            loaded.batch_size == config.batch_size && loaded.train_every == config.train_every &&
            loaded.target_update_every == config.target_update_every && loaded.max_episode_steps == config.max_episode_steps &&
            loaded.seed == config.seed, "checkpoint: saved training configuration must replace constructor defaults");
    require(restored.replay_size() == 0, "checkpoint: resume intentionally starts with an empty replay warmup");
    const State state = terminal_experience().state;
    const ActionMask legal{true, true, true, true};
    require(original.snapshot(state, legal).q_values == restored.snapshot(state, legal).q_values,
            "checkpoint: policy predictions must match after loading");
    for (int sample = 0; sample < 20; ++sample) {
        require(original.select_action(state, legal, 1.0f).action == restored.select_action(state, legal, 1.0f).action,
                "checkpoint RNG: exploration must continue from the saved random generator state");
    }
    fill_terminal_replay(restored);
    const float original_loss = original.optimize();
    const float restored_loss = restored.optimize();
    require(std::abs(original_loss - restored_loss) < 1e-7f,
            "checkpoint Adam continuation: identical restored batches must begin at the same loss");
    const auto continued_policy = parameters(original.policy_network());
    for (size_t index = 0; index < continued_policy.size(); ++index) {
        require(torch::allclose(restored.policy_network()->parameters()[index], continued_policy[index], 1e-6, 1e-7),
                "checkpoint Adam continuation: next update differs at parameter " + std::to_string(index));
    }
}

void checkpoint_distinguishes_missing_corrupt_and_legacy_models() {
    DQNTrainer trainer(small_config());
    TemporaryCheckpoint missing("missing");
    require(!trainer.load(missing.path), "missing checkpoint: an absent file should report false");

    TemporaryCheckpoint corrupt("corrupt");
    {
        std::ofstream output(corrupt.path, std::ios::binary);
        output << "not a torch archive";
    }
    require_throws([&] { trainer.load(corrupt.path); },
                   "corrupt checkpoint: malformed data must throw instead of silently claiming a fresh start");

    TemporaryCheckpoint legacy("legacy");
    set_constant_outputs(trainer.policy_network(), {1.0f, 2.0f, 3.0f, 4.0f});
    torch::serialize::OutputArchive archive;
    trainer.policy_network()->save(archive);
    archive.save_to(legacy.path);
    DQNTrainer warm_start(small_config());
    require(warm_start.load(legacy.path), "legacy checkpoint: a network-only archive should support a weights warm start");
    require(parameters_equal(warm_start.policy_network(), parameters(trainer.policy_network())),
            "legacy checkpoint: saved policy parameters must be preserved");
    require(parameters_equal(warm_start.target_network(), parameters(warm_start.policy_network())),
            "legacy checkpoint: target must initialize from loaded policy weights");
    require(warm_start.total_steps() == 0 && warm_start.updates() == 0 && warm_start.completed_episodes() == 0,
            "legacy checkpoint: missing optimizer/training state must not imply a full training resume");
}

void evaluation_is_deterministic_and_does_not_train() {
    auto config = small_config();
    config.max_episode_steps = 20;
    DQNTrainer trainer(config);
    trainer.train(1);
    const auto policy_before = parameters(trainer.policy_network());
    const auto target_before = parameters(trainer.target_network());
    const auto steps_before = trainer.total_steps();
    const auto updates_before = trainer.updates();
    const auto episodes_before = trainer.completed_episodes();
    const auto replay_before = trainer.replay_size();
    const auto epsilon_before = trainer.epsilon();
    TemporaryCheckpoint checkpoint("evaluation");
    trainer.save(checkpoint.path);
    DQNTrainer untouched(config);
    require(untouched.load(checkpoint.path), "evaluation fixture: saved comparison trainer should load");

    const auto first = trainer.evaluate(5, 4321);
    const auto repeated = trainer.evaluate(5, 4321);
    require(first.games.size() == 5 && repeated.games.size() == 5,
            "evaluation: return every requested game");
    for (size_t index = 0; index < first.games.size(); ++index) {
        const auto& a = first.games[index];
        const auto& b = repeated.games[index];
        require(a.score == b.score && a.max_tile == b.max_tile && a.steps == b.steps && a.truncated == b.truncated,
                "evaluation seed: repeated greedy game " + std::to_string(index) + " must reproduce exactly");
        require(a.epsilon == 0.0f && b.epsilon == 0.0f,
                "greedy evaluation metrics: random-action probability must be reported as zero");
    }
    const auto random_first = trainer.evaluate(5, 9876, true);
    const auto random_repeated = trainer.evaluate(5, 9876, true);
    require(random_first.games.size() == 5 && random_repeated.games.size() == 5,
            "random evaluation: return every requested baseline game");
    for (size_t index = 0; index < random_first.games.size(); ++index) {
        const auto& a = random_first.games[index];
        const auto& b = random_repeated.games[index];
        require(a.score == b.score && a.max_tile == b.max_tile && a.steps == b.steps,
                "random baseline seed: repeated game " + std::to_string(index) + " must reproduce exactly");
        require(a.epsilon == 1.0f && b.epsilon == 1.0f,
                "random evaluation metrics: random-action probability must be reported as one");
    }
    require(parameters_equal(trainer.policy_network(), policy_before) && parameters_equal(trainer.target_network(), target_before),
            "evaluation immutability: neither network should change");
    require(trainer.total_steps() == steps_before && trainer.updates() == updates_before &&
            trainer.completed_episodes() == episodes_before && trainer.replay_size() == replay_before &&
            trainer.epsilon() == epsilon_before, "evaluation immutability: training progress and replay should remain unchanged");
    const ActionMask legal{true, true, true, true};
    for (int sample = 0; sample < 20; ++sample) {
        require(trainer.select_action(State{}, legal, 1.0f).action == untouched.select_action(State{}, legal, 1.0f).action,
                "evaluation immutability: evaluation must not consume the training random generator");
    }
}

} // namespace

int main() {
    try {
        torch::set_num_threads(1);
        double_dqn_uses_online_selection_and_target_values();
        terminal_targets_are_finite_and_detached();
        action_selection_excludes_illegal_directions();
        optimizer_learns_and_synchronizes_target_on_schedule();
        backward_updates_weights_through_every_layer();
        completed_episode_counts_and_stop_callback();
        checkpoint_restores_predictions_state_and_adam_continuation();
        checkpoint_distinguishes_missing_corrupt_and_legacy_models();
        evaluation_is_deterministic_and_does_not_train();
        std::cout << "All training tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Training test failed: " << error.what() << '\n';
        return 1;
    }
}
