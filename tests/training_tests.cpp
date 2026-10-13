#include "training.cpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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

void n_step_targets_use_each_transitions_actual_horizon() {
    const auto rewards = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});
    const auto dones = torch::tensor({false, false, false, true}, torch::kBool);
    const auto online = torch::tensor({{1.0f, 4.0f, 3.0f, 99.0f},
                                      {1.0f, 4.0f, 3.0f, 99.0f},
                                      {1.0f, 4.0f, 3.0f, 99.0f},
                                      {1.0f, 4.0f, 3.0f, 99.0f}});
    const auto target = torch::tensor({{50.0f, 8.0f, 70.0f, 1000.0f},
                                      {50.0f, 8.0f, 70.0f, 1000.0f},
                                      {50.0f, 8.0f, 70.0f, 1000.0f},
                                      {50.0f, 8.0f, 70.0f, 1000.0f}});
    const auto legal = torch::tensor({{true, true, true, false},
                                     {true, true, true, false},
                                     {true, true, true, false},
                                     {true, true, true, false}}, torch::kBool);
    const auto horizons = torch::tensor({1, 2, 3, 2}, torch::kLong);
    const auto result = double_dqn_targets(rewards, dones, online, target, legal, 0.5f, horizons);
    require(torch::allclose(result, torch::tensor({5.0f, 4.0f, 4.0f, 4.0f})),
            "n-step targets: discount the selected legal target by gamma^actual_horizon, with zero terminal bootstrap");
}

Experience sequential_transition(int index, float reward, bool done = false) {
    Experience transition{};
    transition.state[0] = static_cast<float>(index) / 17.0f;
    transition.action = index % kActionCount;
    transition.reward = reward;
    transition.next_state[0] = static_cast<float>(index + 1) / 17.0f;
    if (!done) transition.next_valid[(index + 1) % kActionCount] = true;
    transition.done = done;
    return transition;
}

void n_step_accumulates_discounted_rewards_and_final_bootstrap_state() {
    NStepAccumulator accumulator(3, 0.5f);
    const auto first = sequential_transition(1, 1.0f);
    const auto second = sequential_transition(2, 2.0f);
    const auto third = sequential_transition(3, 3.0f);
    require(accumulator.push(first).empty(), "n-step accumulation: wait for all three transitions before emitting");
    require(accumulator.push(second).empty(), "n-step accumulation: two transitions are insufficient for a three-step prefix");
    const auto emitted = accumulator.push(third);
    require(emitted.size() == 1, "n-step accumulation: emit exactly the oldest full prefix");
    const auto& result = emitted.front();
    require(std::abs(result.reward - 2.75f) < 1e-7f && result.horizon == 3,
            "n-step accumulation: expected reward 1 + 0.5*2 + 0.25*3 and horizon three");
    require(result.state == first.state && result.action == first.action,
            "n-step accumulation: retain the first transition's state and action");
    require(result.next_state == third.next_state && result.next_valid == third.next_valid && !result.done,
            "n-step accumulation: bootstrap from the final transition's next state and legal actions");
}

void n_step_terminal_flush_emits_every_tail_with_its_actual_horizon() {
    NStepAccumulator accumulator(3, 0.5f);
    accumulator.push(sequential_transition(1, 1.0f));
    accumulator.push(sequential_transition(2, 2.0f));
    const auto terminal = sequential_transition(3, 3.0f, true);
    const auto emitted = accumulator.push(terminal);
    require(emitted.size() == 3, "n-step terminal flush: retain the full prefix and both shorter terminal tails");
    const std::array<float, 3> expected_rewards{2.75f, 3.5f, 3.0f};
    for (size_t index = 0; index < emitted.size(); ++index) {
        const auto& result = emitted[index];
        const auto context = "n-step terminal tail " + std::to_string(index) + ": ";
        require(result.done && result.horizon == 3 - static_cast<int>(index),
                context + "mark terminal and retain the actual remaining horizon");
        require(std::abs(result.reward - expected_rewards[index]) < 1e-7f,
                context + "aggregate only this tail's discounted rewards");
        require(result.next_state == terminal.next_state && result.next_valid == terminal.next_valid,
                context + "end at the common terminal state with no legal actions");
    }
    require(accumulator.flush().empty(), "n-step terminal flush: no completed-game transitions may remain pending");
}

void n_step_short_terminal_games_do_not_mix_with_the_next_game() {
    NStepAccumulator accumulator(3, 0.5f);
    accumulator.push(sequential_transition(1, 2.0f));
    const auto first_game = accumulator.push(sequential_transition(2, 4.0f, true));
    require(first_game.size() == 2 && first_game[0].horizon == 2 && first_game[1].horizon == 1,
            "short n-step game: terminal episodes shorter than n must emit both actual horizons");
    require(first_game[0].reward == 4.0f && first_game[1].reward == 4.0f &&
            first_game[0].done && first_game[1].done,
            "short n-step game: expected terminal rewards 2 + 0.5*4 and 4");
    const auto next = sequential_transition(8, 7.0f, true);
    const auto second_game = accumulator.push(next);
    require(second_game.size() == 1 && second_game[0].reward == 7.0f && second_game[0].horizon == 1 &&
            second_game[0].state == next.state && second_game[0].next_state == next.next_state && second_game[0].done,
            "n-step episode boundary: the next game's reward and states must not include any prior-game tail");
}

void n_step_nonterminal_flush_preserves_truncation_bootstrap() {
    NStepAccumulator accumulator(4, 0.5f);
    const auto first = sequential_transition(1, 2.0f);
    const auto last = sequential_transition(2, 4.0f);
    accumulator.push(first);
    accumulator.push(last);
    const auto emitted = accumulator.flush();
    require(emitted.size() == 2 && emitted[0].horizon == 2 && emitted[1].horizon == 1,
            "n-step truncation flush: emit each pending prefix with its actual horizon");
    require(emitted[0].reward == 4.0f && emitted[1].reward == 4.0f,
            "n-step truncation flush: expected discounted tail rewards 2 + 0.5*4 and 4");
    for (size_t index = 0; index < emitted.size(); ++index) {
        require(!emitted[index].done && emitted[index].next_state == last.next_state &&
                emitted[index].next_valid == last.next_valid,
                "n-step truncated tail " + std::to_string(index) + ": preserve nonterminal state and legal bootstrap actions");
    }
    require(accumulator.flush().empty(), "n-step truncation flush: a second flush must contain no duplicate tails");
    const auto next = sequential_transition(8, 8.0f);
    accumulator.push(next);
    const auto next_tail = accumulator.flush();
    require(next_tail.size() == 1 && next_tail[0].state == next.state && next_tail[0].reward == 8.0f &&
            next_tail[0].horizon == 1,
            "n-step truncation boundary: a new game must not inherit flushed transitions");
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

void n_step_training_keeps_move_limit_tails_in_replay() {
    auto config = small_config();
    config.n_step = 3;
    DQNTrainer trainer(config);
    std::vector<EpisodeMetrics> games;
    TrainingCallbacks callbacks;
    callbacks.on_episode = [&](const EpisodeMetrics& game) { games.push_back(game); };
    trainer.train(2, callbacks);
    require(trainer.completed_episodes() == 2 && trainer.total_steps() == 10 && trainer.replay_size() == 10,
            "n-step training integration: retain all ten transitions, including both games' shorter move-limit tails");
    require(trainer.updates() > 0, "n-step training integration: accumulated returns must reach the optimizer");
    require(games.size() == 2 && games[0].steps == 5 && games[1].steps == 5 &&
            games[0].truncated && games[1].truncated,
            "n-step training integration: flush pending returns at each nonterminal five-move boundary");
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

void learning_configuration_preserves_weights_counters_and_adam_state() {
    auto config = small_config();
    config.learning_rate = 0.003;
    DQNTrainer original(config);
    original.train(2);
    fill_terminal_replay(original);
    for (int update = 0; update < 5; ++update) original.optimize();
    require_throws([&] { original.configure_learning(0.995f, 3, false, 8, 500); },
                   "learning configuration: reject retuning a trainer whose replay still contains old-objective transitions");
    require(original.config().gamma == config.gamma && original.config().n_step == 1 &&
            original.config().train_every == config.train_every,
            "learning configuration: a rejected replay retune must preserve the previous settings");

    TemporaryCheckpoint checkpoint("retune");
    original.save(checkpoint.path);
    DQNTrainer restored;
    require(restored.load(checkpoint.path), "learning configuration fixture: resume checkpoint into an empty replay");
    require_throws([&] { restored.configure_learning(1.1f, 3, false, 8, 500); },
                   "learning configuration: reject gamma outside the valid range");
    require_throws([&] { restored.configure_learning(0.995f, 0, false, 8, 500); },
                   "learning configuration: reject a zero-step return horizon");
    require_throws([&] { restored.configure_learning(0.995f, 33, false, 8, 500); },
                   "learning configuration: reject a return horizon above the supported maximum of thirty-two");
    require_throws([&] { restored.configure_learning(0.995f, 3, false, 0, 500); },
                   "learning configuration: reject a zero optimizer interval");
    require_throws([&] { restored.configure_learning(0.995f, 3, false, 8, 0); },
                   "learning configuration: reject a zero target synchronization interval");
    restored.configure_learning(0.995f, 3, false, 8, 500);
    require(restored.config().gamma == 0.995f && restored.config().n_step == 3 &&
            !restored.config().symmetry_augmentation && restored.config().train_every == 8 &&
            restored.config().target_update_every == 500,
            "learning configuration: apply the requested settings to a resumed trainer with empty replay");
    require(parameters_equal(restored.policy_network(), parameters(original.policy_network())) &&
            parameters_equal(restored.target_network(), parameters(original.target_network())),
            "learning configuration: retuning must preserve both learned networks exactly");
    require(restored.total_steps() == original.total_steps() && restored.updates() == original.updates() &&
            restored.completed_episodes() == original.completed_episodes() && restored.epsilon() == original.epsilon(),
            "learning configuration: retuning must preserve episode, transition, optimizer, and exploration progress");
    fill_terminal_replay(restored);
    const auto original_loss = original.optimize();
    const auto restored_loss = restored.optimize();
    require(std::abs(original_loss - restored_loss) < 1e-7f,
            "learning configuration Adam: equivalent terminal minibatches must still start at the same loss");
    const auto continued = parameters(original.policy_network());
    for (size_t index = 0; index < continued.size(); ++index) {
        require(torch::allclose(restored.policy_network()->parameters()[index], continued[index], 1e-6, 1e-7),
                "learning configuration Adam: the next parameter update must retain optimizer state at parameter " +
                std::to_string(index));
    }
}

void tuned_learning_options_round_trip_in_checkpoints() {
    auto config = small_config();
    config.gamma = 0.995f;
    config.n_step = 3;
    config.symmetry_augmentation = true;
    config.train_every = 2;
    config.target_update_every = 17;
    DQNTrainer original(config);
    original.train(2);
    TemporaryCheckpoint checkpoint("tuned-options");
    original.save(checkpoint.path);
    DQNTrainer restored;
    require(restored.load(checkpoint.path), "tuned checkpoint: saved archive must load");
    const auto& saved = restored.config();
    require(saved.gamma == config.gamma && saved.n_step == config.n_step &&
            saved.symmetry_augmentation == config.symmetry_augmentation &&
            saved.train_every == config.train_every && saved.target_update_every == config.target_update_every,
            "tuned checkpoint: preserve discount, return horizon, symmetry, optimizer interval, and target interval");
    require(restored.total_steps() == original.total_steps() && restored.updates() == original.updates() &&
            restored.completed_episodes() == original.completed_episodes() && restored.replay_size() == 0,
            "tuned checkpoint: retain completed training progress and restart replay warmup");
    require(parameters_equal(restored.policy_network(), parameters(original.policy_network())) &&
            parameters_equal(restored.target_network(), parameters(original.target_network())),
            "tuned checkpoint: both networks must retain their exact trained parameters");
}

void original_checkpoint_config_defaults_to_one_step_without_symmetry() {
    const auto config = small_config();
    DQNTrainer original(config);
    torch::serialize::OutputArchive archive, policy, target, optimizer;
    original.policy_network()->save(policy);
    original.target_network()->save(target);
    torch::optim::Adam initial_optimizer(original.policy_network()->parameters(),
                                        torch::optim::AdamOptions(config.learning_rate));
    initial_optimizer.save(optimizer);
    archive.write("version", c10::IValue(int64_t{1}));
    archive.write("policy", policy);
    archive.write("target", target);
    archive.write("optimizer", optimizer);
    archive.write("steps", c10::IValue(int64_t{0}));
    archive.write("updates", c10::IValue(int64_t{0}));
    archive.write("episodes", c10::IValue(int64_t{0}));
    std::ostringstream legacy_config, rng;
    legacy_config << std::setprecision(std::numeric_limits<double>::max_digits10)
                  << config.learning_rate << ' ' << config.gamma << ' ' << config.epsilon_start << ' '
                  << config.epsilon_end << ' ' << config.exploration_steps << ' ' << config.replay_capacity << ' '
                  << config.batch_size << ' ' << config.train_every << ' ' << config.target_update_every << ' '
                  << config.max_episode_steps << ' ' << config.seed;
    rng << std::mt19937(config.seed);
    archive.write("config", c10::IValue(legacy_config.str()));
    archive.write("rng", c10::IValue(rng.str()));
    archive.write("milestones", c10::IValue(std::string{}));
    TemporaryCheckpoint checkpoint("original-config");
    archive.save_to(checkpoint.path);
    auto tuned_config = small_config();
    tuned_config.n_step = 3;
    tuned_config.symmetry_augmentation = true;
    DQNTrainer restored(tuned_config);
    require(restored.load(checkpoint.path), "original checkpoint: the eleven-field configuration archive must remain loadable");
    require(restored.config().n_step == 1 && !restored.config().symmetry_augmentation &&
            restored.config().gamma == config.gamma && restored.config().seed == config.seed,
            "original checkpoint: missing new fields must restore one-step, unaugmented training rather than constructor tuning");
    require(parameters_equal(restored.policy_network(), parameters(original.policy_network())) &&
            parameters_equal(restored.target_network(), parameters(original.target_network())),
            "original checkpoint: compatibility loading must preserve both networks");
    fill_terminal_replay(restored);
    require(std::isfinite(restored.optimize()), "original checkpoint: the restored optimizer must still train successfully");
}

void symmetry_augmentation_trains_all_transformed_action_outputs() {
    auto config = small_config();
    config.batch_size = 64;
    config.replay_capacity = 64;
    config.learning_rate = 0.03;
    config.symmetry_augmentation = true;
    config.target_update_every = 1000;
    DQNTrainer trainer(config);
    set_constant_outputs(trainer.policy_network(), {0.0f, 0.0f, 0.0f, 0.0f});
    trainer.sync_target();
    fill_terminal_replay(trainer);
    const auto first_loss = trainer.optimize();
    auto final_loss = first_loss;
    for (int update = 1; update < 45; ++update) final_loss = trainer.optimize();
    require(std::isfinite(first_loss) && std::isfinite(final_loss) && final_loss < first_loss * 0.1f,
            "symmetry optimization: fitting augmented terminal transitions must reduce loss without non-finite values");
    const auto outputs = trainer.snapshot(terminal_experience().state, ActionMask{true, true, true, true}).q_values;
    for (int action = 0; action < kActionCount; ++action) {
        require(outputs[action] > 0.7f && outputs[action] < 1.3f,
                "symmetry optimization action " + std::to_string(action) +
                ": a single stored direction must train each symmetry-equivalent action output toward reward one");
        ActionMask legal{};
        legal[action] = true;
        require(trainer.select_action(terminal_experience().state, legal, 0.0f).action == action,
                "symmetry trained policy action " + std::to_string(action) + ": greedy selection must remain legal");
    }
}

Board mixed_terminal_spawn_board() {
    return Board{{{{2, 2, 8, 2}}, {{8, 16, 32, 64}},
                  {{16, 32, 64, 128}}, {{32, 64, 128, 256}}}};
}

void set_position_sensitive_outputs(QNet& network) {
    torch::NoGradGuard guard;
    for (auto& parameter : network->parameters()) parameter.zero_();
    for (int cell = 0; cell < kInputSize; ++cell) {
        network->fc1->weight.select(0, cell).select(0, cell).fill_(1.0f);
        network->fc2->weight.select(0, cell).select(0, cell).fill_(1.0f);
        network->fc3->weight.select(0, cell).select(0, cell).fill_(1.0f);
        for (int action = 0; action < kActionCount; ++action)
            network->fc4->weight.select(0, action).select(0, cell).fill_(
                0.1f * static_cast<float>((action + 1) * (cell + 1)));
    }
}

std::array<double, kActionCount> scalar_spawn_lookahead(DQNTrainer& trainer, const GameEngine& game) {
    torch::NoGradGuard guard;
    std::array<double, kActionCount> values;
    values.fill(-std::numeric_limits<double>::infinity());
    const auto legal = game.valid_actions();
    for (int action = 0; action < kActionCount; ++action) {
        if (!legal[action]) continue;
        auto moved = game;
        require(moved.move(static_cast<Direction>(action)), "scalar lookahead fixture: root action must change the board");
        values[action] = static_cast<double>(moved.score() - game.score()) / 100.0;
        const auto after = moved.board();
        std::vector<std::pair<int, int>> empty;
        for (int row = 0; row < kGridSize; ++row)
            for (int column = 0; column < kGridSize; ++column)
                if (!after[row][column]) empty.emplace_back(row, column);
        for (const auto& [row, column] : empty) {
            for (int tile : {2, 4}) {
                auto board = after;
                board[row][column] = tile;
                auto leaf = moved;
                leaf.set_board(board);
                const auto next_legal = leaf.valid_actions();
                if (std::none_of(next_legal.begin(), next_legal.end(), [](bool allowed) { return allowed; })) continue;
                auto state = leaf.state();
                const auto input = torch::from_blob(state.data(), {1, kInputSize}, torch::kFloat32).clone();
                const auto outputs = trainer.policy_network()->forward(input).contiguous();
                const auto* q = outputs.data_ptr<float>();
                float best = -std::numeric_limits<float>::infinity();
                for (int next_action = 0; next_action < kActionCount; ++next_action)
                    if (next_legal[next_action]) best = std::max(best, q[next_action]);
                const auto probability = (tile == 2 ? 0.9 : 0.1) / static_cast<double>(empty.size());
                values[action] += static_cast<double>(trainer.config().gamma) * probability * best;
            }
        }
    }
    return values;
}

void lookahead_batch_matches_independent_scalar_spawn_enumeration() {
    auto config = small_config();
    config.gamma = 0.95f;
    DQNTrainer trainer(config);
    set_position_sensitive_outputs(trainer.policy_network());
    GameEngine game(73);
    game.set_board(Board{{{{2, 2, 4, 0}}, {{16, 8, 2, 0}},
                         {{0, 4, 8, 16}}, {{2, 0, 0, 4}}}}, 1234);
    const auto before = game.board();
    const auto expected = scalar_spawn_lookahead(trainer, game);
    const auto actual = trainer.lookahead_values(game);
    for (int action = 0; action < kActionCount; ++action) {
        require(std::isfinite(expected[action]) && std::isfinite(actual[action]) &&
                std::abs(actual[action] - expected[action]) < 1e-5,
                "lookahead action " + std::to_string(action) +
                ": batched values must match independently enumerated uniform cells, 90/10 tile probabilities, and legal scalar Q values");
    }
    require(game.board() == before && game.score() == 1234 && game.moves() == 0,
            "lookahead analysis: value calculation must leave the source board, score, and move count unchanged");
}

void lookahead_constant_future_adds_only_immediate_merge_reward() {
    auto config = small_config();
    config.gamma = 0.5f;
    DQNTrainer trainer(config);
    set_constant_outputs(trainer.policy_network(), {10.0f, 10.0f, 10.0f, 10.0f});
    GameEngine game(73);
    game.set_board(Board{{{{2, 2, 0, 0}}, {{0, 0, 0, 0}},
                         {{0, 0, 0, 0}}, {{0, 0, 0, 0}}}}, 1000);
    const auto values = trainer.lookahead_values(game);
    require(std::isinf(values[0]) && values[0] < 0,
            "lookahead root mask: an unchanged upward move must have negative-infinity value");
    require(std::abs(values[1] - 5.0) < 1e-12 &&
            std::abs(values[2] - 5.04) < 1e-12 && std::abs(values[3] - 5.04) < 1e-12,
            "lookahead constant future: use gamma*10 plus this move's merge gain/100, without adding the accumulated game score");
}

void lookahead_terminal_spawn_probability_is_not_renormalized() {
    auto config = small_config();
    config.gamma = 0.5f;
    DQNTrainer trainer(config);
    set_constant_outputs(trainer.policy_network(), {1000.0f, 2000.0f, 4.0f, 6.0f});
    GameEngine game(73);
    game.set_board(mixed_terminal_spawn_board());
    auto left = game;
    require(left.move(Direction::Left), "terminal spawn fixture: left must merge the initial pair");
    auto spawn_two = left.board();
    spawn_two[0][3] = 2;
    left.set_board(spawn_two);
    require(left.valid_actions() == ActionMask{false, false, true, true},
            "terminal spawn fixture: a two tile must leave only horizontal moves legal");
    auto spawn_four = spawn_two;
    spawn_four[0][3] = 4;
    left.set_board(spawn_four);
    require(left.game_over(), "terminal spawn fixture: a four tile must end the game");
    const auto values = trainer.lookahead_values(game);
    require(std::isinf(values[0]) && values[0] < 0 && std::isinf(values[1]) && values[1] < 0,
            "terminal spawn lookahead: illegal vertical root moves must remain masked");
    require(std::abs(values[2] - (0.04 + 0.5 * 0.9 * 6.0)) < 1e-12,
            "terminal spawn lookahead left: keep the terminal outcome's ten-percent mass and exclude illegal high vertical Q values");
    require(std::abs(values[3] - (0.04 + 0.5 * 0.1 * 6.0)) < 1e-12,
            "terminal spawn lookahead right: keep the terminal outcome's ninety-percent mass without renormalizing surviving outcomes");
}

void disabled_lookahead_selection_preserves_direct_policy_and_rng_sequence() {
    const auto config = small_config();
    DQNTrainer trainer(config);
    DQNTrainer direct(config);
    set_constant_outputs(trainer.policy_network(), {1000.0f, 2000.0f, 4.0f, 6.0f});
    set_constant_outputs(direct.policy_network(), {1000.0f, 2000.0f, 4.0f, 6.0f});
    GameEngine game(73);
    game.set_board(mixed_terminal_spawn_board());
    require(!trainer.config().lookahead, "lookahead defaults: new trainers must retain direct action selection until enabled");
    const auto first = trainer.select_action(game, 0.0f);
    const auto expected_first = direct.select_action(game.state(), game.valid_actions(), 0.0f);
    require(first.action == 3 && first.action == expected_first.action && !first.exploratory,
            "disabled lookahead: the game overload must select the direct network's highest legal Q action");
    for (int sample = 0; sample < 40; ++sample) {
        const auto actual = trainer.select_action(game, 0.4f);
        const auto expected = direct.select_action(game.state(), game.valid_actions(), 0.4f);
        require(actual.action == expected.action && actual.exploratory == expected.exploratory,
                "disabled lookahead RNG sample " + std::to_string(sample) + ": preserve the existing direct-policy random sequence");
    }
}

void enabled_lookahead_selects_expected_return_and_explores_legal_actions() {
    auto config = small_config();
    config.gamma = 0.5f;
    DQNTrainer trainer(config);
    set_constant_outputs(trainer.policy_network(), {1000.0f, 2000.0f, 4.0f, 6.0f});
    trainer.enable_lookahead(true);
    GameEngine game(73);
    game.set_board(mixed_terminal_spawn_board());
    const auto greedy = trainer.select_action(game, 0.0f);
    require(greedy.action == 2 && !greedy.exploratory,
            "enabled lookahead: choose the left move's larger expected return even when direct Q prefers right");
    int seen_left = 0;
    int seen_right = 0;
    for (int sample = 0; sample < 100; ++sample) {
        const auto selected = trainer.select_action(game, 1.0f);
        require(selected.exploratory && (selected.action == 2 || selected.action == 3),
                "lookahead exploration sample " + std::to_string(sample) + ": epsilon one must still choose only legal random moves");
        seen_left += selected.action == 2;
        seen_right += selected.action == 3;
    }
    require(seen_left && seen_right, "lookahead exploration: fixed-seed samples must cover both available directions");
    require_throws([&] { trainer.select_action(game, -0.1f); }, "lookahead exploration: reject negative epsilon");
    require_throws([&] { trainer.select_action(game, 1.1f); }, "lookahead exploration: reject epsilon above one");
}

void lookahead_terminal_roots_have_no_selectable_actions() {
    DQNTrainer trainer(small_config());
    trainer.enable_lookahead(true);
    GameEngine game(73);
    game.set_board(Board{{{{2, 4, 2, 4}}, {{4, 2, 4, 2}},
                         {{2, 4, 2, 4}}, {{4, 2, 4, 2}}}});
    const auto values = trainer.lookahead_values(game);
    for (int action = 0; action < kActionCount; ++action)
        require(std::isinf(values[action]) && values[action] < 0,
                "terminal lookahead action " + std::to_string(action) + ": every root direction must be masked");
    require_throws([&] { trainer.select_action(game, 0.0f); }, "terminal lookahead: greedy action selection must reject a terminal board");
    require_throws([&] { trainer.select_action(game, 1.0f); }, "terminal lookahead: exploration must reject a terminal board");
}

void lookahead_evaluation_preserves_training_state_and_rng() {
    auto config = small_config();
    config.max_episode_steps = 20;
    config.lookahead = true;
    DQNTrainer trainer(config);
    trainer.train(1);
    const auto policy_before = parameters(trainer.policy_network());
    const auto target_before = parameters(trainer.target_network());
    const auto steps_before = trainer.total_steps();
    const auto updates_before = trainer.updates();
    const auto episodes_before = trainer.completed_episodes();
    const auto replay_before = trainer.replay_size();
    const auto epsilon_before = trainer.epsilon();
    TemporaryCheckpoint checkpoint("lookahead-evaluation");
    trainer.save(checkpoint.path);
    DQNTrainer untouched;
    require(untouched.load(checkpoint.path), "lookahead evaluation fixture: comparison checkpoint must load");
    const auto first = trainer.evaluate(5, 4321);
    const auto repeated = trainer.evaluate(5, 4321);
    require(first.games.size() == 5 && repeated.games.size() == 5,
            "lookahead evaluation: return all requested games");
    for (size_t index = 0; index < first.games.size(); ++index) {
        const auto& a = first.games[index];
        const auto& b = repeated.games[index];
        require(a.score == b.score && a.max_tile == b.max_tile && a.steps == b.steps && a.truncated == b.truncated,
                "lookahead evaluation game " + std::to_string(index) + ": repeated fixed seeds must reproduce exactly");
        require(a.epsilon == 0.0f && b.epsilon == 0.0f,
                "lookahead evaluation game " + std::to_string(index) + ": evaluation must report zero exploration");
    }
    require(parameters_equal(trainer.policy_network(), policy_before) &&
            parameters_equal(trainer.target_network(), target_before),
            "lookahead evaluation: neither learned network may change");
    require(trainer.total_steps() == steps_before && trainer.updates() == updates_before &&
            trainer.completed_episodes() == episodes_before && trainer.replay_size() == replay_before &&
            trainer.epsilon() == epsilon_before && trainer.config().lookahead,
            "lookahead evaluation: preserve counters, replay, exploration schedule, and search setting");
    GameEngine game(73);
    for (int sample = 0; sample < 20; ++sample)
        require(trainer.select_action(game, 1.0f).action == untouched.select_action(game, 1.0f).action,
                "lookahead evaluation RNG sample " + std::to_string(sample) + ": evaluation must not consume training randomness");
}

void lookahead_toggle_and_checkpoint_preserve_learning_state() {
    const auto config = small_config();
    DQNTrainer trainer(config);
    trainer.train(2);
    fill_terminal_replay(trainer);
    for (int update = 0; update < 5; ++update) trainer.optimize();
    TemporaryCheckpoint before("before-lookahead");
    trainer.save(before.path);
    DQNTrainer untouched;
    require(untouched.load(before.path) && !untouched.config().lookahead,
            "lookahead toggle fixture: a disabled checkpoint must restore disabled selection");
    const auto policy_before = parameters(trainer.policy_network());
    const auto target_before = parameters(trainer.target_network());
    const auto steps_before = trainer.total_steps();
    const auto updates_before = trainer.updates();
    const auto episodes_before = trainer.completed_episodes();
    const auto replay_before = trainer.replay_size();
    const auto epsilon_before = trainer.epsilon();
    trainer.enable_lookahead(true);
    require(trainer.config().lookahead && trainer.status().lookahead &&
            parameters_equal(trainer.policy_network(), policy_before) &&
            parameters_equal(trainer.target_network(), target_before),
            "lookahead toggle: enable the setting and display status without replacing either network");
    require(trainer.total_steps() == steps_before && trainer.updates() == updates_before &&
            trainer.completed_episodes() == episodes_before && trainer.replay_size() == replay_before &&
            trainer.epsilon() == epsilon_before,
            "lookahead toggle: preserve existing replay, counters, and exploration progress");
    GameEngine game(73);
    for (int sample = 0; sample < 20; ++sample)
        require(trainer.select_action(game, 1.0f).action == untouched.select_action(game, 1.0f).action,
                "lookahead toggle RNG sample " + std::to_string(sample) + ": enabling search must preserve training randomness");
    fill_terminal_replay(untouched);
    const auto actual_loss = trainer.optimize();
    const auto expected_loss = untouched.optimize();
    require(std::abs(actual_loss - expected_loss) < 1e-7f,
            "lookahead toggle Adam: equivalent terminal batches must retain the same loss");
    const auto continued = parameters(untouched.policy_network());
    for (size_t index = 0; index < continued.size(); ++index)
        require(torch::allclose(trainer.policy_network()->parameters()[index], continued[index], 1e-6, 1e-7),
                "lookahead toggle Adam: preserve the next parameter update at parameter " + std::to_string(index));
    TemporaryCheckpoint enabled("enabled-lookahead");
    trainer.save(enabled.path);
    DQNTrainer restored;
    require(restored.load(enabled.path) && restored.config().lookahead && restored.status().lookahead,
            "lookahead checkpoint: round-trip the enabled selection and display setting");
    require(parameters_equal(restored.policy_network(), parameters(trainer.policy_network())) &&
            parameters_equal(restored.target_network(), parameters(trainer.target_network())) &&
            restored.total_steps() == trainer.total_steps() && restored.updates() == trainer.updates() &&
            restored.completed_episodes() == trainer.completed_episodes() && restored.epsilon() == trainer.epsilon(),
            "lookahead checkpoint: restore both networks and completed learning progress exactly");
    const auto before_disable = parameters(trainer.policy_network());
    trainer.enable_lookahead(false);
    require(!trainer.config().lookahead && !trainer.status().lookahead &&
            parameters_equal(trainer.policy_network(), before_disable),
            "lookahead toggle: disabling search must retain the trained policy");
}

void pre_lookahead_checkpoint_preserves_n_step_and_symmetry_without_enabling_search() {
    auto config = small_config();
    config.gamma = 0.995f;
    config.n_step = 3;
    config.symmetry_augmentation = true;
    config.train_every = 2;
    config.target_update_every = 17;
    DQNTrainer original(config);
    torch::serialize::OutputArchive archive, policy, target, optimizer;
    original.policy_network()->save(policy);
    original.target_network()->save(target);
    torch::optim::Adam initial_optimizer(original.policy_network()->parameters(),
                                        torch::optim::AdamOptions(config.learning_rate));
    initial_optimizer.save(optimizer);
    archive.write("version", c10::IValue(int64_t{1}));
    archive.write("policy", policy);
    archive.write("target", target);
    archive.write("optimizer", optimizer);
    archive.write("steps", c10::IValue(int64_t{0}));
    archive.write("updates", c10::IValue(int64_t{0}));
    archive.write("episodes", c10::IValue(int64_t{0}));
    std::ostringstream previous_config, rng;
    previous_config << std::setprecision(std::numeric_limits<double>::max_digits10)
                    << config.learning_rate << ' ' << config.gamma << ' ' << config.epsilon_start << ' '
                    << config.epsilon_end << ' ' << config.exploration_steps << ' ' << config.replay_capacity << ' '
                    << config.batch_size << ' ' << config.train_every << ' ' << config.target_update_every << ' '
                    << config.max_episode_steps << ' ' << config.seed << ' ' << config.n_step << ' '
                    << config.symmetry_augmentation;
    rng << std::mt19937(config.seed);
    archive.write("config", c10::IValue(previous_config.str()));
    archive.write("rng", c10::IValue(rng.str()));
    archive.write("milestones", c10::IValue(std::string{}));
    TemporaryCheckpoint checkpoint("pre-lookahead-config");
    archive.save_to(checkpoint.path);
    auto constructor_config = small_config();
    constructor_config.lookahead = true;
    DQNTrainer restored(constructor_config);
    require(restored.load(checkpoint.path), "pre-lookahead checkpoint: the thirteen-field configuration must remain loadable");
    require(restored.config().gamma == config.gamma && restored.config().n_step == 3 &&
            restored.config().symmetry_augmentation && restored.config().train_every == 2 &&
            restored.config().target_update_every == 17,
            "pre-lookahead checkpoint: preserve nondefault discount, return horizon, symmetry, and update intervals");
    require(!restored.config().lookahead && !restored.status().lookahead,
            "pre-lookahead checkpoint: an absent search field must restore disabled search even when the constructor enabled it");
    require(parameters_equal(restored.policy_network(), parameters(original.policy_network())) &&
            parameters_equal(restored.target_network(), parameters(original.target_network())),
            "pre-lookahead checkpoint: compatibility loading must preserve both networks exactly");
}

void lookahead_analysis_preserves_actual_tile_spawn_rng() {
    DQNTrainer trainer(small_config());
    GameEngine game(9182);
    game.set_board(Board{{{{2, 2, 0, 0}}, {{0, 0, 0, 0}},
                         {{0, 0, 0, 0}}, {{0, 0, 0, 0}}}}, 1000);
    auto control = game;
    for (int step = 0; step < 10; ++step) {
        const auto legal = game.valid_actions();
        const auto first_legal = std::find(legal.begin(), legal.end(), true);
        require(first_legal != legal.end(), "lookahead spawn RNG fixture: a legal real step must remain available");
        const auto action = static_cast<int>(std::distance(legal.begin(), first_legal));
        trainer.lookahead_values(game);
        const auto actual = game.step(action);
        const auto expected = control.step(action);
        const auto context = "lookahead spawn RNG step " + std::to_string(step) + ": ";
        require(actual.moved == expected.moved && actual.score_gain == expected.score_gain && actual.done == expected.done,
                context + "analysis must not change the subsequent real move result");
        require(game.board() == control.board() && game.score() == control.score() && game.moves() == control.moves(),
                context + "the real spawned tile position/value, board, score, and move count must match an untouched control");
    }
}

void learning_rate_configuration_rejects_nonpositive_and_nonfinite_values() {
    DQNTrainer trainer(small_config());
    const auto initial_rate = trainer.config().learning_rate;
    const std::array<double, 4> invalid_rates{
        0.0, -0.1, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()};
    for (size_t index = 0; index < invalid_rates.size(); ++index) {
        const auto rate = invalid_rates[index];
        auto config = small_config();
        config.learning_rate = rate;
        const auto context = "invalid learning rate case " + std::to_string(index) + ": ";
        require_throws([&] { DQNTrainer invalid(config); }, context + "constructor must reject nonpositive or nonfinite rates");
        require_throws([&] { trainer.configure_learning_rate(rate); }, context + "a live override must reject the same invalid rate");
        require(trainer.config().learning_rate == initial_rate, context + "rejection must preserve the prior configured rate");
    }
}

void learning_rate_override_preserves_adam_and_scales_the_next_update() {
    auto config = small_config();
    config.learning_rate = 0.003;
    config.target_update_every = 1000;
    DQNTrainer original(config);
    set_constant_outputs(original.policy_network(), {0.0f, 0.0f, 0.0f, 0.0f});
    original.sync_target();
    original.train(2);
    fill_terminal_replay(original);
    for (int update = 0; update < 5; ++update) original.optimize();
    TemporaryCheckpoint checkpoint("before-rate-change");
    original.save(checkpoint.path);
    DQNTrainer reduced;
    require(reduced.load(checkpoint.path), "learning rate override fixture: a checkpoint with Adam history must load");
    fill_terminal_replay(reduced);
    const auto policy_before = parameters(reduced.policy_network());
    const auto target_before = parameters(reduced.target_network());
    const auto steps_before = reduced.total_steps();
    const auto updates_before = reduced.updates();
    const auto episodes_before = reduced.completed_episodes();
    const auto replay_before = reduced.replay_size();
    const auto epsilon_before = reduced.epsilon();
    const auto lower_rate = config.learning_rate * 0.1;
    reduced.configure_learning_rate(lower_rate);
    require(reduced.config().learning_rate == lower_rate &&
            parameters_equal(reduced.policy_network(), policy_before) &&
            parameters_equal(reduced.target_network(), target_before),
            "learning rate override: change the rate without replacing either learned network");
    require(reduced.total_steps() == steps_before && reduced.updates() == updates_before &&
            reduced.completed_episodes() == episodes_before && reduced.replay_size() == replay_before &&
            reduced.epsilon() == epsilon_before,
            "learning rate override: preserve transition, optimizer, episode, replay, and exploration progress");
    const ActionMask legal{true, true, true, true};
    for (int sample = 0; sample < 20; ++sample)
        require(original.select_action(State{}, legal, 1.0f).action == reduced.select_action(State{}, legal, 1.0f).action,
                "learning rate override RNG sample " + std::to_string(sample) + ": preserve the saved random generator state");
    const auto full_loss = original.optimize();
    const auto reduced_loss = reduced.optimize();
    require(std::abs(full_loss - reduced_loss) < 1e-7f,
            "learning rate override: identical terminal minibatches must retain the same loss before their updates");
    const auto full_parameters = parameters(original.policy_network());
    const auto reduced_parameters = parameters(reduced.policy_network());
    double full_change = 0;
    double reduced_change = 0;
    for (size_t index = 0; index < policy_before.size(); ++index) {
        const auto full_delta = full_parameters[index] - policy_before[index];
        const auto reduced_delta = reduced_parameters[index] - policy_before[index];
        require(torch::allclose(reduced_delta, full_delta * 0.1, 1e-4, 1e-8),
                "learning rate override parameter " + std::to_string(index) +
                ": preserved Adam moments must make a tenfold rate reduction scale the same update by one tenth");
        full_change += full_delta.abs().sum().item<double>();
        reduced_change += reduced_delta.abs().sum().item<double>();
    }
    require(full_change > 1e-5 && std::abs(reduced_change / full_change - 0.1) < 1e-5,
            "learning rate override: require a real parameter update with total change ratio one tenth within float precision");

    TemporaryCheckpoint configured("configured-rate");
    reduced.save(configured.path);
    DQNTrainer restored;
    require(restored.load(configured.path) && restored.config().learning_rate == lower_rate,
            "learning rate checkpoint: retain the explicitly configured rate on reload");
    fill_terminal_replay(restored);
    const auto continued_loss = reduced.optimize();
    const auto restored_loss = restored.optimize();
    require(std::abs(continued_loss - restored_loss) < 1e-7f,
            "learning rate checkpoint: restored training must start from the same loss");
    const auto continued = parameters(reduced.policy_network());
    for (size_t index = 0; index < continued.size(); ++index)
        require(torch::allclose(restored.policy_network()->parameters()[index], continued[index], 1e-6, 1e-7),
                "learning rate checkpoint parameter " + std::to_string(index) +
                ": restore the actual Adam learning rate and accumulated moments for the next update");
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
        n_step_targets_use_each_transitions_actual_horizon();
        n_step_accumulates_discounted_rewards_and_final_bootstrap_state();
        n_step_terminal_flush_emits_every_tail_with_its_actual_horizon();
        n_step_short_terminal_games_do_not_mix_with_the_next_game();
        n_step_nonterminal_flush_preserves_truncation_bootstrap();
        action_selection_excludes_illegal_directions();
        optimizer_learns_and_synchronizes_target_on_schedule();
        backward_updates_weights_through_every_layer();
        completed_episode_counts_and_stop_callback();
        n_step_training_keeps_move_limit_tails_in_replay();
        checkpoint_restores_predictions_state_and_adam_continuation();
        checkpoint_distinguishes_missing_corrupt_and_legacy_models();
        learning_configuration_preserves_weights_counters_and_adam_state();
        tuned_learning_options_round_trip_in_checkpoints();
        original_checkpoint_config_defaults_to_one_step_without_symmetry();
        symmetry_augmentation_trains_all_transformed_action_outputs();
        lookahead_batch_matches_independent_scalar_spawn_enumeration();
        lookahead_constant_future_adds_only_immediate_merge_reward();
        lookahead_terminal_spawn_probability_is_not_renormalized();
        disabled_lookahead_selection_preserves_direct_policy_and_rng_sequence();
        enabled_lookahead_selects_expected_return_and_explores_legal_actions();
        lookahead_terminal_roots_have_no_selectable_actions();
        lookahead_evaluation_preserves_training_state_and_rng();
        lookahead_toggle_and_checkpoint_preserve_learning_state();
        pre_lookahead_checkpoint_preserves_n_step_and_symmetry_without_enabling_search();
        lookahead_analysis_preserves_actual_tile_spawn_rng();
        learning_rate_configuration_rejects_nonpositive_and_nonfinite_values();
        learning_rate_override_preserves_adam_and_scales_the_next_update();
        evaluation_is_deterministic_and_does_not_train();
        std::cout << "All training tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Training test failed: " << error.what() << '\n';
        return 1;
    }
}
