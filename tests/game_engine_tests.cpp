#include "game_engine.cpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace ai2048;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Exception, class Function>
void require_throws(Function function, const std::string& message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

int occupied_cells(const Board& board) {
    int occupied = 0;
    for (const auto& row : board) {
        occupied += static_cast<int>(std::count_if(row.begin(), row.end(), [](int tile) { return tile != 0; }));
    }
    return occupied;
}

int maximum(const Board& board) {
    int result = 0;
    for (const auto& row : board) {
        result = std::max(result, *std::max_element(row.begin(), row.end()));
    }
    return result;
}

void movement_in_each_direction() {
    struct Case {
        const char* name;
        Direction direction;
        Board initial;
        Board expected;
    };
    const Case cases[] = {
        {"left", Direction::Left,
         {{{0, 2, 0, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
         {{{4, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}}},
        {"right", Direction::Right,
         {{{0, 2, 0, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
         {{{0, 0, 0, 4}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}}},
        {"up", Direction::Up,
         {{{0, 0, 0, 0}, {2, 0, 0, 0}, {0, 0, 0, 0}, {2, 0, 0, 0}}},
         {{{4, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}}},
        {"down", Direction::Down,
         {{{0, 0, 0, 0}, {2, 0, 0, 0}, {0, 0, 0, 0}, {2, 0, 0, 0}}},
         {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {4, 0, 0, 0}}}},
    };
    for (const auto& test : cases) {
        GameEngine game(1);
        game.set_board(test.initial, 12);
        const std::string context = std::string("movement ") + test.name + ": ";
        require(game.move(test.direction), context + "slide should change the board");
        require(game.board() == test.expected, context + "unexpected slide/merge result or extra spawned tile");
        require(game.score() == 16, context + "score should increase by the merged tile's value");
        require(game.max_tile() == 4, context + "maximum tile should reflect the merge");
        require(game.moves() == 0, context + "deterministic analysis moves should not count as played steps");
    }
}

void each_tile_merges_only_once() {
    GameEngine game(2);
    game.set_board({{{2, 2, 4, 0}, {2, 2, 2, 2}, {4, 4, 4, 4}, {0, 0, 0, 0}}});
    require(game.move(Direction::Left), "single-merge rule: move should be legal");
    const Board expected{{{4, 4, 0, 0}, {4, 4, 0, 0}, {8, 8, 0, 0}, {0, 0, 0, 0}}};
    require(game.board() == expected, "single-merge rule: a newly formed tile must not merge again in the same move");
    require(game.score() == 28, "single-merge rule: score should sum every merged tile exactly once");
}

void invalid_moves_leave_game_unchanged() {
    GameEngine game(3);
    const Board initial{{{2, 4, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}};
    game.set_board(initial, 24);
    const auto result = game.step(static_cast<int>(Direction::Left));
    require(!result.moved, "invalid move: an already compressed row without merges should not move");
    require(result.score_gain == 0, "invalid move: reward input must report zero score gain");
    require(!result.done, "invalid move: other available directions should keep the game alive");
    require(game.board() == initial, "invalid move: no tile should spawn or change");
    require(game.score() == 24 && game.moves() == 0 && game.max_tile() == 4,
            "invalid move: score, maximum, and successful-step count should be unchanged");
}

void legal_step_spawns_exactly_one_tile() {
    GameEngine game(4);
    game.set_board({{{0, 2, 2, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}}, 8);
    const auto result = game.step(static_cast<int>(Direction::Left));
    require(result.moved && result.score_gain == 4, "legal step: merge should report exactly four score points");
    require(game.board()[0][0] == 4, "legal step: merged tile should remain at the left edge");
    require(occupied_cells(game.board()) == 2, "legal step: one merged tile plus exactly one spawned tile expected");
    require(game.score() == 12 && game.moves() == 1, "legal step: score and successful-step count should update once");
    require(game.max_tile() == maximum(game.board()), "legal step: maximum tile must include the newly spawned tile");
}

void legal_action_masks_and_terminal_boards() {
    GameEngine game(5);
    game.set_board({{{2, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}});
    require(game.valid_actions() == ActionMask{false, true, false, true},
            "corner mask: only down and right should change a lone top-left tile");

    const Board full_mergeable{{{2, 2, 8, 16}, {8, 16, 32, 64}, {16, 32, 64, 128}, {32, 64, 128, 256}}};
    game.set_board(full_mergeable);
    require(game.valid_actions() == ActionMask{false, false, true, true},
            "full-board mask: only horizontal moves should merge the adjacent pair");
    require(!game.game_over(), "full-board mask: a full board with a merge is not terminal");
    require(game.board() == full_mergeable && game.score() == 0,
            "action mask: querying legal actions must not change board or score");

    const Board terminal{{{2, 4, 8, 16}, {4, 8, 16, 32}, {8, 16, 32, 64}, {16, 32, 64, 128}}};
    game.set_board(terminal);
    require(game.valid_actions() == ActionMask{false, false, false, false},
            "terminal mask: no action should be legal without empty cells or equal neighbors");
    require(game.game_over(), "terminal mask: an immobile board should be terminal");
    const auto result = game.step(static_cast<int>(Direction::Up));
    require(!result.moved && result.done && result.score_gain == 0,
            "terminal step: an immobile board should remain terminal without score gain");
    require(game.board() == terminal, "terminal step: tiles must remain unchanged");
}

void final_spawn_can_end_the_game() {
    GameEngine game(9);
    game.set_board({{{8, 16, 32, 32}, {16, 32, 128, 8}, {32, 64, 256, 16}, {64, 128, 512, 32}}});
    const auto result = game.step(static_cast<int>(Direction::Left));
    require(result.moved && result.score_gain == 64, "final spawn: the last available pair should merge");
    require(occupied_cells(game.board()) == kInputSize, "final spawn: the only empty cell should receive a tile");
    require(result.done && game.game_over(), "final spawn: terminal status must be computed after adding the new tile");
}

void reset_tracks_initial_maximum() {
    GameEngine game(6);
    bool saw_four = false;
    for (int trial = 0; trial < 100; ++trial) {
        require(occupied_cells(game.board()) == 2, "reset: a new game must contain exactly two tiles");
        require(game.max_tile() == maximum(game.board()), "reset: initial maximum must include both spawned tiles");
        require(game.max_tile() == 2 || game.max_tile() == 4, "reset: initial tiles must be two or four");
        require(game.score() == 0 && game.moves() == 0 && !game.game_over(), "reset: score, move count, and terminal state should reset");
        saw_four = saw_four || game.max_tile() == 4;
        game.reset();
    }
    require(saw_four, "reset fixture: repeated seeded starts should cover initial four tiles");
}

void spawned_four_can_raise_the_maximum() {
    GameEngine game(10);
    bool saw_four = false;
    for (int trial = 0; trial < 100; ++trial) {
        game.set_board({{{2, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}});
        const auto result = game.step(static_cast<int>(Direction::Right));
        require(result.moved && result.score_gain == 0, "spawn maximum: fixture should slide without merging");
        require(game.max_tile() == maximum(game.board()), "spawn maximum: a new four must update the maximum even without a merge");
        saw_four = saw_four || game.max_tile() == 4;
    }
    require(saw_four, "spawn maximum fixture: seeded trials should include at least one newly spawned four");
}

void equal_seeds_reproduce_games() {
    GameEngine first(1234);
    GameEngine second(1234);
    for (int step = 0; step < 300; ++step) {
        require(first.board() == second.board(), "seed reproducibility: identical seeds and actions must produce identical boards");
        require(first.score() == second.score() && first.max_tile() == second.max_tile(),
                "seed reproducibility: identical runs must have identical score and maximum");
        if (first.game_over()) {
            first.reset();
            second.reset();
        }
        const int action = step % kActionCount;
        const auto one = first.step(action);
        const auto two = second.step(action);
        require(one.moved == two.moved && one.done == two.done && one.score_gain == two.score_gain,
                "seed reproducibility: step results must agree");
    }
}

void invalid_move_does_not_consume_spawn_randomness() {
    GameEngine first(123);
    GameEngine second(123);
    const Board initial{{{2, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}};
    first.set_board(initial);
    second.set_board(initial);
    first.step(static_cast<int>(Direction::Left));
    first.step(static_cast<int>(Direction::Right));
    second.step(static_cast<int>(Direction::Right));
    require(first.board() == second.board(), "invalid move: rejected moves must not consume random tile spawns");
}

void board_and_action_validation() {
    GameEngine game(7);
    const auto initial = game.board();
    for (int tile : {-2, 1, 3, 6}) {
        Board invalid{};
        invalid[0][0] = tile;
        require_throws<std::invalid_argument>([&] { game.set_board(invalid); },
                                             "board validation: tile " + std::to_string(tile) + " should be rejected");
        require(game.board() == initial, "board validation: rejection must preserve the current board");
    }
    for (int action : {-1, 4}) {
        require_throws<std::out_of_range>([&] { game.step(action); },
                                        "action validation: action " + std::to_string(action) + " should be rejected");
        require(game.board() == initial, "action validation: rejection must preserve the current board");
    }
    require_throws<std::out_of_range>([&] { game.move(static_cast<Direction>(4)); },
                                    "direction validation: an invalid enum value should be rejected");
}

void state_preserves_logarithmic_encoding() {
    GameEngine game(8);
    game.set_board({{{0, 2, 4, 8}, {16, 32, 64, 128}, {256, 512, 1024, 2048}, {4096, 8192, 16384, 32768}}});
    const auto state = game.state();
    for (int index = 0; index < kInputSize; ++index) {
        require(std::abs(state[index] - static_cast<float>(index) / 17.0f) < 1e-6f,
                "state encoding: incorrect row-major log2 normalization at cell " + std::to_string(index));
    }
}

void symmetry_maps_preserve_coordinates_and_actions() {
    const std::array<State, kSymmetryCount> expected_states{{
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3, 16, 12, 8, 4},
        {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
        {4, 8, 12, 16, 3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13},
        {4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13},
        {16, 12, 8, 4, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1},
        {13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4},
        {1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16},
    }};
    const std::array<std::array<int, kActionCount>, kSymmetryCount> expected_actions{{
        {0, 1, 2, 3}, {3, 2, 0, 1}, {1, 0, 3, 2}, {2, 3, 1, 0},
        {0, 1, 3, 2}, {3, 2, 1, 0}, {1, 0, 2, 3}, {2, 3, 0, 1},
    }};
    const std::array<int, kSymmetryCount> inverse{0, 3, 2, 1, 4, 5, 6, 7};
    const State original = expected_states[0];
    Board board{};
    for (int index = 0; index < kInputSize; ++index) {
        board[index / kGridSize][index % kGridSize] = 1 << (index + 1);
    }
    for (int symmetry = 0; symmetry < kSymmetryCount; ++symmetry) {
        const std::string context = "symmetry " + std::to_string(symmetry) + ": ";
        const auto state = transform_state(original, symmetry);
        const auto transformed_board = transform_board(board, symmetry);
        require(state == expected_states[symmetry], context + "unexpected reflect-then-clockwise coordinate mapping");
        for (int index = 0; index < kInputSize; ++index) {
            require(transformed_board[index / kGridSize][index % kGridSize] ==
                        (1 << static_cast<int>(expected_states[symmetry][index])),
                    context + "board and state coordinates disagree at cell " + std::to_string(index));
        }
        require(transform_state(state, inverse[symmetry]) == original,
                context + "inverse transform must recover every original state cell");
        require(transform_board(transformed_board, inverse[symmetry]) == board,
                context + "inverse transform must recover every original board cell");
        std::array<bool, kActionCount> seen{};
        for (int action = 0; action < kActionCount; ++action) {
            const int transformed_action = transform_action(action, symmetry);
            const std::string action_context = context + "action " + std::to_string(action) + ": ";
            require(transformed_action == expected_actions[symmetry][action],
                    action_context + "direction must follow the same coordinate transform");
            require(!seen[transformed_action], action_context + "direction map must be one-to-one");
            seen[transformed_action] = true;
            require(transform_action(transformed_action, inverse[symmetry]) == action,
                    action_context + "inverse transform must recover the original direction");
            ActionMask mask{};
            mask[action] = true;
            ActionMask expected_mask{};
            expected_mask[expected_actions[symmetry][action]] = true;
            require(transform_actions(mask, symmetry) == expected_mask,
                    action_context + "legal mask must follow the direction permutation");
        }
    }
}

void transformed_moves_preserve_boards_rewards_and_legal_masks() {
    struct Case {
        const char* name;
        Board board;
    };
    const Case cases[] = {
        {"sparse mergeable", {{{2, 2, 4, 0}, {8, 0, 8, 16}, {0, 32, 0, 32}, {64, 128, 256, 512}}}},
        {"horizontal only", {{{2, 2, 8, 16}, {8, 16, 32, 64}, {16, 32, 64, 128}, {32, 64, 128, 256}}}},
        {"terminal", {{{2, 4, 8, 16}, {4, 8, 16, 32}, {8, 16, 32, 64}, {16, 32, 64, 128}}}},
    };
    for (const auto& test : cases) {
        for (int symmetry = 0; symmetry < kSymmetryCount; ++symmetry) {
            for (int action = 0; action < kActionCount; ++action) {
                const std::string context = std::string("move symmetry ") + test.name + ", transform " +
                    std::to_string(symmetry) + ", action " + std::to_string(action) + ": ";
                GameEngine original(11);
                GameEngine transformed(11);
                original.set_board(test.board, 24);
                transformed.set_board(transform_board(test.board, symmetry), 24);
                require(transformed.state() == transform_state(original.state(), symmetry),
                        context + "encoding must commute with transforming the board");
                require(transformed.valid_actions() == transform_actions(original.valid_actions(), symmetry),
                        context + "legal moves must match before the move");
                const bool moved = original.move(static_cast<Direction>(action));
                const bool transformed_moved = transformed.move(static_cast<Direction>(transform_action(action, symmetry)));
                require(transformed_moved == moved, context + "move/no-op status must be unchanged");
                require(transformed.board() == transform_board(original.board(), symmetry),
                        context + "deterministic move must commute with transforming the board");
                require(transformed.score() == original.score(), context + "merge reward must be identical");
                require(transformed.max_tile() == original.max_tile(), context + "maximum tile must be identical");
                require(transformed.valid_actions() == transform_actions(original.valid_actions(), symmetry),
                        context + "legal moves must match after the move");
                require(transformed.game_over() == original.game_over(), context + "terminal status must be identical");
            }
        }
    }
}

void symmetry_inputs_are_validated() {
    for (int symmetry : {-1, kSymmetryCount}) {
        const std::string context = "symmetry validation " + std::to_string(symmetry) + ": ";
        require_throws<std::out_of_range>([&] { transform_state(State{}, symmetry); },
                                        context + "state transform should reject an invalid id");
        require_throws<std::out_of_range>([&] { transform_board(Board{}, symmetry); },
                                        context + "board transform should reject an invalid id");
        require_throws<std::out_of_range>([&] { transform_action(0, symmetry); },
                                        context + "action transform should reject an invalid id");
        require_throws<std::out_of_range>([&] { transform_actions(ActionMask{}, symmetry); },
                                        context + "mask transform should reject an invalid id");
    }
    for (int action : {-1, kActionCount}) {
        require_throws<std::out_of_range>([&] { transform_action(action, 0); },
                                        "symmetry action validation: reject action " + std::to_string(action));
    }
}

}  // namespace

int main() {
    try {
        movement_in_each_direction();
        each_tile_merges_only_once();
        invalid_moves_leave_game_unchanged();
        legal_step_spawns_exactly_one_tile();
        legal_action_masks_and_terminal_boards();
        final_spawn_can_end_the_game();
        reset_tracks_initial_maximum();
        spawned_four_can_raise_the_maximum();
        equal_seeds_reproduce_games();
        invalid_move_does_not_consume_spawn_randomness();
        board_and_action_validation();
        state_preserves_logarithmic_encoding();
        symmetry_maps_preserve_coordinates_and_actions();
        transformed_moves_preserve_boards_rewards_and_legal_masks();
        symmetry_inputs_are_validated();
        std::cout << "All game engine tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Game engine test failed: " << error.what() << '\n';
        return 1;
    }
}
