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
        std::cout << "All game engine tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Game engine test failed: " << error.what() << '\n';
        return 1;
    }
}
