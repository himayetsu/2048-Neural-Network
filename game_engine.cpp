#ifndef AI2048_GAME_ENGINE_CPP
#define AI2048_GAME_ENGINE_CPP

// Declarations and implementation are kept together; include this component once.

#include <array>
#include <cstdint>
#include <random>

namespace ai2048 {

constexpr int kGridSize = 4;
constexpr int kInputSize = kGridSize * kGridSize;
constexpr int kActionCount = 4;
constexpr int kSymmetryCount = 8;

using Board = std::array<std::array<int, kGridSize>, kGridSize>;
using State = std::array<float, kInputSize>;
using ActionMask = std::array<bool, kActionCount>;

enum class Direction { Up = 0, Down = 1, Left = 2, Right = 3 };

// Symmetries 0..7 first reflect columns for ids >= 4, then rotate clockwise
// (id % 4) times. Actions and legal-action masks use the same coordinate map.
Board transform_board(const Board& board, int symmetry);
State transform_state(const State& state, int symmetry);
int transform_action(int action, int symmetry);
ActionMask transform_actions(const ActionMask& actions, int symmetry);

struct MoveResult {
    bool moved;
    int score_gain;
    bool done;
};

class GameEngine {
public:
    explicit GameEngine(std::uint32_t seed = std::random_device{}());

    void reset();
    // Sets a board for analysis or testing and resets the move counter.
    void set_board(const Board& board, int score = 0);
    // Applies the deterministic part of a move, without spawning a tile.
    bool move(Direction direction);
    // Applies a move and spawns one tile if the board changed.
    MoveResult step(int action);

    const Board& board() const;
    int score() const;
    int max_tile() const;
    // Number of successful steps since reset() or set_board().
    int moves() const;
    bool game_over() const;
    ActionMask valid_actions() const;
    State state() const;

private:
    void spawn_tile();

    Board board_{};
    int score_ = 0;
    int max_tile_ = 0;
    int moves_ = 0;
    std::mt19937 rng_;
};

}  // namespace ai2048

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ai2048 {
namespace {

struct SlideResult {
    Board board;
    int score_gain = 0;
};

void validate_action(int action) {
    if (action < 0 || action >= kActionCount) {
        throw std::out_of_range("2048 action must be between 0 and 3");
    }
}

void validate_symmetry(int symmetry) {
    if (symmetry < 0 || symmetry >= kSymmetryCount) {
        throw std::out_of_range("2048 symmetry must be between 0 and 7");
    }
}

std::pair<int, int> transformed_cell(int row, int column, int symmetry) {
    if (symmetry >= 4) {
        column = kGridSize - 1 - column;
    }
    for (int rotation = 0; rotation < symmetry % 4; ++rotation) {
        const int next_row = column;
        column = kGridSize - 1 - row;
        row = next_row;
    }
    return {row, column};
}

std::pair<int, int> cell_for(Direction direction, int line, int offset) {
    switch (direction) {
        case Direction::Up: return {offset, line};
        case Direction::Down: return {kGridSize - 1 - offset, line};
        case Direction::Left: return {line, offset};
        case Direction::Right: return {line, kGridSize - 1 - offset};
    }
    throw std::out_of_range("2048 direction must be between 0 and 3");
}

SlideResult slide(const Board& board, Direction direction) {
    validate_action(static_cast<int>(direction));
    SlideResult result{board};
    for (int line = 0; line < kGridSize; ++line) {
        std::array<int, kGridSize> tiles{};
        int count = 0;
        for (int offset = 0; offset < kGridSize; ++offset) {
            const auto [row, column] = cell_for(direction, line, offset);
            if (board[row][column] != 0) {
                tiles[count++] = board[row][column];
            }
        }

        std::array<int, kGridSize> merged{};
        int destination = 0;
        for (int source = 0; source < count; ++source) {
            int tile = tiles[source];
            if (source + 1 < count && tile == tiles[source + 1]) {
                if (tile > std::numeric_limits<int>::max() / 2) {
                    throw std::overflow_error("2048 merged tile exceeds the integer range");
                }
                tile *= 2;
                if (result.score_gain > std::numeric_limits<int>::max() - tile) {
                    throw std::overflow_error("2048 move score exceeds the integer range");
                }
                result.score_gain += tile;
                ++source;  // A tile can merge only once in this move.
            }
            merged[destination++] = tile;
        }

        for (int offset = 0; offset < kGridSize; ++offset) {
            const auto [row, column] = cell_for(direction, line, offset);
            result.board[row][column] = merged[offset];
        }
    }
    return result;
}

int largest_tile(const Board& board) {
    int largest = 0;
    for (const auto& row : board) {
        largest = std::max(largest, *std::max_element(row.begin(), row.end()));
    }
    return largest;
}

}  // namespace

Board transform_board(const Board& board, int symmetry) {
    validate_symmetry(symmetry);
    Board result{};
    for (int row = 0; row < kGridSize; ++row) {
        for (int column = 0; column < kGridSize; ++column) {
            const auto [next_row, next_column] = transformed_cell(row, column, symmetry);
            result[next_row][next_column] = board[row][column];
        }
    }
    return result;
}

State transform_state(const State& state, int symmetry) {
    validate_symmetry(symmetry);
    State result{};
    for (int row = 0; row < kGridSize; ++row) {
        for (int column = 0; column < kGridSize; ++column) {
            const auto [next_row, next_column] = transformed_cell(row, column, symmetry);
            result[next_row * kGridSize + next_column] = state[row * kGridSize + column];
        }
    }
    return result;
}

int transform_action(int action, int symmetry) {
    validate_action(action);
    validate_symmetry(symmetry);
    constexpr std::array<int, kActionCount> row_delta{-1, 1, 0, 0};
    constexpr std::array<int, kActionCount> column_delta{0, 0, -1, 1};
    int row = row_delta[action];
    int column = column_delta[action];
    if (symmetry >= 4) {
        column = -column;
    }
    for (int rotation = 0; rotation < symmetry % 4; ++rotation) {
        const int next_row = column;
        column = -row;
        row = next_row;
    }
    if (row == -1) return static_cast<int>(Direction::Up);
    if (row == 1) return static_cast<int>(Direction::Down);
    if (column == -1) return static_cast<int>(Direction::Left);
    return static_cast<int>(Direction::Right);
}

ActionMask transform_actions(const ActionMask& actions, int symmetry) {
    validate_symmetry(symmetry);
    ActionMask result{};
    for (int action = 0; action < kActionCount; ++action) {
        result[transform_action(action, symmetry)] = actions[action];
    }
    return result;
}

GameEngine::GameEngine(std::uint32_t seed) : rng_(seed) {
    reset();
}

void GameEngine::reset() {
    board_ = {};
    score_ = 0;
    max_tile_ = 0;
    moves_ = 0;
    spawn_tile();
    spawn_tile();
}

void GameEngine::set_board(const Board& board, int score) {
    for (const auto& row : board) {
        for (int tile : row) {
            if (tile < 0 || tile == 1 || (tile != 0 && (tile & (tile - 1)) != 0)) {
                throw std::invalid_argument("2048 tiles must be zero or a power of two of at least two");
            }
        }
    }
    if (score < 0) {
        throw std::invalid_argument("2048 score cannot be negative");
    }
    board_ = board;
    score_ = score;
    max_tile_ = largest_tile(board_);
    moves_ = 0;
}

bool GameEngine::move(Direction direction) {
    const auto result = slide(board_, direction);
    if (result.board == board_) {
        return false;
    }
    if (score_ > std::numeric_limits<int>::max() - result.score_gain) {
        throw std::overflow_error("2048 total score exceeds the integer range");
    }
    board_ = result.board;
    score_ += result.score_gain;
    max_tile_ = largest_tile(board_);
    return true;
}

MoveResult GameEngine::step(int action) {
    validate_action(action);
    const int previous_score = score_;
    const bool moved = move(static_cast<Direction>(action));
    if (moved) {
        ++moves_;
        spawn_tile();
    }
    return {moved, score_ - previous_score, game_over()};
}

void GameEngine::spawn_tile() {
    std::array<std::pair<int, int>, kInputSize> empty_cells{};
    int count = 0;
    for (int row = 0; row < kGridSize; ++row) {
        for (int column = 0; column < kGridSize; ++column) {
            if (board_[row][column] == 0) {
                empty_cells[count++] = {row, column};
            }
        }
    }
    if (count == 0) {
        return;
    }
    std::uniform_int_distribution<int> position(0, count - 1);
    const auto [row, column] = empty_cells[position(rng_)];
    std::uniform_real_distribution<float> value(0.0f, 1.0f);
    board_[row][column] = value(rng_) < 0.9f ? 2 : 4;
    max_tile_ = std::max(max_tile_, board_[row][column]);
}

const Board& GameEngine::board() const { return board_; }
int GameEngine::score() const { return score_; }
int GameEngine::max_tile() const { return max_tile_; }
int GameEngine::moves() const { return moves_; }

ActionMask GameEngine::valid_actions() const {
    ActionMask mask{};
    for (int action = 0; action < kActionCount; ++action) {
        mask[action] = slide(board_, static_cast<Direction>(action)).board != board_;
    }
    return mask;
}

bool GameEngine::game_over() const {
    const auto mask = valid_actions();
    return std::none_of(mask.begin(), mask.end(), [](bool legal) { return legal; });
}

State GameEngine::state() const {
    State result{};
    for (int row = 0; row < kGridSize; ++row) {
        for (int column = 0; column < kGridSize; ++column) {
            const int tile = board_[row][column];
            if (tile != 0) {
                result[row * kGridSize + column] = std::log2(static_cast<float>(tile)) / 17.0f;
            }
        }
    }
    return result;
}

}  // namespace ai2048

#endif  // AI2048_GAME_ENGINE_CPP
