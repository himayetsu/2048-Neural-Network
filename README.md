# 2048 AI Neural Network

A Double Deep Q-Network (Double DQN) implementation that learns to play 2048. Features real-time visualization of the game, neural network activity, and action values.

![2048 AI](https://img.shields.io/badge/C%2B%2B-17-blue) ![PyTorch](https://img.shields.io/badge/LibTorch-2.0%2B-orange) ![SDL2](https://img.shields.io/badge/SDL2-2.0-green)

## Features

- **Complete 2048 Game Engine**: Full implementation of 2048 game mechanics
- **Double DQN Training**: Experience replay, target network, and legal-action masking
- **Real-time Visualization**: 
  - Game board with colored tiles
  - Dynamic network activations and weighted signals
  - Legal moves, raw Q-values, and exploration status
- **Reproducible Evaluation**: Greedy, random legal-move, and untrained baselines with CSV metrics
- **Resumable Training**: Checkpoints include network, optimizer, and training state

## Architecture

```
Input Layer (16 neurons) - 4x4 grid, log-normalized tile values
    ↓
Hidden Layer 1 (256 neurons) - ReLU activation
    ↓
Hidden Layer 2 (128 neurons) - ReLU activation  
    ↓
Hidden Layer 3 (128 neurons) - ReLU activation
    ↓
Output Layer (4 neurons) - Q-values for UP, DOWN, LEFT, RIGHT
```

The source is split by responsibility:

| Files | Responsibility |
|-------|----------------|
| `game_engine.cpp` | Board mechanics, legal moves, tile spawning, score, and state encoding; standard C++ only |
| `training.cpp` | Network, replay buffer, Double DQN updates, evaluation, checkpoints, and shared display data |
| `training_visualization.cpp` | SDL windows, rendering, and keyboard events |
| `2048_ai.cpp` | Command-line options and application startup |
| `tests/` | Focused game-engine and training regression tests |

Each component combines its declarations and implementation in a guarded `.cpp`
file. `2048_ai.cpp` includes the training and visualization components; training
includes the game engine, and visualization includes training. CMake compiles
one translation unit per application or test, so included components are not
also compiled separately.

`GameEngine::move()` performs a deterministic slide/merge for analysis;
`step()` additionally spawns a tile after a legal move. `DQNTrainer::train()`
learns from games, while `evaluate()` runs a fixed policy without learning.

## Dependencies

- **LibTorch** (PyTorch C++ API) >= 2.0
- **SDL2** >= 2.0
- **SDL2_ttf** >= 2.0
- **CMake** >= 3.14
- **C++17** compatible compiler

## Installation

### Windows

1. **Install vcpkg** (recommended package manager):
   ```powershell
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   .\bootstrap-vcpkg.bat
   .\vcpkg integrate install
   ```

2. **Install SDL2 and SDL2_ttf**:
   ```powershell
   .\vcpkg install sdl2:x64-windows sdl2-ttf:x64-windows
   ```

3. **Download LibTorch**:
   - Go to https://pytorch.org/get-started/locally/
   - Select: LibTorch, C++/Java, Windows, CPU (or CUDA if you have a GPU)
   - Extract to a known location (e.g., `C:\libtorch`)

4. **Build**:
   ```powershell
   mkdir build
   cd build
   cmake .. -DCMAKE_PREFIX_PATH="C:\libtorch;C:\vcpkg\installed\x64-windows" -DCMAKE_TOOLCHAIN_FILE="C:\vcpkg\scripts\buildsystems\vcpkg.cmake"
   cmake --build . --config Release
   ```

### Linux (Ubuntu/Debian)

1. **Install dependencies**:
   ```bash
   sudo apt update
   sudo apt install libsdl2-dev libsdl2-ttf-dev cmake build-essential
   ```

2. **Download LibTorch**:
   ```bash
   wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
   unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
   ```

3. **Build**:
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
   make -j$(nproc)
   ```

### macOS

For this already configured Apple Silicon checkout, use [MACOS_SETUP.md](MACOS_SETUP.md).
For a new installation:

1. **Install dependencies**:
   ```bash
   brew install sdl2 sdl2_ttf cmake
   ```

2. **Download LibTorch** from PyTorch website

3. **Build**:
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
   make -j$(sysctl -n hw.ncpu)
   ```

## Usage

Run these commands from the repository root after building:

```bash
# Train with visualization; resumes 2048_model.pt if present
./build/2048_ai

# Train a new model without graphics
./build/2048_ai --fresh --no-render --episodes 1000 --seed 42 --model score_model.pt

# Resume for 1,000 additional episodes
./build/2048_ai --no-render --episodes 1000 --model score_model.pt

# Watch the learned policy without exploration or further training
./build/2048_ai --eval --model score_model.pt --episodes 20 --seed 1000000

./build/2048_ai --help
```

Training defaults to 10,000 additional episodes; evaluation defaults to 20.
Press **Esc**, close either window, or press **Ctrl+C** to stop. Training saves
on exit and every 100 completed episodes; **S** saves during graphical training.
`--eval` never learns or saves a model. Use `--delay-ms` to control display speed.

`--seed` (default 42) and `--exploration-steps` configure new training runs.
Resuming a current-format checkpoint restores its saved configuration instead.
For evaluation, `--seed` selects the game seeds (default 1,000,000); use a range
outside the training seeds when measuring progress.

### Checkpoints

`--model PATH` selects the checkpoint, defaulting to `2048_model.pt`. The `.pt`
archive contains the policy, target network, Adam optimizer, configuration,
trainer random generator, counters, and milestones. The `.state` and
`.milestones` companions are human-readable summaries; the archive is authoritative.
The replay buffer is intentionally omitted and warms up again after loading.
An interrupted game restarts on resume, so resuming is not identical to an
uninterrupted run.

Legacy checkpoints load their network weights with a fresh optimizer and
exploration schedule. A corrupt or incompatible checkpoint stops the run with
an error and is not overwritten. Use a separate model path when starting a new
experiment with `--fresh`.

## Visualization Windows

1. **Game Window**: Current board, score, maximum tile, and episode.
2. **Neural Network Window**: Sampled neuron activations and weighted signals
   (red = positive, green = negative). Output bars show signed **Q-values**,
   which estimate discounted future reward. Blocked actions are marked; green
   outlines identify the best legal action and white outlines identify the
   selected action. The window also shows exploration, loss, replay size, and
   optimizer updates.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.0001 | Adam optimizer learning rate |
| `gamma` | 0.99 | Discount factor for future rewards |
| `epsilon_start` | 1.0 | Initial probability of a random legal move |
| `epsilon_end` | 0.01 | Final exploration probability |
| `exploration_steps` | 100000 | Transitions over which epsilon decreases linearly |
| `batch_size` | 256 | Experience replay batch size and initial warmup size |
| `replay_capacity` | 100000 | Maximum stored transitions |
| `train_every` | 4 | Transitions between optimizer updates |
| `target_update_every` | 250 | Optimizer updates between target-network copies |
| `max_episode_steps` | 5000 | Game limit; capped games are reported as truncated |

Defaults live in `TrainingConfig` in `training.cpp`. Both exploration and greedy
selection use legal moves. Double DQN selects the next legal action with the
policy network and values it with the target network. Terminal transitions
have no future-value term. Training uses Huber loss and clips the gradient norm
to 1.0.

## State Representation

The game state is represented as a row-major 16-element vector where each element corresponds to a cell in the 4x4 grid:

- Empty cells = 0.0
- Tiles are normalized: `log2(tile_value) / 17.0`

This normalization helps the network learn because:

- All values are in [0, 1] range
- The logarithmic scale captures the exponential nature of 2048 tiles

## Reward Function

```cpp
reward = score_gained / 100.0f;
```

This optimizes discounted game score. There are no recurring corner, empty-cell,
monotonicity, or terminal bonuses. Invalid moves are excluded by the action mask.

## Verify Training

See [TRAINING_VALIDATION.md](TRAINING_VALIDATION.md) for the audit, regression
checks, two-seed benchmark results, and commands to watch the verified models.

Build and run the regression tests:

```bash
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Compare policies on the same held-out game seeds and episode count:

```bash
# Random legal-move baseline
./build/2048_ai --eval --random --no-render --episodes 200 --seed 1000000 --csv random_eval.csv

# Save the same initial weights as a new seed-42 training run, then evaluate them
./build/2048_ai --fresh --no-render --episodes 0 --seed 42 --model untrained_seed42.pt
./build/2048_ai --eval --no-render --episodes 200 --seed 1000000 --model untrained_seed42.pt --csv untrained_eval.csv

# Saved trained network, with exploration disabled
./build/2048_ai --eval --no-render --episodes 200 --seed 1000000 --model score_model.pt --csv trained_eval.csv
```

Evaluation reports mean and median score, counts reaching each tile milestone,
and truncated games. CSV output records each game's score, maximum tile, steps,
truncation status, loss, and exploration probability. Optimizer-update counts
show whether learning steps are running; improvement over the baselines measures
playing strength. Individual games are noisy, so compare many games rather than
a single high score.

## Tips for Training

1. **Watch evaluation games** to judge the policy. Early training deliberately selects many random moves.
2. **Use headless training** for throughput. This Mac uses CPU; CUDA is selected on supported installations.
3. **Keep experiments separate** with `--model`, fixed seeds, and CSV metrics.
4. **Measure changes** against the same held-out baselines before extending a training run or tuning `TrainingConfig`.

## License

MIT License - Feel free to use and modify!

## Acknowledgments

- Inspired by [Ethan-Blesch/Snake-neural-network](https://github.com/Ethan-Blesch/Snake-neural-network)
- 2048 game created by Gabriele Cirulli
