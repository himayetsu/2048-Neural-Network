# 2048 AI Neural Network

A Deep Q-Network (DQN) implementation that learns to play the game 2048. Features real-time visualization of the neural network weights and activations.

![2048 AI](https://img.shields.io/badge/C%2B%2B-17-blue) ![PyTorch](https://img.shields.io/badge/LibTorch-2.0%2B-orange) ![SDL2](https://img.shields.io/badge/SDL2-2.0-green)

## Features

- **Complete 2048 Game Engine**: Full implementation of 2048 game mechanics
- **Deep Q-Network (DQN)**: Neural network with experience replay and target network
- **Real-time Visualization**: 
  - Game board with colored tiles
  - Static network weight visualization
  - Dynamic activation visualization during gameplay
- **Configurable Training**: Adjustable hyperparameters and episode count

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

```bash
# Run with visualization (default)
./2048_ai

# Run without visualization (faster training)
./2048_ai --no-render

# Specify number of episodes
./2048_ai --episodes 50000
```

## Visualization Windows

1. **Game Window**: Shows the current 2048 board state with colored tiles
2. **Q-Network Visualization**: Static view of network weights (red = positive, green = negative)
3. **Dynamic Q-Network Activity**: Real-time activation visualization showing:
   - Neuron brightness = activation intensity
   - Neuron size = activation magnitude
   - Connection color = weighted signal flow
   - Output bars = action probabilities

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | 0.0005 | Adam optimizer learning rate |
| `GAMMA` | 0.99 | Discount factor for future rewards |
| `EPSILON_START` | 1.0 | Initial exploration rate |
| `EPSILON_END` | 0.001 | Minimum exploration rate |
| `EPSILON_DECAY` | 0.9995 | Exploration decay per episode |
| `BATCH_SIZE` | 128 | Experience replay batch size |
| `REPLAY_BUFFER_SIZE` | 50000 | Maximum replay buffer size |
| `TARGET_UPDATE_EVERY` | 1000 | Steps between target network updates |

## State Representation

The game state is represented as a 16-element vector where each element corresponds to a cell in the 4x4 grid:
- Empty cells = 0.0
- Tiles are normalized: `log2(tile_value) / 17.0`

This normalization helps the network learn because:
- All values are in [0, 1] range
- The logarithmic scale captures the exponential nature of 2048 tiles

## Reward Function

```cpp
reward = 0
if (valid_move):
    reward += score_gained / 100.0      // Reward for merging
    if (new_max_tile):
        reward += log2(max_tile) * 0.5  // Bonus for achieving new high tile
    if (max_tile_in_corner):
        reward += 0.1                    // Strategy reward
    if (created_empty_space):
        reward += 0.05 * spaces_gained   // Reward for keeping board clear
else:
    reward = -0.5                        // Penalty for invalid move

if (game_over):
    reward -= 1.0                        // Penalty for losing
    reward += log2(max_tile) * 0.2       // Consolation based on final tile
```

## Tips for Training

1. **Patience**: The agent needs thousands of episodes to learn effective strategies
2. **GPU**: Use CUDA-enabled LibTorch for faster training
3. **Monitoring**: Watch the max tile metric - reaching 512+ regularly indicates learning
4. **Tuning**: Try adjusting `GAMMA` (0.95-0.99) and `LEARNING_RATE` (0.0001-0.001)

## License

MIT License - Feel free to use and modify!

## Acknowledgments

- Inspired by [Ethan-Blesch/Snake-neural-network](https://github.com/Ethan-Blesch/Snake-neural-network)
- 2048 game created by Gabriele Cirulli
