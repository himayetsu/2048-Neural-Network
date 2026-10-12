# Running this checkout on macOS

This checkout is built for Apple Silicon (`arm64`). Start it from Terminal:

```sh
cd /Users/henry.h.li/Personal/2048
./build/2048_ai
```

The app displays the game and neural network while training. Press **Esc** in
either window to quit and save, or **S** to save manually. **Ctrl+C** also stops
and saves in graphical or headless training. The default run is 10,000
additional episodes. Model files are saved in the current directory as
`2048_model.pt`, `2048_model.pt.state`, and `2048_model.pt.milestones`, and are
loaded automatically on the next run.

For a shorter run or training without graphics:

```sh
./build/2048_ai --episodes 100
./build/2048_ai --no-render --episodes 100
./build/2048_ai --help
```

This version selects CUDA or CPU; it uses CPU on this Mac. Early training starts
with random legal moves and reduces exploration linearly from 100% to 1% over
100,000 transitions. Use evaluation to watch the learned policy with exploration
disabled:

```sh
./build/2048_ai --eval --episodes 20 --seed 1000000
```

Evaluation performs no learning and never saves the model. It defaults to 20
games on seeds starting at 1,000,000, separate from the new-training seed of 42.
The network window shows raw Q-values, legal moves, the selected action, and
whether training selected it through random exploration.

## Training and evaluation

Start a separate experiment without replacing the default checkpoint:

```sh
./build/2048_ai --fresh --no-render --episodes 1000 --seed 42 \
  --exploration-steps 100000 --model score_model.pt
./build/2048_ai --no-render --episodes 1000 --model score_model.pt
```

`--seed` and `--exploration-steps` configure new training runs. A resumed run
restores the checkpoint's saved configuration. Current checkpoints include the
policy, target network, Adam optimizer, configuration, trainer random generator,
counters, and milestones. The replay buffer warms up again after loading;
an interrupted game restarts. Legacy models load their weights with a fresh
optimizer and exploration schedule. Corrupt or incompatible checkpoints stop
with an error and are not overwritten.

Compare the trained policy with random legal moves and an untrained network on
the same held-out game seeds:

```sh
./build/2048_ai --eval --random --no-render --episodes 200 --seed 1000000
./build/2048_ai --fresh --no-render --episodes 0 --seed 42 --model untrained_seed42.pt
./build/2048_ai --eval --no-render --episodes 200 --seed 1000000 --model untrained_seed42.pt
./build/2048_ai --eval --no-render --episodes 200 --seed 1000000 --model score_model.pt
```

Use mean/median score and tile milestone rates to assess progress. Add
`--csv PATH` to save per-game metrics. A nonzero training loss alone does not
establish good gameplay. See [README.md](README.md) for the training algorithm
and defaults.

## Installed dependencies

All added dependencies are local to this checkout. The existing Apple command
line tools provide the compiler and Make.

| Dependency | Version | Location |
| --- | --- | --- |
| CMake | 3.31.10 | `build/tools/CMake.app/Contents/bin/cmake` |
| SDL2 | 2.32.10 | `build/deps` |
| SDL2_ttf | 2.24.0 | `build/deps` |
| FreeType | Bundled with SDL2_ttf | Statically linked into SDL2_ttf |
| LibTorch | 2.9.1, macOS arm64 | `libtorch` |
| OpenMP runtime | Bundled with LibTorch | `libtorch/lib/libomp.dylib` |

Keep `build/` and `libtorch/`: they contain the installed libraries and tools,
not just disposable compiler output. Both directories are already Git-ignored.
Homebrew and Python packages are not needed to run the built application.

## Rebuild after editing the app

From the repository root:

```sh
./build/tools/CMake.app/Contents/bin/cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_PREFIX_PATH="$PWD/libtorch;$PWD/build/deps"
./build/tools/CMake.app/Contents/bin/cmake --build build --parallel 8
./build/tools/CMake.app/Contents/bin/ctest --test-dir build --output-on-failure
```

The application is split into `game_engine.cpp` (pure board mechanics),
`training.cpp` (Double DQN, checkpoints, and shared display data), and
`training_visualization.cpp` (SDL display). Each component combines declarations
and implementation in a guarded `.cpp` file. `2048_ai.cpp` handles command-line
options and startup and includes the training and visualization components.
CMake compiles one translation unit per application or test to avoid compiling
included components twice. The engine tests include `game_engine.cpp` and need
only standard C++; the training tests include `training.cpp` and link LibTorch
without SDL. CTest runs both regression suites, including in Release builds.

## Installation details

Official downloads used:

- [CMake 3.31.10](https://github.com/Kitware/CMake/releases/tag/v3.31.10)
- [SDL2 2.32.10](https://github.com/libsdl-org/SDL/releases/tag/release-2.32.10)
- [SDL2_ttf 2.24.0](https://github.com/libsdl-org/SDL_ttf/releases/tag/release-2.24.0)
- [LibTorch 2.9.1 for macOS arm64](https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.9.1.zip)

Downloads and SDL source trees are retained under `build/downloads` and
`build/sources`. SDL libraries were built in Release mode for arm64 and
installed into `build/deps`. SDL2_ttf uses its vendored FreeType with HarfBuzz
and sample applications disabled. Build and configuration logs are in `build`.

The downloaded LibTorch CPU library referred to a Homebrew OpenMP path even
though OpenMP was bundled. Its local dependency path was corrected and its
ad-hoc signature refreshed with:

```sh
install_name_tool -change /opt/homebrew/opt/libomp/lib/libomp.dylib \
  @loader_path/libomp.dylib libtorch/lib/libtorch_cpu.dylib
codesign --force --sign - libtorch/lib/libtorch_cpu.dylib
```

The installed dependency versions and local library path adjustment are
unchanged by the source refactor. Existing setup smoke-test models are isolated
under `build/smoke`.
