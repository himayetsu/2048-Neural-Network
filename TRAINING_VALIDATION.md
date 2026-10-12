# Training validation — 2026-10-12

The training machinery is functional, and two independent 5,000-game training
runs improved over both random play and their own initial networks. Neither
5,000-game model reached 2048 in the final evaluation at that stage. A subsequent
seed-42 continuation to 55,000 games reached 2048 in 2 of 200 evaluation games,
as detailed below.

## What changed

The original code already called `backward()` and `Adam::step()`. Weak play did
not mean that training was disabled. The audit found these concrete issues:

- Initial/spawned tiles did not update the maximum-tile field. The old reward
  therefore sometimes treated empty corners as the maximum tile before a merge.
- Saved episode indices repeated on resume. Adam state and the lagged target
  network were not saved. Load errors were treated as missing files, followed by
  automatic saving that could overwrite an unreadable checkpoint.
- Headless mode skipped the interrupt flag, so Ctrl+C did not stop training.
- The UI displayed `softmax(Q)` as action probabilities, although the policy
  actually used epsilon-greedy selection.

The old exploration schedule also made training difficult to judge: after
1,000 games, approximately 90.5% of actions were still random; after 10,000,
approximately 36.8% were random. The new schedule decreases exploration from
100% to 1% over 100,000 environment transitions, and `--eval` uses no exploration.

The following are deliberate learning-design changes, not claims that the old
Bellman update was mathematically invalid:

- Restrict both action selection and future-value targets to legal moves.
- Use score gained / 100 as reward. Repeated corner, monotonicity, empty-cell,
  and terminal bonuses previously optimized a different objective from score.
- Use Double DQN targets and Huber loss, with gradient-norm clipping.
- Sample minibatches directly from a bounded replay buffer and construct tensors
  in contiguous memory instead of shuffling the entire buffer each update.

The 16 → 256 → 128 → 128 → 4 network and log2(tile)/17 input encoding remain.
The implementation follows the target/replay principles in the
[PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
and the online-selection/target-evaluation separation in the
[Double DQN paper](https://arxiv.org/abs/1509.06461).

## Correctness checks

The native Apple Silicon Release build passed 21 focused checks across two CTest
executables. Tests use explicit failure checks and remain active with `NDEBUG`.

- Twelve engine checks cover all directions, single-merge rules, score accounting,
  legal masks, no spawning or RNG consumption on invalid moves, terminal states
  after spawning, initial/spawned maximum tiles, validation, encoding, and seeds.
- Nine training checks cover exact Double DQN target values, illegal-action
  masking, detached and finite terminal targets (including infinite unused
  values), fixed-minibatch loss reduction, finite gradients and weight updates
  through all four layers, target synchronization, episode/step counters,
  interruptions, checkpoint restoration, and deterministic evaluation.
- Checkpoint tests verify exact policy/target weights and configuration, trainer
  RNG restoration, and an equivalent next Adam update after loading. The replay
  buffer is deliberately not saved and must warm up again.
- Evaluation tests verify unchanged policy/target weights, training counters,
  replay size, epsilon, and training RNG.

Application checks also passed: headless SIGINT saved and exited; evaluation did
not change any checkpoint/summary files; resume advanced the completed-game
counter; malformed checkpoints failed without being overwritten; invalid CLI
options failed. The integrated graphical app completed five games, 608
transitions, and 89 optimizer updates, then saved successfully. An SDL harness
also checked rendering, save-key events, and window-close events.

Run the deterministic regression suites with:

```sh
./build/tools/CMake.app/Contents/bin/cmake --build build --parallel 4
./build/tools/CMake.app/Contents/bin/ctest --test-dir build --output-on-failure
```

## Measured playing strength

Configuration: the defaults in `training.cpp`, CPU, one Torch thread, training
seeds 42 and 7. Each model trained for 5,000 games: an initial 1,000-game run,
then a 4,000-game continuation, with replay warming up again at that boundary.
Seed 42 used 1,067,026 transitions / 266,629 optimizer updates; seed 7 used
1,074,187 transitions / 268,419 updates.

The final evaluation used 200 independent game seeds, 2,000,000–2,000,199, for
every policy. Evaluation performed no learning, exploration, or checkpoint
saving. The random baseline sampled uniformly among legal moves. Untrained
baselines used each model's exact seed-specific initial weights. No game was
truncated.

| Policy | Mean score | Median score | Games reaching 512+ | Games reaching 1024+ | Games reaching 2048+ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random legal moves | 1,057.72 | 1,042 | 0 / 200 | 0 / 200 | 0 / 200 |
| Untrained, seed 42 | 2,374.78 | 2,414 | 3 / 200 | 0 / 200 | 0 / 200 |
| Trained 5,000 games, seed 42 | 4,278.42 | 3,750 | 74 / 200 | 2 / 200 | 0 / 200 |
| Untrained, seed 7 | 832.98 | 696 | 0 / 200 | 0 / 200 | 0 / 200 |
| Trained 5,000 games, seed 7 | 4,118.68 | 3,716 | 71 / 200 | 3 / 200 | 0 / 200 |

An untrained network can already outperform random moves by repeatedly
preferring particular directions. This is why comparing only against random
play is insufficient. Both trained runs beat their own initialization in the
final evaluation; the seed-42 mean increased by about 80%, and seed-7 by about
394%.

For transparency, earlier diagnostic evaluation used seeds
1,000,000–1,000,199 and produced these mean scores:

| Training seed | Before training | After 1,000 games | After 5,000 games |
| --- | ---: | ---: | ---: |
| 42 | 2,309.46 | 2,103.38 | 4,143.20 |
| 7 | 754.36 | 2,577.84 | 4,515.34 |

The seed-42 run had not improved after 1,000 games. Improvement was not assumed
from a changing loss. Hyperparameters were not changed between these runs.
Two training seeds and 200 final evaluation games are useful evidence of
learning, not a guarantee about every seed, longer training, or winning 2048.

## Seed-42 continuation to 55,000 games

The original seed-42 5,000-game checkpoint completed 50,000 additional training
games. Evaluation then reused the same 200 seeds, 2,000,000–2,000,199, with no
truncated games. These results measure longitudinal progress on the same
benchmark; a fresh held-out evaluation remains unmeasured.

| Policy | Mean score | Median score | Games reaching 512+ | Games reaching 1024+ | Games reaching 2048+ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trained 55,000 games, seed 42 | 5,547.32 | 4,964 | 107 / 200 | 24 / 200 | 2 / 200 |

The mean score increased by 29.7% from the 5,000-game checkpoint's 4,278.42.
The checkpoint is retained locally in the ignored
`build/runs/train-50000-20261012-173329/visible/model.pt`.

## Artifacts and reproduction

Per-game CSVs, logs, and checkpoints are retained locally in the ignored
`build/verification/` directory:

- `trained-42.pt` and `trained-7.pt`: the final trained models.
- `untrained.pt` and `untrained-7.pt`: matching initial weights.
- `trained-42-1000.pt` and `trained-7-1000.pt`: intermediate checkpoints.
- `training-*.csv`: training metrics; `final-*.csv`: final evaluation games.
- `cli-checks.log`: application-level verification.

Watch a verified model without further training:

```sh
./build/2048_ai --eval --model build/verification/trained-42.pt --episodes 5
```

Reproduce a seed-42 experiment in a separate directory:

```sh
mkdir -p build/reproduction
./build/2048_ai --fresh --no-render --episodes 0 --seed 42 --model build/reproduction/initial.pt
./build/2048_ai --fresh --no-render --episodes 1000 --seed 42 --model build/reproduction/trained.pt --quiet
./build/2048_ai --no-render --episodes 4000 --model build/reproduction/trained.pt --quiet
./build/2048_ai --eval --no-render --episodes 200 --seed 2000000 --model build/reproduction/initial.pt --csv build/reproduction/initial.csv --quiet
./build/2048_ai --eval --no-render --episodes 200 --seed 2000000 --model build/reproduction/trained.pt --csv build/reproduction/trained.csv --quiet
./build/2048_ai --eval --random --no-render --episodes 200 --seed 2000000 --csv build/reproduction/random.csv --quiet
```

Exact numerical reproducibility is expected on the same CPU/toolchain. Other
devices, library versions, or C++ random-distribution implementations can differ.
