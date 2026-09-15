"""Train a MaskablePPO snake on one curriculum stage.

    python -m rl.train --stage solo --timesteps 2000000
    python -m rl.train --stage duel_weak --timesteps 2000000 --from runs/solo/model.zip
    python -m rl.train --stage four_weak --timesteps 2000000 --from runs/duel_weak/model.zip

Each run writes runs/<run name>/:
- model.zip: the final model
- best/best_model.zip: the model with the best evaluation reward so far
- checkpoints/: snapshots during training
- tb/: TensorBoard logs (tensorboard --logdir runs)
- stage.json: what was trained, for how long, and how fast
"""

import argparse
import json
import time
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

import torch

from rl.envs import STAGES, MaskCachingVecEnv, make_env
from rl.model import new_model

RUNS = Path("runs")


class StopAfter(BaseCallback):
    """Ends training once `minutes` have passed, so a stage fits its time budget."""

    def __init__(self, minutes: float):
        super().__init__()
        self.seconds = minutes * 60

    def _on_training_start(self) -> None:
        self.start = time.perf_counter()

    def _on_step(self) -> bool:
        return time.perf_counter() - self.start < self.seconds


def build_vec_env(stage: str, n_envs: int, seed: int, subprocess: bool = True) -> VecEnv:
    """n_envs copies of the stage's environment, each in its own process if `subprocess`.

    Every step waits for the behavior-tree opponents to decide, and that's
    most of the work, so spreading environments over CPU cores is what
    makes training fast.
    """
    env_fns = [lambda: make_env(stage) for _ in range(n_envs)]
    if subprocess and n_envs > 1:
        vec_env: VecEnv = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        vec_env = DummyVecEnv(env_fns)
    vec_env.seed(seed)
    return MaskCachingVecEnv(vec_env)


def train(
    stage: str,
    timesteps: int,
    n_envs: int = 16,
    start_from: str | Path | None = None,
    device: str = "auto",
    run_name: str | None = None,
    seed: int = 0,
    eval_every: int = 200_000,
    eval_games: int = 50,
    subprocess: bool = True,
    runs_dir: Path = RUNS,
    threads: int | None = None,
    max_minutes: float | None = None,
) -> Path:
    """Train for `timesteps` moves on `stage`. Returns the saved model's path.

    `threads` caps PyTorch's CPU threads. By default it takes every core,
    which competes with the environment processes for the same cores.
    """
    if threads:
        torch.set_num_threads(threads)
    run_dir = Path(runs_dir) / (run_name or stage)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = build_vec_env(stage, n_envs, seed, subprocess)
    # Evaluation games use seeds far from the training ones.
    eval_env = build_vec_env(stage, min(n_envs, 8), seed + 1_000_000, subprocess)
    tensorboard_log = str(run_dir / "tb")

    if start_from:
        model = MaskablePPO.load(start_from, env=env, device=device, tensorboard_log=tensorboard_log)
    else:
        model = new_model(env, device=device, tensorboard_log=tensorboard_log, seed=seed)

    every = max(eval_every // n_envs, 1)  # callbacks count steps per environment
    callbacks = [
        CheckpointCallback(save_freq=every, save_path=str(run_dir / "checkpoints"), name_prefix="model"),
        MaskableEvalCallback(
            eval_env,
            n_eval_episodes=eval_games,
            eval_freq=every,
            best_model_save_path=str(run_dir / "best"),
            log_path=str(run_dir / "eval"),
            deterministic=True,
            verbose=0,
        ),
    ]
    if max_minutes:
        callbacks.append(StopAfter(max_minutes))

    steps_before = model.num_timesteps if start_from else 0
    start = time.perf_counter()
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        tb_log_name=stage,
        reset_num_timesteps=start_from is None,
    )
    elapsed = time.perf_counter() - start

    model_path = run_dir / "model.zip"
    model.save(model_path)
    trained = int(model.num_timesteps) - steps_before
    summary = {
        "stage": stage,
        "description": STAGES[stage].description,
        "timesteps": trained,
        "timestep_budget": timesteps,
        "minute_budget": max_minutes,
        "total_timesteps": int(model.num_timesteps),
        "started_from": str(start_from) if start_from else None,
        "n_envs": n_envs,
        "device": str(model.device),
        "torch_threads": torch.get_num_threads(),
        "seed": seed,
        "seconds": round(elapsed, 1),
        "steps_per_second": round(trained / elapsed),
    }
    (run_dir / "stage.json").write_text(json.dumps(summary, indent=2))
    env.close()
    eval_env.close()
    print(json.dumps(summary, indent=2))
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=list(STAGES))
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--from", dest="start_from", help="continue training this model.zip")
    parser.add_argument("--device", default="auto", help="cuda, cpu or auto")
    parser.add_argument("--run-name", help="folder under runs/ (default: the stage name)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=200_000)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--threads", type=int, help="PyTorch CPU threads (default: all cores)")
    parser.add_argument("--max-minutes", type=float, help="stop and save after this long")
    args = parser.parse_args()
    train(
        args.stage,
        args.timesteps,
        n_envs=args.envs,
        start_from=args.start_from,
        device=args.device,
        run_name=args.run_name,
        seed=args.seed,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        threads=args.threads,
        max_minutes=args.max_minutes,
    )


if __name__ == "__main__":
    main()
