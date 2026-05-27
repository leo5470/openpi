"""LIBERO-Plus rollout client for openpi-served policies (e.g. pi05_libero).

Fork of openpi/examples/libero/main.py adapted per the deployment plan §2.
Deltas from the upstream file (kept intentionally small):

  1. num_trials_per_task defaults to 1 -- LIBERO-Plus encodes the variation in
     the BDDL itself, so each of the thousands of tasks per suite gets exactly
     one rollout, not 50.
  2. Per-rollout JSONL logging ({suite, task_idx, task_name, success, steps,
     wall_seconds, video_path}) flushed after every episode, so a run is
     resumable/aggregatable and feeds aggregate_libero_plus.py directly.
  3. Videos are named by task.name (the perturbed BDDL stem), NOT the language
     instruction -- many Plus perturbations of one task share a language
     string, so the upstream language-based filename would clobber videos.
  4. bddl_file_name is passed to OffScreenRenderEnv as a *str*, not a Path:
     LIBERO-Plus's ControlEnv does `"_view_" in bddl_file_name` (env_wrapper.py
     L207) which raises TypeError on a pathlib.Path. Vanilla LIBERO never did
     string ops on the name, so this only matters for the Plus env.

Everything else (180-deg image rotation, resize_with_pad to 224, the 8-dim
eef-pos + axisangle + gripper-qpos state, replan-every-5 of a >=5 chunk,
per-suite max_steps) is unchanged -- it matches how pi05_libero was trained.
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib
import time

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    # Model server
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # LIBERO-Plus environment
    task_suite_name: str = "libero_spatial"  # libero_spatial | libero_object | libero_goal | libero_10
    num_steps_wait: int = 10  # steps to let objects settle in sim before acting
    num_trials_per_task: int = 1  # Plus: one rollout per (already-perturbed) task

    # Utils
    video_out_path: str = "data/libero_plus/videos"  # per-suite subdir is appended
    results_jsonl: str = "data/libero_plus/rollouts.jsonl"  # consumed by aggregate_libero_plus.py
    save_video: bool = True  # set False to skip mp4 writing on a full 10k run

    seed: int = 7  # affects object positions even with a fixed init state


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name} ({num_tasks_in_suite} tasks)")

    video_dir = pathlib.Path(args.video_out_path) / args.task_suite_name
    video_dir.mkdir(parents=True, exist_ok=True)
    results_path = pathlib.Path(args.results_jsonl)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    results_file = results_path.open("a", buffering=1)  # line-buffered: resumable mid-run
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        for episode_idx in range(args.num_trials_per_task):
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            done = False  # guard: referenced below even if the first step raises
            episode_steps = 0
            replay_images = []
            wall_start = time.time()

            while t < max_steps + args.num_steps_wait:
                try:
                    # Do nothing for the first few steps: the sim drops objects
                    # and we wait for them to settle.
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # rotate 180 deg to match training preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    if args.save_video:
                        replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"want to replan every {args.replan_steps} steps, but policy "
                            f"only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    episode_steps += 1
                    if done:
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:  # noqa: BLE001 - one bad task must not kill the suite
                    logging.error(f"task {task_id} ({task.name}) raised: {e}")
                    break

            total_episodes += 1
            wall_seconds = round(time.time() - wall_start, 2)

            video_path = ""
            if args.save_video and replay_images:
                suffix = "success" if done else "failure"
                video_path = str(video_dir / f"{task.name}_{suffix}.mp4")
                imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)

            # One JSONL line per rollout -> aggregate_libero_plus.py join key is task_name.
            results_file.write(json.dumps({
                "suite": args.task_suite_name,
                "task_idx": task_id,
                "task_name": task.name,
                "success": bool(done),
                "steps": episode_steps,
                "wall_seconds": wall_seconds,
                "video_path": video_path,
            }) + "\n")

        if (task_id + 1) % 50 == 0 or task_id + 1 == num_tasks_in_suite:
            logging.info(
                f"[{args.task_suite_name}] {task_id + 1}/{num_tasks_in_suite} tasks, "
                f"running success rate {total_successes / max(total_episodes, 1) * 100:.1f}%"
            )

    results_file.close()
    logging.info(
        f"Done {args.task_suite_name}: {total_successes}/{total_episodes} = "
        f"{total_successes / max(total_episodes, 1) * 100:.1f}% -> {results_path}"
    )


def _get_libero_env(task, resolution, seed):
    """Initialize the LIBERO-Plus env and return (env, language)."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    # str(): Plus's ControlEnv parses perturbation codes via `"_view_" in name`,
    # which raises TypeError on a Path (env_wrapper.py L207).
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # seed affects object positions even with a fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """Copied from robosuite transform_utils."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
