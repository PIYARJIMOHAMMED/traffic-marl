"""
Compatibility test for trained PPO agents.

This test:
1. Does NOT execute at import time.
2. Creates the current TrafficEnvironment.
3. Finds a PPO checkpoint compatible with the environment observation size.
4. Validates model observation/action spaces.
5. Runs a short inference episode.
6. Skips cleanly if no compatible checkpoint exists.

This prevents legacy 5-D / 60-D checkpoints from breaking the whole
pytest collection process.
"""

from pathlib import Path
import sys

import numpy as np
import pytest
from stable_baselines3 import PPO


# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rl.environments.traffic_env import TrafficEnvironment


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "models"

# Prefer current/final models first.
MODEL_PRIORITY = [
    "ppo_marl_final.zip",
    "ppo_marl_120k_steps.zip",
    "ppo_marl_100000_steps.zip",
    "ppo_marl_250000_steps.zip",
    "ppo_marl_200000_steps.zip",
    "ppo_marl_150000_steps.zip",
    "ppo_smoke_test.zip",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_environment():
    """Create the current traffic environment."""
    return TrafficEnvironment(
        scenario_name="baseline",
        render_mode=None,
    )


def find_compatible_model(env):
    """
    Find a PPO checkpoint whose observation and action spaces match
    the current environment.

    Legacy checkpoints are ignored rather than causing pytest failure.
    """

    if not MODELS_DIR.exists():
        return None, "models directory does not exist"

    candidates = []

    # First try known/current model names.
    for filename in MODEL_PRIORITY:
        path = MODELS_DIR / filename
        if path.exists():
            candidates.append(path)

    # Then inspect any other PPO zip files.
    for path in sorted(MODELS_DIR.glob("*.zip")):
        if path not in candidates:
            candidates.append(path)

    env_obs_shape = env.observation_space.shape
    env_action_nvec = getattr(env.action_space, "nvec", None)

    for model_path in candidates:

        try:
            model = PPO.load(
                str(model_path),
                device="cpu",
            )

            model_obs_shape = model.observation_space.shape

            # Observation-space compatibility.
            if model_obs_shape != env_obs_shape:
                continue

            # Action-space compatibility.
            if env_action_nvec is not None:
                model_nvec = getattr(
                    model.action_space,
                    "nvec",
                    None,
                )

                if model_nvec is None:
                    continue

                if not np.array_equal(
                    np.asarray(model_nvec),
                    np.asarray(env_action_nvec),
                ):
                    continue

            return model_path, None

        except Exception:
            # Ignore corrupt/incompatible/legacy checkpoints.
            continue

    return (
        None,
        (
            f"no compatible PPO checkpoint found "
            f"for observation shape {env_obs_shape}"
        ),
    )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_environment_observation_shape():
    """The current environment must expose the expected 120-D observation."""

    env = get_environment()

    try:
        assert env.observation_space.shape == (120,), (
            "Current MARL environment should expose a 120-D observation "
            f"but found {env.observation_space.shape}"
        )
    finally:
        env.close()


def test_environment_action_space():
    """The current environment must expose four binary intersection actions."""

    env = get_environment()

    try:
        assert hasattr(env.action_space, "nvec")

        assert np.array_equal(
            np.asarray(env.action_space.nvec),
            np.asarray([2, 2, 2, 2]),
        ), (
            "Expected MultiDiscrete([2,2,2,2]) action space, "
            f"found {env.action_space}"
        )

    finally:
        env.close()


def test_trained_ppo_inference():
    """
    Load a checkpoint compatible with the current environment and
    perform a short inference run.
    """

    env = get_environment()

    try:
        model_path, reason = find_compatible_model(env)

        if model_path is None:
            pytest.skip(reason)

        print()
        print("=" * 70)
        print("TRAINED PPO COMPATIBILITY TEST")
        print("=" * 70)
        print(f"Model       : {model_path}")
        print(f"Observation : {env.observation_space.shape}")
        print(f"Action      : {env.action_space}")
        print("=" * 70)

        model = PPO.load(
            str(model_path),
            device="cpu",
        )

        # Explicit compatibility checks.
        assert model.observation_space.shape == env.observation_space.shape

        obs, info = env.reset(seed=42)

        assert obs.shape == (120,)
        assert np.all(np.isfinite(obs))

        # Run a short inference test.
        for _ in range(10):

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            assert action is not None

            obs, reward, terminated, truncated, info = env.step(action)

            assert obs.shape == (120,)
            assert np.all(np.isfinite(obs))
            assert np.isfinite(reward)

            if terminated or truncated:
                obs, info = env.reset(seed=42)

        print("Inference test PASSED.")
        print("=" * 70)

    finally:
        env.close()