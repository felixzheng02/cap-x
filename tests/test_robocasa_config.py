"""The shared RoboCasa YAML must pin the local VLM and not force multi-turn.

    cd cap-x && .venv-client/bin/python -m pytest tests/test_robocasa_config.py -v
"""

from __future__ import annotations

from capx.envs.launch import LaunchArgs
from capx.utils.launch_utils import _load_config

CONFIG = "env_configs/robocasa/robocasa.yaml"


def test_shared_yaml_pins_local_qwen_not_the_8110_default():
    args = LaunchArgs(config_path=CONFIG)
    env_factory, config, _ = _load_config(args)
    assert args.model == "qwen3.8-27b"
    assert args.server_url == "http://127.0.0.1:8000/v1/chat/completions"
    assert env_factory["cfg"]["low_level"] == "remote_robot_low_level"
    assert config["use_img_differencing"] is False
    assert "multi_turn_prompt" not in env_factory["cfg"]


def test_img_differencing_keeps_multiturn_prompt():
    args = LaunchArgs(config_path=CONFIG, use_img_differencing=True)
    env_factory, config, _ = _load_config(args)
    assert config["use_img_differencing"] is True
    assert "FINISH" in env_factory["cfg"]["multi_turn_prompt"]
    assert args.visual_differencing_model == "qwen3.8-27b"
    assert args.visual_differencing_model_server_url == (
        "http://127.0.0.1:8000/v1/chat/completions"
    )
