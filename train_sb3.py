import logging
from pathlib import Path
from typing import Dict

import gym
import hydra
import torch as th
import wandb
from omegaconf import DictConfig
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch import nn

from agents.rl_birdview.utils.rl_birdview_wrapper import RlBirdviewWrapper
from agents.rl_birdview.utils.sb3_callback import SB3Callback
from carla_gym.utils import config_utils
from utils import server_utils

log = logging.getLogger(__name__)


class XtMaCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor based on XtMaCNN for autonomous driving.
    Adapts the provided CNN for use with Stable-Baselines3.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        states_neurons=[256],
    ):
        super(XtMaCNNFeatureExtractor, self).__init__(observation_space, features_dim)

        # Extract dimensions
        n_input_channels = observation_space.spaces["birdview"].shape[0]
        state_dim = observation_space.spaces["state"].shape[0]

        # CNN for processing birdview
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            sample_birdview = th.as_tensor(
                observation_space.spaces["birdview"].sample()[None]
            ).float()
            n_flatten = self.cnn(sample_birdview).shape[1]

        # State processing network
        states_neurons = [state_dim] + states_neurons
        state_linear_layers = []
        for i in range(len(states_neurons) - 1):
            state_linear_layers.append(
                nn.Linear(states_neurons[i], states_neurons[i + 1])
            )
            state_linear_layers.append(nn.ReLU())
        self.state_linear = nn.Sequential(*state_linear_layers)

        # Combined features network
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + states_neurons[-1], 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

        # Initialize weights
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        # Process birdview (normalize pixel values)
        birdview = observations["birdview"].float() / 255.0
        state = observations["state"]

        # Extract CNN features
        cnn_features = self.cnn(birdview)

        # Process state vector
        state_features = self.state_linear(state)

        # Combine features
        combined = th.cat([cnn_features, state_features], dim=1)
        features = self.linear(combined)

        return features


@hydra.main(config_path="config", config_name="train_sb3")
def main(cfg: DictConfig):
    if cfg.kill_running:
        server_utils.kill_carla()
    set_random_seed(cfg.seed, using_cuda=True)

    # start carla servers
    server_manager = server_utils.CarlaServerManager(
        cfg.carla_sh_path, configs=cfg.train_envs
    )
    server_manager.start()

    obs_configs = {
        "hero": {
            "birdview": {
                "module": "birdview.chauffeurnet",
                "width_in_pixels": 192,
                "pixels_ev_to_bottom": 40,
                "pixels_per_meter": 5.0,
                "history_idx": [-16, -11, -6, -1],
                "scale_bbox": True,
                "scale_mask_col": 1.0,
            },
            "speed": {"module": "actor_state.speed"},
            "control": {"module": "actor_state.control"},
            "velocity": {"module": "actor_state.velocity"},
        }
    }
    reward_configs = {
        "hero": {"entry_point": "reward.valeo_action:ValeoAction", "kwargs": {}}
    }
    terminal_configs = {
        "hero": {"entry_point": "terminal.valeo_no_det_px:ValeoNoDetPx", "kwargs": {}}
    }

    config_utils.check_h5_maps(cfg.train_envs, obs_configs, cfg.carla_sh_path)

    def env_maker(config):
        log.info(f'making port {config["port"]}')
        env = gym.make(
            config["env_id"],
            obs_configs=obs_configs,
            reward_configs=reward_configs,
            terminal_configs=terminal_configs,
            host="localhost",
            port=config["port"],
            seed=cfg.seed,
            no_rendering=True,
            **config["env_configs"],
        )
        env = RlBirdviewWrapper(env, ["control", "vel_xy"], True)
        return env

    if cfg.dummy:
        env = DummyVecEnv(
            [
                lambda config=config: env_maker(config)
                for config in server_manager.env_configs
            ]
        )
    else:
        env = SubprocVecEnv(
            [
                lambda config=config: env_maker(config)
                for config in server_manager.env_configs
            ]
        )

    # TODO: Make SB3 algorithm configurable
    agent = TD3(
        "MultiInputPolicy",
        env,
        learning_rate=1e-5,
        buffer_size=50000,
        learning_starts=2000,
        batch_size=256,
        policy_kwargs={
            "features_extractor_class": XtMaCNNFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 512},
            # "net_arch": [dict(pi=[256, 256], qf=[256, 256])],
        },
    )

    last_checkpoint_path = (
        Path(hydra.utils.get_original_cwd()) / "outputs" / "checkpoint.txt"
    )
    if last_checkpoint_path.exists():
        with open(last_checkpoint_path, "r") as f:
            wb_run_path = f.read()
            api = wandb.Api()
            run = api.run(wb_run_path)
            all_ckpts = [f for f in run.files() if "ckpt" in f.name]
            if all_ckpts:
                f = max(
                    all_ckpts, key=lambda x: int(x.name.split("_")[1].split(".")[0])
                )
                log.info(f"Resume checkpoint latest {f.name}")

                f.download(replace=True)
                agent.load(f.name)

    # wandb init
    wb_callback = SB3Callback(cfg, env)
    ckpt_callback = CheckpointCallback(10000, "sb3_ckpt", type(agent).__name__)
    callback = CallbackList([wb_callback, ckpt_callback])

    # save wandb run path to file such that bash file can find it
    last_checkpoint_path = (
        Path(hydra.utils.get_original_cwd()) / "outputs" / "checkpoint.txt"
    )
    with open(last_checkpoint_path, "w") as f:
        f.write(wandb.run.path)

    agent.learn(
        total_timesteps=int(cfg.total_timesteps),
        callback=callback,
        log_interval=4,
        reset_num_timesteps=False,
    )

    server_manager.stop()


if __name__ == "__main__":
    main()
    log.info("train_sb3.py DONE!")
