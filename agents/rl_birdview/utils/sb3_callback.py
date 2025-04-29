import numpy as np
import time
from pathlib import Path
import wandb
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from omegaconf import OmegaConf


class SB3Callback(BaseCallback):
    def __init__(self, cfg, vec_env: VecEnv):
        super(SB3Callback, self).__init__(verbose=1)

        # save_dir = Path.cwd()
        # self._save_dir = save_dir
        self._video_path = Path('video')
        self._video_path.mkdir(parents=True, exist_ok=True)
        self._ckpt_dir = Path('ckpt')
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        # wandb.init(project=cfg.wb_project, dir=save_dir, name=cfg.wb_runname)
        wandb.init(project=cfg.wb_project, name=cfg.wb_name, notes=cfg.wb_notes, tags=cfg.wb_tags)
        wandb.config.update(OmegaConf.to_container(cfg))

        wandb.save('./config_agent.yaml')
        wandb.save('.hydra/*')

        self.vec_env = vec_env

        self._eval_step = 5000
        self._buffer_step = int(1e5)
        self.ep_stat_buffer = deque(maxlen=100)

    def _init_callback(self):
        self.n_epoch = 0
        self._last_time_buffer = self.model.num_timesteps
        self._last_time_eval = self.model.num_timesteps

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self):
        pass

    def _on_rollout_end(self):
        # save rollout statistics
        avg_ep_stat = self.get_avg_ep_stat(self.model.ep_info_buffer, prefix="rollout/")
        wandb.log(avg_ep_stat, step=self.model.num_timesteps)

        print(f"n_epoch: {self.n_epoch}, num_timesteps: {self.model.num_timesteps}")
        # save time
        time_elapsed = time.time() - self.model.start_time
        wandb.log(
            {
                "time/n_epoch": self.n_epoch,
                "time/sec_per_epoch": time_elapsed / (self.n_epoch + 1),
                "time/fps": self.model.num_timesteps / time_elapsed,
                "time/n_updates": self.model._n_updates,
            },
            step=self.model.num_timesteps,
        )

        # evaluate and save checkpoint
        if self.model.num_timesteps - self._last_time_eval >= self._eval_step:
            self._last_time_eval = self.model.num_timesteps
            eval_video_path = (
                self._video_path / f"eval_{self.model.num_timesteps}.mp4"
            ).as_posix()
            avg_ep_stat, ep_events = self.evaluate_policy(
                self.vec_env, self.model.policy, eval_video_path
            )
            # log to wandb
            wandb.log(
                {f"video/{self.model.num_timesteps}": wandb.Video(eval_video_path)},
                step=self.model.num_timesteps,
            )
            wandb.log(avg_ep_stat, step=self.model.num_timesteps)

            # save model
            ckpt_path = (
                self._ckpt_dir / f"ckpt_{self.model.num_timesteps}.pth"
            ).as_posix()
            self.model.save(ckpt_path)
            wandb.save(f"{ckpt_path}")
        self.n_epoch += 1

    @staticmethod
    def evaluate_policy(env: VecEnv,
                        policy: BasePolicy,
                        video_path: str,
                        min_eval_steps: int = 3000):
        policy = policy.eval()
        t0 = time.time()
        for i in range(env.num_envs):
            env.set_attr('eval_mode', True, indices=i)
        obs = env.reset()

        list_render = []
        ep_stat_buffer = []
        ep_events = {}
        for i in range(env.num_envs):
            ep_events[f'venv_{i}'] = []

        n_step = 0
        n_timeout = 0
        env_done = np.array([False]*env.num_envs)
        # while n_step < min_eval_steps:
        while n_step < min_eval_steps or not np.all(env_done):
            actions, state = policy.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(actions)

            list_render.append(env.render(mode='rgb_array'))

            n_step += 1
            env_done |= done

            for i in np.where(done)[0]:
                ep_stat_buffer.append(info[i]['episode_stat'])
                ep_events[f'venv_{i}'].append(info[i]['episode_event'])
                n_timeout += int(info[i]['timeout'])

        # conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
        encoder = ImageEncoder(video_path, list_render[0].shape, 30, 30)
        for im in list_render:
            encoder.capture_frame(im)
        encoder.close()

        avg_ep_stat = SB3Callback.get_avg_ep_stat(ep_stat_buffer, prefix='eval/')
        avg_ep_stat['eval/eval_timeout'] = n_timeout

        duration = time.time() - t0
        avg_ep_stat['time/t_eval'] = duration
        avg_ep_stat['time/fps_eval'] = n_step * env.num_envs / duration

        for i in range(env.num_envs):
            env.set_attr('eval_mode', False, indices=i)
        obs = env.reset()
        return avg_ep_stat, ep_events

    @staticmethod
    def get_avg_ep_stat(ep_stat_buffer, prefix=''):
        avg_ep_stat = {}
        if len(ep_stat_buffer) > 0:
            for ep_info in ep_stat_buffer:
                for k, v in ep_info.items():
                    k_avg = f'{prefix}{k}'
                    if k_avg in avg_ep_stat:
                        avg_ep_stat[k_avg] += v
                    else:
                        avg_ep_stat[k_avg] = v

            n_episodes = float(len(ep_stat_buffer))
            for k in avg_ep_stat.keys():
                avg_ep_stat[k] /= n_episodes
            avg_ep_stat[f'{prefix}n_episodes'] = n_episodes

        return avg_ep_stat
