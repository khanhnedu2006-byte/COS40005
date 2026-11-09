"""Write PPO rollout progress as JSON lines to stderr."""

import json
import sys

from stable_baselines3.common.callbacks import BaseCallback


class PPOProgress(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episodes = []

    def _emit(self, event, **fields):
        print(json.dumps({"event": event, **fields}), file=sys.stderr, flush=True)

    def _on_training_start(self):
        self.episodes = []
        self._emit("training_start", steps=self.num_timesteps)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "episode" in info:
                self.episodes.append((info["episode"]["r"], bool(info.get("success"))))
        return True

    def _on_rollout_end(self):
        count = len(self.episodes)
        self._emit(
            "rollout",
            steps=self.num_timesteps,
            completed_updates=self.model._n_updates,
            episodes=count,
            mean_episode_reward=round(sum(r for r, _ in self.episodes) / count, 3) if count else None,
            success_rate=round(sum(s for _, s in self.episodes) / count, 3) if count else None,
        )
        self.episodes = []

    def _on_training_end(self):
        self._emit("training_end", steps=self.num_timesteps, completed_updates=self.model._n_updates)
