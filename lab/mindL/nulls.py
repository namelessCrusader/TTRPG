"""The null models (learned-mind plan, section 7; Buddy rule 6: no claim
without one). Same world, same reader numbers, same reward (the drop in
weighted drive error), same decisions to learn from.

N1  Bandit: a contextual linear bandit over the acts. It predicts the
    IMMEDIATE reward of an option (the drop of drive error over that one act)
    with the same crossed features as the mind's actor (W[verb] . z +
    U[verb, target]), learns by the normalised delta rule, and chooses by a
    softmax (tau 0.005 and lr 0.1: the best of five settings on seed 0). No
    time, no value of the next state, no modules.

N2  PPO (stable-baselines3, CPU only): one monolithic policy, learned online
    over ONE character's decisions in a lifetime. FocalEnv is the gymnasium
    wrapper: the lifetime runs in a worker thread with the rest of the cast on
    `rules`; when the focal character must choose, the thread hands the menu
    to PPO and waits for its act. An observation is the menu, at most K
    options (the menu is 36 on average, 134 at most; past K the first K - 1
    and "wait" are kept), each row the reader's raw numbers (reader.raw: verb
    and target one-hot, then the option numbers and the drive needs) and a
    1 for a real option. An episode is one night (it ends at night's end, or
    at death). The reward of a step is the drop in drive error from one
    choice to the next.
    The policy scores each option with ONE shared network (raw -> 64 -> 1;
    rows that are not options get -1e8), so its input is the same as the
    mind's and the bandit's; the value network reads the whole menu
    (flattened, 64-64). This is the fair version of "monolithic": no menu
    slot has a fixed meaning.
"""
from __future__ import annotations

import queue
import threading

import numpy as np

from lab.mindL import drives as H
from lab.mindL import reader
from lab.mindL.mind import Agent, PerCharacter, softmax
from lab.world2.lifetime import Cast, lifetime
from lab.world2.policy import Rules


# ---------------------------------------------------------------- N1 bandit
class Bandit(Agent):
    def __init__(self, name, kind, seed=0, tau=0.005, lr=0.1):
        super().__init__(name, kind, seed)
        self.W = np.zeros((reader.NV, reader.ZDIM))
        self.U = np.zeros((reader.NV, reader.NT))
        self.tau, self.lr = tau, lr

    def predict(self, menu):
        return np.einsum("nz,nz->n", self.W[menu.v], menu.Z) + self.U[menu.v, menu.t]

    def pick(self, w, p, opts, menu, need, s):
        q = self.predict(menu)
        pi = softmax(q / self.tau)
        a = int(self.rng.choice(len(opts), p=pi))
        return a, pi

    def learn(self, w, p, evs, need, E, s, terminal):
        st = self.step
        r = H.reward(st.E, E)
        v, t, z = st.menu.v[st.a], st.menu.t[st.a], st.menu.Z[st.a]
        err = r - (self.W[v] @ z + self.U[v, t])
        norm = z @ z + 1.0
        self.W[v] += self.lr * err * z / norm
        self.U[v, t] += self.lr * err / norm


def make_bandit(seed=0, **cfg):
    return PerCharacter(lambda n, k: Bandit(n, k, seed, **cfg), "bandit")


# ---------------------------------------------------------------- N2 PPO
K = 128


def menu_obs(opts, need, k=K):
    """(the flat observation, the option index of each row)."""
    idx = list(range(len(opts)))
    if len(idx) > k:
        idx = idx[:k - 1] + [idx[-1]]
    X = reader.raw(reader.encode([opts[i] for i in idx], need))
    obs = np.zeros((k, reader.RAW + 1), dtype=np.float32)
    obs[:len(idx), :reader.RAW] = X
    obs[:len(idx), reader.RAW] = 1.0
    return obs.ravel(), idx


class _Bridge:
    """The focal character's policy inside the world thread."""

    def __init__(self, env):
        self.env, self.name, self.ended = env, "ppo", True

    def probs(self, w, p, opts):
        obs, idx = menu_obs(opts, H.needs(w, p), self.env.k)
        self.ended = False
        self.env.truncated += len(opts) > self.env.k
        self.env._out.put(("decide", obs, self.env.err(w, p)))
        a = self.env._in.get()
        if a is None:
            raise SystemExit
        i = idx[a] if a < len(idx) else idx[-1]
        pc = [0.0] * len(opts)
        pc[i] = 1.0
        return pc, pc

    def _end(self, w, p):
        if not self.ended:
            self.ended = True
            self.env._out.put(("end", self.env.err(w, p)))

    def on_death(self, w, person, event):
        if person.name == self.env.focal:
            self._end(w, person)

    def night_end(self, w, night):
        self._end(w, w.people[self.env.focal])


def make_env(focal, nights, seed, flags="all", k=K, on_night=None, turns=300, fade=1.0, cast=Cast, err=None):
    """`cast(default, by)` makes the lifetime's Cast (a test may record more). `err(w, p)`: the error
    whose drop is the reward (default the drive error; L4 swaps in a designer's)."""
    import gymnasium as gym

    class FocalEnv(gym.Env):
        metadata: dict = {}

        def __init__(self):
            self.focal, self.nights, self.seed_, self.k = focal, nights, seed, k
            self.observation_space = gym.spaces.Box(-5.0, 5.0, shape=(k * (reader.RAW + 1),), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(k)
            self._in, self._out = queue.Queue(), queue.Queue()
            self.thread, self.finished, self.stats, self.E = None, False, None, 0.0
            self.decisions, self.truncated = 0, 0
            self.err = err or (lambda w, p: w.drive_error(p))
            self._zero = np.zeros(self.observation_space.shape, dtype=np.float32)

        def _world(self):
            try:
                c = cast(Rules(), {focal: _Bridge(self)})
                _, stats = lifetime(c, nights, seed, turns, flags, on_night=on_night, fade=fade)
                self._out.put(("done", stats))
            except SystemExit:
                pass
            except BaseException as e:          # hand it to the learner's thread
                self._out.put(("error", e))

        def _next(self):
            msg = self._out.get()
            if msg[0] == "error":
                raise msg[1]
            return msg

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            if self.finished:
                return self._zero, {}
            if self.thread is None:
                self.thread = threading.Thread(target=self._world, daemon=True)
                self.thread.start()
            msg = self._next()
            if msg[0] == "done":
                self.finished, self.stats = True, msg[1]
                return self._zero, {}
            assert msg[0] == "decide", msg[0]
            self.E = msg[2]
            return msg[1], {}

        def step(self, action):
            self._in.put(int(action))
            self.decisions += 1
            msg = self._next()
            if msg[0] == "decide":
                r, self.E = self.E - msg[2], msg[2]
                return msg[1], r, False, False, {}
            assert msg[0] == "end", msg[0]
            return self._zero, self.E - msg[1], True, False, {}

        def close(self):
            if self.thread is not None and self.thread.is_alive():
                self._in.put(None)

    return FocalEnv()


def menu_policy(k, f):
    """An SB3 ActorCriticPolicy whose actor scores each menu row with one
    shared network, and masks the rows that are not options."""
    import torch.nn as nn
    from stable_baselines3.common.policies import ActorCriticPolicy

    class OptionScorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(f, 64), nn.Tanh(), nn.Linear(64, 1))
            nn.init.orthogonal_(self.net[2].weight, gain=0.01)
            nn.init.zeros_(self.net[2].bias)

        def forward(self, x):
            x = x.view(-1, k, f + 1)
            logits = self.net(x[..., :f]).squeeze(-1)
            return logits.masked_fill(x[..., f] < 0.5, -1e8)

    class MenuPolicy(ActorCriticPolicy):
        def _build(self, lr_schedule):
            super()._build(lr_schedule)
            self.action_net = OptionScorer()
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    return MenuPolicy


PPO_CFG = dict(n_steps=256, batch_size=64, n_epochs=4, learning_rate=1e-3, gamma=0.94, gae_lambda=0.95,
               ent_coef=0.01, clip_range=0.2)
# n_steps and learning_rate: the better of two settings on seed 0 (the other was
# SB3's default 512 and 3e-4). gamma per DECISION: an act lasts about 3 turns, and the mind's 0.98 per turn ** 3 = 0.94


def run_ppo(focal, nights, seed=0, flags="all", on_night=None, k=K, threads=2, cast=Cast, err=None, fade=1.0, **cfg):
    """One lifetime with PPO learning online as `focal`. Returns (stats, info)."""
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    torch.set_num_threads(threads)
    env = make_env(focal, nights, seed, flags, k, on_night, fade=fade, cast=cast, err=err)

    class Stop(BaseCallback):
        def _on_step(self):
            return not env.finished

    c = {**PPO_CFG, **cfg}
    model = PPO(menu_policy(k, reader.RAW), env, seed=seed, device="cpu", verbose=0,
                policy_kwargs=dict(net_arch=dict(pi=[], vf=[64, 64])), **c)
    model.learn(total_timesteps=10 ** 9, callback=Stop())
    env.close()
    assert env.finished, "the lifetime did not finish"
    return env.stats, {"decisions": env.decisions, "truncated_menus": env.truncated, "ppo": c, "K": k}
