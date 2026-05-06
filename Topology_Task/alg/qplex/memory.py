from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from functools import partial

from common.imports import *
from common.utils import add_trailing_dim, nested_to_numpy, recursive_index_numpy

def to_torch(object: object, device: th.device):
    if isinstance(object, dict):
        return {key: to_torch(value, device=device) for key, value in object.items()}
    return th.from_numpy(object).to(device)
    
class Buffer:
    def __init__(self, agent, envs, args, state_dim, device):
        self.obs_space = envs.observation_space[agent]
        self.act_space = envs.action_space[agent]

        self.to_torch = partial(to_torch, device=device)

        self.capacity = args.buffer_size
        self.n_envs = args.n_envs
        self.batch_size = args.batch_size
        self.sampling_seed = args.seed
        self.idx, self.full = 0, False

        self.capacity //= self.n_envs    # Fix capacity based on n° of envs

        self.base_shape = (self.capacity, self.n_envs,)
        self.obs = None
        self.action = None
        self.reward = None
        self.next_obs = None
        self.done = None
        self.state = None
        self.next_state = None

        self.attributes = ['obs', 'action', 'reward', 'next_obs', 'done', 'state', 'next_state']

    def _allocate_like(self, value):
        if isinstance(value, dict):
            return {key: self._allocate_like(val) for key, val in value.items()}
        return np.zeros(self.base_shape + tuple(value.shape[1:]), dtype=value.dtype)

    def _assign_at_idx(self, storage, value):
        if isinstance(storage, dict):
            for key in storage:
                self._assign_at_idx(storage[key], value[key])
        else:
            storage[self.idx] = value

    def _store_attr(self, name, value):
        value = nested_to_numpy(value)
        if getattr(self, name) is None:
            setattr(self, name, self._allocate_like(value))
        self._assign_at_idx(getattr(self, name), value)

    def _set_sampling(self):
        self.sampling_seed += 1
        np.random.seed(self.sampling_seed)

    def store(self, obs, action, reward, next_obs, done, state, next_state):
        if self.idx == self.capacity: 
            self.full, self.idx = True, 0

        self._store_attr("obs", obs)
        self._store_attr("action", action)
        self._store_attr("reward", np.asarray(reward, dtype=np.float32))
        self._store_attr("next_obs", next_obs)
        self._store_attr("done", np.asarray(done, dtype=np.float32))
        self._store_attr("state", state)
        self._store_attr("next_state", next_state)

        self.idx += 1    

    def sample(self):
        self._set_sampling()    # Set seed for sequential sampling in MARL (i.e. sample same idxs for each agent)

        idxs = self.get_sample_idxs()

        batch = {a: recursive_index_numpy(getattr(self, a), idxs) for a in self.attributes}

        # Fix shapes for training
        batch['action'] = add_trailing_dim(batch['action'])
        batch['reward'] = add_trailing_dim(batch['reward'])
        batch['done'] = add_trailing_dim(batch['done'])

        return {k: self.to_torch(v) for k, v in batch.items()}

    def get_sample_idxs(self):
        # Choice avoids sampling with replacement
        sample_size = min(self.batch_size, self.size)

        # (experience, env_id) idxs
        return (np.random.choice(np.arange(0, self.size), size=sample_size, replace=False), 
                np.random.randint(0, high=self.n_envs, size=sample_size)) 

    def clear(self) -> None:
        self.idx, self.full = 0, False
    
    @property
    def size(self) -> int:
        return self.capacity if self.full else self.idx
