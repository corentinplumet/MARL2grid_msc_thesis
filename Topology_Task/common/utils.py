from argparse import ArgumentTypeError

from .imports import *

seed = 0    # Global variable to store the seed

# Taking np.array from gym and casting into tensor. Observations can either be
# the original flat arrays or nested dicts with flat and graph entries.
def cast_np_to_tensors(obj, device='cpu', dtype=th.float32):
    if isinstance(obj, dict):
        return {
            key: cast_np_to_tensors(value, device=device, dtype=dtype)
            for key, value in obj.items()
        }
    if isinstance(obj, th.Tensor):
        return obj.to(device=device, dtype=dtype)
    return th.tensor(obj, dtype=dtype, device=device)


def get_flat_obs(obs):
    return obs["flat"] if isinstance(obs, dict) and "flat" in obs else obs


def get_state_graph_obs(obs_dict):
    first_agent = sorted(obs_dict.keys())[0]
    obs = obs_dict[first_agent]
    if not isinstance(obs, dict) or "state_graph" not in obs:
        raise ValueError("A GNN critic/mixer requires graph observations, but no state_graph was found.")
    return obs["state_graph"]


def strip_state_graph(obs):
    if isinstance(obs, dict) and "state_graph" in obs:
        return {key: value for key, value in obs.items() if key != "state_graph"}
    return obs


# Creating the joint obs from the tensor dict
def stack_agent_obs_by_env(obs_dict):
    # Just concatenate all agent flat tensors on the last dimension
    return th.cat([get_flat_obs(obs) for obs in obs_dict.values()], dim=-1)


def get_joint_obs(obs_dict, encoder: str, decentralized: bool = True):
    flat = stack_agent_obs_by_env(obs_dict) if decentralized else get_flat_obs(obs_dict["agent_0"])
    if encoder == "gnn":
        return {"flat": flat, "state_graph": get_state_graph_obs(obs_dict)}
    return flat


def clone_nested(obj):
    if isinstance(obj, dict):
        return {key: clone_nested(value) for key, value in obj.items()}
    if isinstance(obj, th.Tensor):
        return obj.clone()
    return obj.copy() if hasattr(obj, "copy") else obj


def zeros_like_with_leading(obj, leading_shape, device=None):
    if isinstance(obj, dict):
        return {
            key: zeros_like_with_leading(value, leading_shape, device=device)
            for key, value in obj.items()
        }
    shape = tuple(leading_shape) + tuple(obj.shape)
    target_device = device if device is not None else obj.device
    return th.zeros(shape, dtype=obj.dtype, device=target_device)


def set_nested_at_step(target, step, value):
    if isinstance(target, dict):
        for key in target:
            set_nested_at_step(target[key], step, value[key])
    else:
        target[step] = value


def set_nested_env_index(target, env_idx, value):
    if isinstance(target, dict):
        for key in target:
            set_nested_env_index(target[key], env_idx, value[key])
    else:
        target[env_idx] = value


def flatten_rollout_obs(obj):
    if isinstance(obj, dict):
        return {key: flatten_rollout_obs(value) for key, value in obj.items()}
    return obj.reshape((-1,) + tuple(obj.shape[2:]))


def index_nested(obj, indices):
    if isinstance(obj, dict):
        return {key: index_nested(value, indices) for key, value in obj.items()}
    return obj[indices]


def nested_to_numpy(obj):
    if isinstance(obj, dict):
        return {key: nested_to_numpy(value) for key, value in obj.items()}
    if isinstance(obj, th.Tensor):
        return obj.detach().cpu().numpy()
    return np.asarray(obj)


def recursive_index_numpy(obj, idxs):
    if isinstance(obj, dict):
        return {key: recursive_index_numpy(value, idxs) for key, value in obj.items()}
    return obj[idxs]


def add_trailing_dim(obj):
    if isinstance(obj, dict):
        return {key: add_trailing_dim(value) for key, value in obj.items()}
    return obj[..., np.newaxis]


def any_gnn_enabled(args) -> bool:
    encoder_keys = [
        "actor_encoder",
        "critic_encoder",
        "cost_critic_encoder",
        "q_encoder",
        "mixer_encoder",
    ]
    return any(getattr(args, key, "mlp") == "gnn" for key in encoder_keys)

# Creating a list of dicts for each env's actions
def split_action_tensor_dict(action_dict):
    def to_env_action(action):
        if isinstance(action, th.Tensor):
            action = action.detach().cpu().numpy()
        if isinstance(action, np.ndarray) and action.shape == ():
            return action.item()
        return action

    return [
        {agent_id: to_env_action(action) for agent_id, action in zip(action_dict.keys(), per_env_actions)}
        for per_env_actions in zip(*action_dict.values())
    ]

def set_torch(n_threads: int = 0, deterministic: bool = True, cuda: bool = False) -> th.device:
    """Configure PyTorch settings including the number of threads, determinism, and CUDA usage.

    Args:
        n_threads: Number of threads for PyTorch operations.
        deterministic: Whether to use deterministic algorithms.
        cuda: Whether to enable CUDA.

    Returns:
        The PyTorch device configured based on availability and input flags.
    """
    th.set_num_threads(n_threads)
    th.backends.cudnn.deterministic = deterministic
    return th.device("cuda" if th.cuda.is_available() and cuda else "cpu")

def set_random_seed(s: Optional[int] = None) -> None:
    """Set the random seed for reproducibility across different libraries.

    Args:
        s: Seed value to set. If None, the global seed value is used.
    """
    global seed
    if s is not None: seed = s
    rnd.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

def str2bool(s: str) -> bool:
    """Convert a string representation of a boolean to an actual boolean value.

    Args:
        s: String to convert.

    Returns:
        The boolean value corresponding to the input string.

    Raises:
        ArgumentTypeError: If the string does not represent a boolean value.
    """
    if s.lower() == 'true':return True
    elif s.lower() == 'false': return False
    raise ArgumentTypeError('Boolean value expected.')


def Linear(input_dim: int, output_dim: int, act_fn: str = 'leaky_relu', init_weight_uniform: bool = True) -> nn.Linear:
    """Create and initialize a linear layer with appropriate weights.
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/    
    
    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
        act_fn: Activation function.
        init_weight_uniform: Whether to uniformly sample initial weights.

    Returns:
        The initialized layer.
    """
    act_fn = act_fn.lower()
    gain = nn.init.calculate_gain(act_fn)
    layer = nn.Linear(input_dim, output_dim)
    if init_weight_uniform: nn.init.xavier_uniform_(layer.weight, gain=gain)
    else: nn.init.xavier_normal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.00)
    return layer

class Tanh(nn.Module):
    """In-place tanh module."""

    def forward(self, input: th.Tensor) -> th.Tensor:
        """Forward pass for the in-place tanh activation function.

        Args:
            input (th.Tensor): Input tensor.

        Returns:
            Output tensor after applying in-place tanh.
        """
        return th.tanh_(input)

# Useful for picking up the right activation in the networks
th_act_fns = {
    'tanh': Tanh(),
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(),
}
