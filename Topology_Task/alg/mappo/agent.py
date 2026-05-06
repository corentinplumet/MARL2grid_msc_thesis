from torch.distributions import Categorical, Normal

from common.imports import *
from common.gnn import GraphAndFlatEncoder
from common.utils import get_flat_obs
from common.utils import Linear, th_act_fns

class Actor(nn.Module):
    """Neural network-based agent for policy gradient methods, supporting both discrete and continuous action spaces.

    Attributes:
        critic (nn.Sequential): Critic network for value estimation.
        actor (nn.Sequential): Actor network for action selection.
        logstd (nn.Parameter): Log standard deviation for continuous action spaces.
    """

    def __init__(self, id: int, envs: gym.Env, args: Dict[str, Any], continuous_actions: bool):
        """
        Initialize the Agent with specified environment, arguments, and action type.

        Args:
            envs: The environment.
            args: Arguments for configuration.
            continuous_actions: Flag indicating continuous or discrete action space.
        """
        super().__init__()
  
        self.encoder_type = args.actor_encoder
        flat_dim = np.prod(envs.observation_space[f"agent_{id}"].shape)

        if self.encoder_type == "gnn":
            if envs.graph_specs is None:
                raise ValueError("actor_encoder=gnn requires graph observations from the environment.")
            self.encoder = GraphAndFlatEncoder(
                envs.graph_specs[f"agent_{id}"],
                flat_dim=flat_dim,
                args=args,
                use_flat=args.gnn_concat_flat,
            )
            input_dim = self.encoder.out_dim
        else:
            self.encoder = nn.Identity()
            input_dim = flat_dim

        # Actor network setup
        actor_layers = args.actor_layers
        act_str, act_fn = args.actor_act_fn, th_act_fns[args.actor_act_fn]
        layers = []
        layers.extend([
            Linear(input_dim, actor_layers[0], act_str),
            act_fn
        ])
        for idx, embed_dim in enumerate(actor_layers[1:], start=1): 
            layers.extend([Linear(actor_layers[idx-1], embed_dim, act_str), act_fn])
        
        # Final layer differs for continuous vs. discrete actions
        if continuous_actions: raise("Redispatching actions are not yet implemented")
        else:
            layers.append(Linear(actor_layers[-1], np.prod(envs.action_space[f"agent_{id}"].n)))
            self.get_action = self.get_discrete_action
            self.get_eval_action = self.get_eval_discrete_action
        self.actor = nn.Sequential(*layers)

    def _encode(self, x):
        if self.encoder_type == "gnn":
            return self.encoder(x, graph_key="graph")
        return get_flat_obs(x)

    def get_discrete_action(self, x: th.Tensor, action: th.Tensor = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Sample discrete actions and compute log probabilities and entropy.

        Args:
            x: Input observations.
            action: Specific action to take. Defaults to None.

        Returns:
            A tuple containing tensors for the sampled discrete actions, the log probability of the sampled actions, and the entropy of the action distribution.
        """
        logits = self.actor(self._encode(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy()
    
    def get_eval_discrete_action(self, x: th.Tensor) -> th.Tensor:
        """Evaluate discrete actions without exploration.

        Args:
            x: Input observations.

        Returns:
            A tensor with deterministic discrete actions for evaluation.
        """
        return self.get_discrete_action(x)[0]

    def get_continuous_action(self, x: th.Tensor, action: th.Tensor = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        raise("Redispatching actions are not yet implemented")
       
    def get_eval_continuous_action(self, x: th.Tensor) -> th.Tensor:
        raise("Redispatching actions are not yet implemented")

class Critic(nn.Module):
    """Neural network-based agent for policy gradient methods, supporting both discrete and continuous action spaces.

    Attributes:
        critic (nn.Sequential): Critic network for value estimation.
        actor (nn.Sequential): Actor network for action selection.
        logstd (nn.Parameter): Log standard deviation for continuous action spaces.
    """

    def __init__(self, envs: gym.Env, args: Dict[str, Any]):
        """
        Initialize the Critic with specified environment, and arguments.

        Args:
            envs: The environment.
            args: Arguments for configuration.
        """
        super().__init__()

        self.encoder_type = args.critic_encoder
        flat_dim = sum(space.shape[0] for space in envs.observation_space.values()) if args.decentralized else envs.observation_space['agent_0'].shape[-1]

        if self.encoder_type == "gnn":
            if envs.graph_specs is None:
                raise ValueError("critic_encoder=gnn requires graph observations from the environment.")
            self.encoder = GraphAndFlatEncoder(
                envs.graph_specs["state"],
                flat_dim=flat_dim,
                args=args,
                use_flat=args.gnn_concat_flat,
            )
            input_dim = self.encoder.out_dim
        else:
            self.encoder = nn.Identity()
            input_dim = flat_dim

        # Critic network setup
        critic_layers = args.critic_layers
        act_str, act_fn = args.critic_act_fn, th_act_fns[args.critic_act_fn]
        layers = []
        layers.extend([
            Linear(np.prod((input_dim, )), critic_layers[0], act_str),
            act_fn
        ])

        for idx, embed_dim in enumerate(critic_layers[1:], start=1): 
            layers.extend([Linear(critic_layers[idx-1], embed_dim, act_str), act_fn])
        layers.append(Linear(critic_layers[-1], 1, 'linear'))
        self.critic = nn.Sequential(*layers)

    def _encode(self, x):
        if self.encoder_type == "gnn":
            return self.encoder(x, graph_key="state_graph")
        return get_flat_obs(x)

    def get_value(self, x: th.Tensor) -> th.Tensor:
        """Compute value estimate (critic output) for given observations.

        Args:
            x: Input observations.

        Returns:
            A tensor containing value estimates.
        """
        return self.critic(self._encode(x))
