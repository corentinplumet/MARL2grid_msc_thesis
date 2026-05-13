from common.imports import *
from common.utils import str2bool

def get_alg_args() -> Namespace:
    """Parse command-line arguments for PPO.

    This function sets up and parses arguments for configuring the training and evaluation of a PPO agent.

    Returns:
        A namespace containing the parsed arguments.
    """
    parser = ap.ArgumentParser()

    parser.add_argument("--total-timesteps", type=int, default=25000000, help="Total timesteps for the experiment")
    parser.add_argument("--n-steps", type=int, default=20000, help="Steps per policy rollout")    # 20k for 1 env
    parser.add_argument("--eval-freq", type=int, default=100, help="Total timesteps between deterministic evals")

    parser.add_argument('--actor-layers', nargs='+', type=int, default=[256, 256], help='Actor network size')
    parser.add_argument('--critic-layers', nargs='+', type=int, default=[256, 256], help='Critic network size')
    parser.add_argument('--actor-act-fn', type=str, default='relu', help='Actor activation function')
    parser.add_argument('--critic-act-fn', type=str, default='relu', help='Critic activation function')
    parser.add_argument("--actor-encoder", type=str, default="mlp", choices=["mlp", "gnn"], help="Observation encoder for actor policies")
    parser.add_argument("--critic-encoder", type=str, default="mlp", choices=["mlp", "gnn"], help="Observation encoder for the centralized critic")
    parser.add_argument("--gnn-type", type=str, default="gat", choices=["gcn", "gat", "gine", "graphsage"], help="PyTorch Geometric convolution used by GNN encoders")
    parser.add_argument("--gnn-hidden-dim", type=int, default=128, help="Hidden dimension for GNN encoders")
    parser.add_argument("--gnn-out-dim", type=int, default=128, help="Output embedding dimension for GNN encoders")
    parser.add_argument("--gnn-layers", type=int, default=2, help="Number of message-passing layers")
    parser.add_argument("--gnn-heads", type=int, default=1, help="Number of attention heads for GAT encoders")
    parser.add_argument("--gnn-readout-aggr", type=str, default="mean", choices=["mean", "sum", "max"], help="Graph-level node pooling used to produce one graph embedding")
    parser.add_argument("--graphsage-aggr", type=str, default="mean", choices=["mean", "sum"], help="GraphSAGE neighborhood aggregation")
    parser.add_argument("--gnn-aggr", dest="graphsage_aggr", type=str, choices=["mean", "sum"], help=ap.SUPPRESS)
    parser.add_argument("--gnn-layer-norm", type=str2bool, default=True, help="Use layer normalization inside GNN layers")
    parser.add_argument("--gnn-concat-flat", type=str2bool, default=False, help="Concatenate flat observations to GNN embeddings before the head")
    parser.add_argument("--gnn-graph-type", type=str, default="bus", choices=["bus"], help="Use busbars as graph nodes")
    parser.add_argument("--gnn-include-neighbors", type=str2bool, default=True, help="Include one-hop neighboring substations in local agent busbar graphs")
    parser.add_argument("--actor-lr", type=float, default=3e-4, help="Learning rate for the actor")
    parser.add_argument("--critic-lr", type=float, default=3e-4, help="Learning rate for the critic")
    parser.add_argument("--anneal-lr", type=str2bool, default=True, help="Toggles learning rate annealing")

    parser.add_argument("--gamma", type=float, default=.9, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=.95, help="Lambda for the genralized advantage estimation")

    parser.add_argument("--update-epochs", type=int, default=5, help="Number of update epochs")
    parser.add_argument("--n-minibatches", type=int, default=4, help="Number of minibatches")
    parser.add_argument("--max-grad-norm", type=float, default=10, help="Maximum norm for gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None, help="Target KL divergence threshold")

    parser.add_argument("--norm-adv", type=str2bool, default=True, help="Toggles advantage normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="Surrogate clip coefficient")
    parser.add_argument("--clip-vfloss", type=str2bool, default=True, help="Toggles clip for value function loss")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")

    return parser.parse_known_args()[0]
