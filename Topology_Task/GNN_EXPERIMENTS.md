# GNN Encoder Experiments

All GNN encoders are opt-in. The default remains the original flat MLP setup.

The GNN path requires PyTorch Geometric:

```bash
python -m pip install torch-geometric
```

## MAPPO

Baseline:

```bash
python main.py --alg MAPPO --env-id bus14 --actor-encoder mlp --critic-encoder mlp --seed 0
```

Actor only:

```bash
python main.py --alg MAPPO --env-id bus14 --actor-encoder gnn --critic-encoder mlp --seed 0
```

Critic only:

```bash
python main.py --alg MAPPO --env-id bus14 --actor-encoder mlp --critic-encoder gnn --seed 0
```

Actor and critic:

```bash
python main.py --alg MAPPO --env-id bus14 --actor-encoder gnn --critic-encoder gnn --seed 0
```

The GNN graph is busbar-only. Passing `--gnn-graph-type bus` is optional because it is the default:

```bash
python main.py --alg MAPPO --env-id bus14 --actor-encoder gnn --critic-encoder gnn --gnn-graph-type bus --seed 0
```

## LagrMAPPO

Baseline:

```bash
python main.py --alg LAGRMAPPO --constraints-type 1 --env-id bus14 --actor-encoder mlp --critic-encoder mlp --cost-critic-encoder mlp --seed 0
```

Cost critic only:

```bash
python main.py --alg LAGRMAPPO --constraints-type 1 --env-id bus14 --actor-encoder mlp --critic-encoder mlp --cost-critic-encoder gnn --seed 0
```

All value estimators:

```bash
python main.py --alg LAGRMAPPO --constraints-type 1 --env-id bus14 --actor-encoder mlp --critic-encoder gnn --cost-critic-encoder gnn --seed 0
```

Actor plus value estimators:

```bash
python main.py --alg LAGRMAPPO --constraints-type 1 --env-id bus14 --actor-encoder gnn --critic-encoder gnn --cost-critic-encoder gnn --seed 0
```

## QPLEX

Baseline:

```bash
python main.py --alg QPLEX --env-id bus14 --q-encoder mlp --mixer-encoder mlp --seed 0
```

Agent Q-networks only:

```bash
python main.py --alg QPLEX --env-id bus14 --q-encoder gnn --mixer-encoder mlp --seed 0
```

Mixer state only:

```bash
python main.py --alg QPLEX --env-id bus14 --q-encoder mlp --mixer-encoder gnn --seed 0
```

Q-networks and mixer:

```bash
python main.py --alg QPLEX --env-id bus14 --q-encoder gnn --mixer-encoder gnn --seed 0
```

## Shared GNN Knobs

```bash
--gnn-hidden-dim 128
--gnn-out-dim 128
--gnn-layers 2
--gnn-type gat
--gnn-heads 1
--gnn-readout-aggr mean
--graphsage-aggr mean
--gnn-layer-norm true
--gnn-concat-flat false
--gnn-graph-type bus
--gnn-include-neighbors true
```

`--gnn-type` selects the PyTorch Geometric layer: `gcn`, `gat`, `gine`, or `graphsage`.

`--gnn-readout-aggr` selects how node embeddings are pooled into one graph embedding: `mean`, `sum`, or `max`.

`--graphsage-aggr` only applies when `--gnn-type graphsage`; it selects GraphSAGE neighborhood aggregation: `mean` or `sum`.

`--gnn-graph-type` is kept as a compatibility flag, but only `bus` is supported. The graph uses one node per busbar and activates the current bus-to-bus line connections from `topo_vect`.

`--gnn-concat-flat true` feeds the original flat observation alongside the graph embedding. This is useful for checking whether the GNN helps by itself or mainly as an auxiliary encoder.

`--gnn-include-neighbors true` expands each local agent busbar graph with one-hop neighboring substations. Set it to `false` for stricter local-only graphs.
