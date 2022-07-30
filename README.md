# Implementing Graph Attention Topic Modeling Network

https://yangliang.github.io/pdf/www20.pdf

### Installing PyTorch Geometric

```shell
pip3 install --no-binary torch-cluster torch-geometric  torch-scatter torch-spline-conv torch-sparse
```

### Commands

```shell
python3 example.py --epochs=100 --num_topic=10 --d_model=200 --output_dim=30 --dataset=toydata --model=gaton --num_layer=2
```

### Links

#### Citations

- https://www.semanticscholar.org/paper/Graph-Attention-Topic-Modeling-Network-Yang-Wu/7dc880f91dd016ca47cb68325c76143c986d4d20#citing-papers