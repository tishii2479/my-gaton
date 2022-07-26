from typing import Any, List
from torch import Tensor

# Type alias for Item
Item = Any


class Config:
    r'''
    Configs for model, training
    '''
    word_embedding_dim: int
    num_item: int
    num_seq: int
    num_head = 4
    d_model = 50
    output_dim = 3
    lr = 0.002
    l2_lambda = 0.0005
    dropout = 0.6
    epochs = 3


class Sequence:
    def __init__(
        self,
        sequence,
        indicies: List[int]
    ):
        self.sequence = sequence
        self.indicies = indicies


class BipartiteGraphData:
    r'''
    Container for graph data
    '''

    def __init__(
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        x_item: Tensor,
        x_seq: Tensor
    ):
        r'''
        Args:
            x_item: embedding of items (word, product)
                shape: (U, item_embedding_size)
            x_seq: embedding of sequence (document, purchase history)
                shape: (num_seq, U)
            edge_index: edge information in COO format, [[seq_index], [item_index]]
                info: value describe the index in the bipartite group
                shape: (2, num_edges)
            edge_weight: weight for all edges, shares index with edge_index
                shape: (num_edges,)
        '''
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.x_item = x_item
        self.x_seq = x_seq
