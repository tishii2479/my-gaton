from typing import Any, List
from torch import Tensor

# Type alias for Item
Item = Any


class Config:
    r'''
    Configs for model, training
    '''
    item_embedding_dim: int
    num_item: int
    num_seq: int
    num_head = 4
    num_topic = 5
    num_layer = 2
    d_model = 50
    output_dim = 5
    lr = 0.002
    weight_decay = 5e-4
    l2_lambda = 0.0005
    dropout = 0.6
    epochs = 3
    verbose = False
    task = 'topic-modeling'  # topic-modeling, classification
    dataset = 'toydata'
    model = 'gaton'


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
        x_seq: Tensor,
        seq_labels: Tensor = None
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
            seq_labels: label for sequence, will be set when config.task = 'classification'
        '''
        assert len(edge_index[0]) == len(
            edge_weight), 'edge_index and edge_weight should have same length'
        if seq_labels is not None:
            assert len(x_seq) == len(
                seq_labels), f'seq_labels: {len(seq_labels)} and x_seq: {len(x_seq)} should have same length'

        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.x_item = x_item
        self.x_seq = x_seq
        self.seq_labels = seq_labels
