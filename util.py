import torch
from torch import Tensor


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


def group_topics(
    h_seq: Tensor,
    num_topic: int
):
    topics = [[] for _ in range(num_topic)]

    for i, h in enumerate(h_seq):
        topic = torch.argmax(h).item()
        topics[topic].append(i)
        print(f'index: {i}, probability: {h}, to: {topic}')

    return topics
