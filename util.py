from torch_geometric.data import Data
import torch


class BipartiteData(Data):
    def __init__(self, edge_index=None, x_item=None, x_seq=None):
        super().__init__()
        self.edge_index = edge_index
        self.x_item = x_item
        self.x_seq = x_seq

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_item.size(0)], [self.x_seq.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)
