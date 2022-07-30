import torch
from torch import nn, Tensor
from torch_geometric.nn import GATConv
import torch.nn.functional as F


from containers import Config


class GATON(nn.Module):
    def __init__(
        self,
        config: Config
    ):
        super().__init__()
        self.config = config
        self.W_item = nn.Linear(config.item_embedding_dim,
                                config.d_model, bias=False)
        self.W_seq = nn.Linear(config.num_item, config.d_model, bias=False)

        self.convs = nn.ModuleList([GATConv(config.d_model, config.d_model, config.num_head,
                                            dropout=config.dropout, concat=False) for _ in range(config.num_layer)])
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(config.d_model, affine=False) for _ in range(config.num_layer)])

    def forward(
        self,
        x_item: Tensor,
        x_seq: Tensor,
        edge_index: Tensor
    ):
        h_item = self.W_item.forward(x_item)
        h_seq = self.W_seq.forward(x_seq)

        offset = len(h_item)

        H = torch.concat([h_item, h_seq])

        for i in range(self.config.num_layer):
            H = self.convs[i].forward(H, edge_index)
            H = F.relu(H)
            H = self.batch_norms[i].forward(H)

        h_item, h_seq = H[:offset], H[offset:]

        return h_item, h_seq
