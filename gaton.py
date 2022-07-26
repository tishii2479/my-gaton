import torch
from torch import nn, Tensor
from torch_geometric.nn import GATConv
import torch.nn.functional as F


from config import Config


class GATON(nn.Module):
    def __init__(
        self,
        config: Config
    ):
        super().__init__()

        self.W_item = nn.Linear(config.word_embedding_dim, config.d)
        self.W_sequment = nn.Linear(config.num_item, config.d)

        self.conv1_seq_item = GATConv(
            (config.d, config.d), config.d, config.num_head)
        self.conv1_item_seq = GATConv(
            (config.d, config.d), config.d, config.num_head)

        self.conv2_seq_item = GATConv(
            (config.num_head * config.d, config.num_head * config.d), config.output_dim, heads=1)
        self.conv2_item_seq = GATConv(
            (config.num_head * config.d, config.num_head * config.d), config.output_dim, heads=1)

    def forward(
        self,
        x_item: Tensor,
        x_seq: Tensor,
        edge_index: Tensor
    ):
        swapped_edge_index = torch.index_select(
            edge_index, 0, torch.tensor([1, 0], dtype=torch.long))

        h_item = self.W_item.forward(x_item)
        h_seq = self.W_item.forward(x_seq)

        h_item2 = self.conv1_seq_item.forward((h_seq, h_item), edge_index)
        h_seq2 = self.conv1_item_seq.forward(
            (h_item, h_seq), swapped_edge_index)

        h_item2 = F.elu(h_item2)
        h_seq2 = F.elu(h_seq2)

        h_item3 = self.conv2_seq_item.forward((h_seq2, h_item2), edge_index)
        h_seq3 = self.conv2_item_seq.forward(
            (h_item2, h_seq2), swapped_edge_index)

        h_item3 = F.softmax(h_item3, dim=1)
        h_seq3 = F.softmax(h_seq3, dim=1)

        return h_item3, h_seq3