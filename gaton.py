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

        self.W_item = nn.Linear(config.item_embedding_dim,
                                config.d_model, bias=False)
        self.W_seq = nn.Linear(config.num_item, config.d_model, bias=False)

        self.conv1_seq_item = GATConv(
            (config.d_model, config.d_model), config.d_model, config.num_head, dropout=config.dropout)
        self.conv1_item_seq = GATConv(
            (config.d_model, config.d_model), config.d_model, config.num_head, dropout=config.dropout)

        hidden_size = config.num_head * config.d_model
        self.conv2_seq_item = GATConv(
            (hidden_size, hidden_size), config.num_topic, heads=1, dropout=config.dropout)
        self.conv2_item_seq = GATConv(
            (hidden_size, hidden_size), config.num_topic, heads=1, dropout=config.dropout)

        self.output_layer = nn.Linear(config.num_topic, config.output_dim)

    def forward(
        self,
        x_item: Tensor,
        x_seq: Tensor,
        edge_index: Tensor
    ):
        h_item, h_seq = self.calc_probability(x_item, x_seq, edge_index)

        h_item = self.output_layer.forward(h_item)
        h_seq = self.output_layer.forward(h_seq)

        h_item = F.sigmoid(h_item)
        h_seq = F.sigmoid(h_seq)

        return h_item, h_seq

    def calc_probability(
        self,
        x_item: Tensor,
        x_seq: Tensor,
        edge_index: Tensor
    ):
        # Swap rows [0, 1] to [1, 0] (=swap src and dst of `edge_index`)
        swapped_edge_index = torch.index_select(
            edge_index, 0, torch.tensor([1, 0], dtype=torch.long))

        h_item = self.W_item.forward(x_item)
        h_seq = self.W_seq.forward(x_seq)

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
