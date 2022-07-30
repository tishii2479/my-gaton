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
            (config.d_model, config.d_model), config.d_model, config.num_head, dropout=config.dropout, concat=False)
        self.conv1_item_seq = GATConv(
            (config.d_model, config.d_model), config.d_model, config.num_head, dropout=config.dropout, concat=False)

        hidden_size = config.d_model
        self.batch_norm1 = nn.BatchNorm1d(hidden_size, affine=False)

        self.conv2_seq_item = GATConv(
            (hidden_size, hidden_size), config.output_dim, heads=1, dropout=config.dropout, concat=False)
        self.conv2_item_seq = GATConv(
            (hidden_size, hidden_size), config.output_dim, heads=1, dropout=config.dropout, concat=False)

        self.batch_norm2 = nn.BatchNorm1d(config.output_dim, affine=False)

    def forward(
        self,
        x_item: Tensor,
        x_seq: Tensor,
        edge_index: Tensor
    ):
        # Swap rows [0, 1] to [1, 0] (=swap src and dst of `edge_index`)
        swapped_edge_index = torch.index_select(
            edge_index, 0, torch.tensor([1, 0], dtype=torch.long))
        # print(x_seq[0])

        # TODO: maybe this part is causing bug when clustering
        h_item = self.W_item.forward(x_item)
        h_seq = self.W_seq.forward(x_seq)

        if torch.isnan(h_seq[0]).any():
            print(self.conv2_seq_item.state_dict())

        h_item2 = self.conv1_seq_item.forward((h_seq, h_item), edge_index)
        h_seq2 = self.conv1_item_seq.forward(
            (h_item, h_seq), swapped_edge_index)

        h_item2 = F.relu(h_item2)
        h_seq2 = F.relu(h_seq2)

        h_item2 = self.batch_norm1.forward(h_item2)
        h_seq2 = self.batch_norm1.forward(h_seq2)

        h_item3 = self.conv2_seq_item.forward((h_seq2, h_item2), edge_index)
        h_seq3 = self.conv2_item_seq.forward(
            (h_item2, h_seq2), swapped_edge_index)

        h_item3 = F.relu(h_item3)
        h_seq3 = F.relu(h_seq3)

        h_item3 = self.batch_norm2.forward(h_item3)
        h_seq3 = self.batch_norm2.forward(h_seq3)

        # h_item3 = F.softmax(h_item3, dim=1)
        # h_seq3 = F.softmax(h_seq3, dim=1)

        return h_item3, h_seq3
