from typing import List, Tuple
import torch
from torch import Tensor
from torch.optim import Adam
import torch.nn.functional as F


from containers import Config, BipartiteGraphData
from gaton import GATON


class Trainer():
    def __init__(
        self,
        graph_data: BipartiteGraphData,
        config: Config
    ):
        # torch.autograd.set_detect_anomaly(True)
        r'''
        Args:
            graph_data: bipartite graphdata of graph structure
            edge_weight: weight for all edges, shares index with edge_index
                shape: (num_edges,)
            config: configs for model
        '''
        self.graph_data = graph_data
        self.config = config
        self.model = GATON(config)
        print(self.model.modules)
        self.optimizer = Adam(self.model.parameters(),
                              lr=config.lr, weight_decay=config.weight_decay)

        if self.config.objective == 'topic-modeling':
            self.loss_f = self.loss_topic_modeling
        elif self.config.objective == 'classification':
            self.loss_f = self.loss_classification
        else:
            Exception

        self.target_labels = torch.zeros(
            (self.config.num_seq, self.config.num_item), requires_grad=False)
        for seq_index, item_index, edge_weight in zip(self.graph_data.edge_index[0], self.graph_data.edge_index[1], self.graph_data.edge_weight):
            self.target_labels[seq_index][item_index] = edge_weight

    def fit(self) -> List[float]:
        r'''
        Fit to graph data
        Return:
            list of loss for all epochs
        '''
        losses = []
        for epoch in range(1, self.config.epochs + 1):
            loss = self.iter()
            losses.append(loss.item())
            print(f'Epoch: {epoch}, loss: {loss}')
        return losses

    def iter(self) -> Tensor:
        r'''
        Iteration for optimizing weight
        Return:
            loss
        '''
        self.model.train()
        self.optimizer.zero_grad()
        h_item, h_seq = self.model.forward(self.graph_data.x_item, self.graph_data.x_seq,
                                           self.graph_data.edge_index)
        loss = self.loss_f(h_item, h_seq)
        loss.backward()
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def eval(self) -> Tuple[Tensor, Tensor]:
        r'''
        Evaluate model
        Return:
            (h_item, h_seq)
            shape:
                h_item: (num_item, output_dim)
                h_seq: (num_seq, output_dim)
        '''
        self.model.eval()
        h_item, h_seq = self.model.forward(self.graph_data.x_item, self.graph_data.x_seq,
                                           self.graph_data.edge_index)
        return h_item, h_seq

    def loss_topic_modeling(
        self,
        h_item: Tensor,
        h_seq: Tensor
    ):
        r'''
        TODO: make loss function injectable
        TODO: use better matrix calculation
        '''
        loss = 0
        print('-' * 30)
        for seq_idx in range(self.config.num_seq):
            if torch.isnan(h_seq[seq_idx]).any():
                print('is nan')
                exit(1)
            pred = F.softmax(torch.matmul(h_seq[seq_idx], h_item.T), dim=0)
            if self.config.verbose and (seq_idx < 1 or seq_idx >= self.config.num_seq - 1):
                print(
                    f'target: {self.target_labels[seq_idx]}, pred: {pred}, diff: {self.target_labels[seq_idx] - pred}')
            loss += torch.sum(torch.sqrt(
                (self.target_labels[seq_idx] - pred) ** 2 + 1e-7))

        l2_norm = sum(p.pow(2.0).sum()
                      for p in self.model.parameters())
        loss += self.config.l2_lambda * l2_norm
        return loss

    def loss_classification(
        self,
        _: Tensor,
        h_seq: Tensor
    ):
        r'''
        TODO: make loss function injectable
        '''
        loss = 0
        print('*' * 30)
        loss += F.cross_entropy(h_seq, self.graph_data.seq_labels)
        l2_norm = sum(p.pow(2.0).sum()
                      for p in self.model.parameters())
        loss += self.config.l2_lambda * l2_norm
        return loss
