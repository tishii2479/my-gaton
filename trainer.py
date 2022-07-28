from typing import List, Tuple
import torch
from torch import Tensor
from torch.optim import SGD
import torch.nn.functional as F


from containers import Config, BipartiteGraphData
from gaton import GATON


class Trainer():
    def __init__(
        self,
        graph_data: BipartiteGraphData,
        config: Config
    ):
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
        self.optimizer = SGD(self.model.parameters(),
                             lr=config.lr, weight_decay=config.weight_decay)

        if self.config.objective == 'topic-modeling':
            self.loss_f = self.loss_topic_modeling
        elif self.config.objective == 'classification':
            self.loss_f = self.loss_classification
        else:
            Exception

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
        for i in range(len(self.graph_data.edge_weight)):
            seq_idx = self.graph_data.edge_index[0][i]
            item_idx = self.graph_data.edge_index[1][i]
            n_ou = self.graph_data.edge_weight[i]
            if self.config.verbose and i % 300 == 0:
                print(
                    f'h_seq: {h_seq[seq_idx]}, n_ou: {n_ou.item()}, pred: {torch.inner(h_item[item_idx], h_seq[seq_idx]).item()}, loss: {((n_ou - torch.inner(h_item[item_idx], h_seq[seq_idx])) ** 2).item()}')
            loss += (n_ou - torch.inner(h_item[item_idx], h_seq[seq_idx])) ** 2

        l2_norm = sum(p.pow(2.0).sum()
                      for p in self.model.parameters())
        loss += self.config.l2_lambda * l2_norm
        return loss

    def loss_classification(
        self,
        h_item: Tensor,
        h_seq: Tensor
    ):
        r'''
        TODO: make loss function injectable
        '''
        loss = 0
        print('-' * 30)
        loss += F.cross_entropy(h_seq, self.graph_data.seq_labels)
        l2_norm = sum(p.pow(2.0).sum()
                      for p in self.model.parameters())
        loss += self.config.l2_lambda * l2_norm
        return loss
