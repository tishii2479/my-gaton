from typing import List, Tuple
import torch
from torch import Tensor
from torch.optim import Adam


from config import Config
from gaton import GATON
from util import BipartiteGraphData


class Trainer():
    def __init__(
        self,
        data: BipartiteGraphData,
        config: Config
    ):
        r'''
        Args:
            data: bipartite data of graph structure
            edge_weight: weight for all edges, shares index with edge_index
                shape: (num_edges,)
            config: configs for model
        '''
        self.data = data
        self.config = config
        self.model = GATON(config)
        self.optimizer = Adam(self.model.parameters(), lr=config.lr)

    def fit(self) -> List[Tensor]:
        r'''
        Fit to graph data
        Return:
            list of loss for all epochs
        '''
        losses = []
        for epoch in range(1, self.config.epochs + 1):
            loss = self.iter()
            losses.append(loss)
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
        h_item, h_seq = self.model.forward(self.data.x_item, self.data.x_seq,
                                           self.data.edge_index)
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
        h_item, h_seq = self.model.forward(self.data.x_item, self.data.x_seq,
                                           self.data.edge_index)
        return h_item, h_seq

    def loss_f(
        self,
        h_item: Tensor,
        h_seq: Tensor
    ):
        r'''
        TODO: make loss function injectable
        '''
        loss = 0
        for i in range(len(self.data.edge_weight)):
            seq_idx = self.data.edge_index[0][i]
            item_idx = self.data.edge_index[1][i]
            n_ou = self.data.edge_weight[i]
            loss += (n_ou - torch.inner(h_item[item_idx], h_seq[seq_idx])) ** 2

        l2_norm = sum(p.pow(2.0).sum()
                      for p in self.model.parameters())
        loss += self.config.l2_lambda * l2_norm
        return loss
