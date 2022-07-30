from typing import List, Tuple
import torch
from torch import Tensor
from torch.optim import Adam
import torch.nn.functional as F


from containers import Config, BipartiteGraphData
from gaton import GATON
from util import create_target_labels


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

        if self.config.task == 'topic-modeling':
            self.loss_f = self.loss_topic_modeling
        elif self.config.task == 'classification':
            self.loss_f = self.loss_classification
        else:
            Exception

        self.target_labels = create_target_labels(
            self.graph_data.edge_index,
            self.graph_data.edge_weight,
            self.config.num_seq,
            self.config.num_item
        )

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
        Calc loss of each prediction
        '''
        loss = 0
        print('-' * 30)
        pred = F.softmax(torch.matmul(h_seq, h_item.T), dim=1)
        mat_size = len(h_seq) * len(h_item)
        for i in range(10):
            print(self.target_labels[0][i], pred[0][i])
        loss += torch.sum(torch.sqrt(
            (self.target_labels - pred) ** 2 + 1e-7)) / mat_size
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
        pred = F.softmax(h_seq, dim=1)
        loss += F.cross_entropy(pred, self.graph_data.seq_labels)
        l2_norm = sum(p.pow(2.0).sum()
                      for p in self.model.parameters())
        loss += self.config.l2_lambda * l2_norm
        return loss
