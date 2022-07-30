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
        self.best_model_path = 'weights/gaton.pt'
        print(self.model.modules)

        self.optimizer = Adam(self.model.parameters(),
                              lr=config.lr, weight_decay=config.weight_decay)

        if self.config.task == 'topic-modeling':
            self.loss_f = self.loss_topic_modeling
        elif self.config.task == 'classification':
            self.loss_f = self.loss_classification
        else:
            assert False, f'config.task = {self.config.task} is invalid.'

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
        best_loss = 1e10

        for epoch in range(1, self.config.epochs + 1):
            loss = self.iter().item()
            losses.append(loss)
            print(f'Epoch: {epoch}, loss: {loss}')
            if loss < best_loss:
                best_loss = loss
                torch.save(self.model.state_dict(), self.best_model_path)

        # Use best model
        losses.append(best_loss)
        self.model.load_state_dict(torch.load(self.best_model_path))
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
        self.model.load_state_dict(torch.load(self.best_model_path))
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
        loss += torch.sqrt(F.mse_loss(pred, self.target_labels))
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
