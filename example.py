from typing import List, Tuple
import torch
from torch import Tensor
from torch.optim import Adam


from config import Config
from gaton import GATON
from util import BipartiteData


class Trainer():
    def __init__(
        self,
        data: BipartiteData,
        edge_weight: Tensor,
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
        self.edge_weight = edge_weight
        self.config = config
        self.model = GATON(config)
        self.optimizer = Adam(self.model.parameters(), lr=config.lr)

    def train(self) -> List[Tensor]:
        r'''
        Train
        TODO: train -> fit?
        Return:
            list of loss for all epochs
        '''
        losses = []
        for epoch in range(1, self.config.epochs + 1):
            loss = self.train_step()
            losses.append(loss)
            print(f'Epoch: {epoch}, loss: {loss}')
        return losses

    def train_step(self) -> Tensor:
        r'''
        Train step
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
    def test(self) -> Tuple[Tensor, Tensor]:
        r'''
        Return:
            h_item, h_seq
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
        for i in range(len(self.edge_weight)):
            seq_idx = self.data.edge_index[0][i]
            item_idx = self.data.edge_index[1][i]
            n_ou = self.edge_weight[i]
            loss += (n_ou - torch.inner(h_item[item_idx], h_seq[seq_idx])) ** 2

        l2_norm = sum(p.pow(2.0).sum()
                      for p in self.model.parameters())
        loss += self.config.l2_lambda * l2_norm
        return loss


class TopicModeler():
    def __init__(
        self,
        x_item: Tensor,
        x_seq: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        config: Config
    ):
        r'''
        Args:
            x_item: embedding of items (word, product)
                shape: (U, item_embedding_size)
            x_seq: embedding of sequence (document, purchase history)
                shape: (num_seq, U)
            edge_index: edge information in COO format, [[seq_index], [item_index]]
                info: value describe the index in the bipartite group
                shape: (2, num_edges)
            edge_weight: weight for all edges, shares index with edge_index
                shape: (num_edges,)
            config: configs for model
        '''
        self.data = BipartiteData(edge_index, x_item, x_seq)
        self.config = config
        self.trainer = Trainer(
            self.data,
            edge_weight,
            config
        )

    def fit(self) -> List[Tensor]:
        losses = self.trainer.train()
        return losses

    def top_topics(self):
        _, h_seq = self.trainer.test()
        num_topic = self.config.output_dim
        topics = [[] for _ in range(num_topic)]

        for i, h in enumerate(h_seq):
            topic = torch.argmax(h).item()
            topics[topic].append(i)
            print(i, h, topic)

        return topics


if __name__ == '__main__':
    config = Config()

    edge_index = torch.tensor([
        [0, 0, 1, 1],  # sequence index
        [0, 1, 2, 3],  # item index
    ])
    edge_weight = torch.tensor([0.8, 0.2, 0.3, 0.7])

    x_item = torch.randn(config.num_item, config.word_embedding_dim)
    x_seq = torch.randn(config.num_seq, config.word_embedding_dim)

    modeler = TopicModeler(x_item, x_seq, edge_index, edge_weight, config)
    modeler.fit()
    top_topics = modeler.top_topics()

    print(top_topics)
