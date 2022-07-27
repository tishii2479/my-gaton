from trainer import Trainer
import torch
import unittest

from util import group_topics
from containers import Config, BipartiteGraphData


class TestTrainer(unittest.TestCase):
    def test_trainer(self):
        config = Config()
        config.item_embedding_dim = 4
        config.num_item = 4
        config.num_seq = 2

        edge_index = torch.tensor([
            [0, 0, 1, 1],  # sequence index
            [0, 1, 2, 3],  # item index
        ])
        edge_weight = torch.tensor([0.8, 0.2, 0.3, 0.7])

        x_item = torch.randn(config.num_item, config.item_embedding_dim)
        x_seq = torch.randn(config.num_seq, config.item_embedding_dim)

        data = BipartiteGraphData(edge_index, edge_weight, x_item, x_seq)
        trainer = Trainer(data, config)
        trainer.fit()

        _, h_seq = trainer.eval()
        topics = group_topics(h_seq, num_topic=config.output_dim)
        print(topics)


if __name__ == '__main__':
    unittest.main()
