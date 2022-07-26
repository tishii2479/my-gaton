import torch
from torch import Tensor

from containers import BipartiteGraphData, Config
from util import group_topics
from trainer import Trainer


def main():
    # TODO: read from command line
    config = Config()

    edge_index = torch.tensor([
        [0, 0, 1, 1],  # sequence index
        [0, 1, 2, 3],  # item index
    ])
    edge_weight = torch.tensor([0.8, 0.2, 0.3, 0.7])
    documents = [
        ['my', 'name', 'is', 'hello'],
        ['a', 'b', 'c', 'd'],
    ]

    x_item = torch.randn(config.num_item, config.word_embedding_dim)
    x_seq = torch.randn(config.num_seq, config.word_embedding_dim)

    data = BipartiteGraphData(edge_index, edge_weight, x_item, x_seq)
    trainer = Trainer(data, config)
    trainer.fit()
    _, h_seq = trainer.eval()
    topics = group_topics(h_seq, num_topic=config.output_dim)

    print(topics)


if __name__ == '__main__':
    main()
