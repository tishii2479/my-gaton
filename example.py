import argparse
from collections import Counter
from typing import List, Tuple

from sklearn.decomposition import NMF
from torch import Tensor
import torch

from containers import BipartiteGraphData, Config
from util import create_target_labels, top_cluster_items, visualize_cluster, group_topics, visualize_loss
from trainer import Trainer
from preprocess import preprocess

from toydata import create_labeled_toydata, create_toydata
from realdata import create_movielens_data


def get_data(
    dataset: str,
    task: str,
    num_topic: int
):
    r'''
    Return:
        (raw_sequences, seq_labels), (items, item_embedding)
        info: seq_labels is None when solving topic-modeling dataset
    '''
    if dataset == 'movielens':
        return create_movielens_data()
    elif dataset == 'toydata':
        if task == 'topic-modeling':
            return create_toydata(num_topic)
        elif task == 'classification':
            return create_labeled_toydata(num_topic)
    else:
        assert False, f'dataset: {dataset}, task: {task} is invalid'


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=50)
    parser.add_argument('--num_topic', type=int, default=10)
    parser.add_argument('--output_dim', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--l2_lambda', type=float, default=0)
    parser.add_argument('--task', type=str, default='topic-modeling')
    parser.add_argument('--dataset', type=str, default='toydata')
    parser.add_argument('--model', type=str, default='gaton')

    return parser.parse_args()


def read_config() -> Config:
    args = read_args()
    config = Config()
    config.d_model = args.d_model
    config.epochs = args.epochs
    config.output_dim = args.output_dim
    config.num_topic = args.num_topic
    config.num_head = args.heads
    config.lr = args.lr
    config.verbose = args.verbose
    config.dropout = args.dropout
    config.l2_lambda = args.l2_lambda
    config.task = args.task
    config.dataset = args.dataset
    config.model = args.model
    return config


def train_gaton(
    graph_data: BipartiteGraphData,
    config: Config
):
    r'''
    Train GATON
    Return:
        (h_item, h_seq, losses)
    '''
    trainer = Trainer(graph_data, config)
    losses = trainer.fit()
    h_item, h_seq = trainer.eval()
    return h_item, h_seq, losses


def train_nmf(
    graph_data: BipartiteGraphData,
    config: Config
):
    r'''
    Train NMF
    Return:
        (h_item, h_seq, losses)
    '''
    target_labels = create_target_labels(
        graph_data.edge_index,
        graph_data.edge_weight,
        config.num_seq,
        config.num_item
    )
    model = NMF(n_components=config.output_dim)
    h_seq = model.fit_transform(target_labels)
    h_item = model.components_

    mat_size = (len(h_seq) * len(h_item))
    mean_reconstruction_err = model.reconstruction_err_ / mat_size

    print(mean_reconstruction_err)
    return h_item, h_seq, [mean_reconstruction_err]


def train(
    graph_data: BipartiteGraphData,
    config: Config
) -> Tuple[Tensor, Tensor, List[float]]:
    r'''
    Train config.model
    Return:
        (h_item, h_seq, losses)
    '''
    if config.model == 'gaton':
        return train_gaton(graph_data, config)
    elif config.model == 'nmf':
        return train_nmf(graph_data, config)
    else:
        assert False, f'config.model = {config.model} is invalid.'


def main():
    # torch.autograd.set_detect_anomaly(True)
    config = read_config()
    (raw_sequences, seq_labels), (items,
                                  item_embedding) = get_data(config.dataset, config.task, config.num_topic)

    graph_data, sequences, _ = preprocess(
        (raw_sequences, seq_labels), (items, item_embedding))

    config.num_item = len(items)
    config.num_seq = len(sequences)
    config.item_embedding_dim = graph_data.x_item.size(1)

    print(f'model: {config.model}')

    _, h_seq, losses = train(graph_data, config)

    cluster_labels = group_topics(h_seq, config.num_topic)
    print(cluster_labels)

    seq_cnt = Counter(cluster_labels)

    top_items = top_cluster_items(
        config.num_topic, cluster_labels, sequences, num_top_item=10, num_item=config.num_item)

    for cluster, (top_items, ratios) in enumerate(top_items):
        print(f'Top items for cluster {cluster} (size {seq_cnt[cluster]}): \n' +
              '\n'.join([str(items[top_items[index]]) + ' ' + str(ratios[index]) for index in range(10)]))
        print()
    print(losses[-1])

    visualize_loss(losses)
    visualize_cluster(h_seq, config.num_topic, cluster_labels)


if __name__ == '__main__':
    main()
