import argparse

from containers import Config
from util import top_cluster_items, visualize_cluster, group_topics, visualize_loss
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

    print(f'dataset: {dataset}, task: {task} is invalid')
    Exception


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
    return config


def main():
    config = read_config()
    (raw_sequences, seq_labels), (items,
                                  item_embedding) = get_data(config.dataset, config.task, config.num_topic)

    graph_data, sequences, _ = preprocess(
        (raw_sequences, seq_labels), (items, item_embedding))

    config.num_item = len(items)
    config.num_seq = len(sequences)
    config.item_embedding_dim = graph_data.x_item.size(1)

    trainer = Trainer(graph_data, config)
    losses = trainer.fit()
    _, h_seq = trainer.eval()

    cluster_labels = group_topics(h_seq, config.num_topic)
    print(cluster_labels)

    seq_cnt = [0] * config.num_seq
    for e in cluster_labels:
        seq_cnt[e] += 1

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
