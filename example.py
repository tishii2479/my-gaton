import argparse

from containers import Config
from util import group_topics, top_topic_items
from trainer import Trainer
from preprocess import preprocess

from toydata import create_labeled_toydata, create_toydata


def get_data(objective: str, num_topic: int):
    if objective == 'topic-modeling':
        (raw_sequences, seq_labels), (items,
                                      item_embedding) = create_toydata(num_topic)
    elif objective == 'classification':
        (raw_sequences, seq_labels), (items, item_embedding) = create_labeled_toydata(
            num_topic)
    else:
        Exception
    return (raw_sequences, seq_labels), (items, item_embedding)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=50)
    parser.add_argument('--num_topic', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--l2_lambda', type=float, default=0)
    parser.add_argument('--objective', type=str, default='topic-modeling')

    return parser.parse_args()


def read_config() -> Config:
    args = read_args()
    config = Config()
    config.d_model = args.d_model
    config.epochs = args.epochs
    config.output_dim = args.num_topic
    config.num_topic = args.num_topic
    config.num_head = args.heads
    config.lr = args.lr
    config.verbose = args.verbose
    config.dropout = args.dropout
    config.l2_lambda = args.l2_lambda
    config.objective = args.objective
    return config


def main():
    config = read_config()
    (raw_sequences, seq_labels), (items,
                                  item_embedding) = get_data(config.objective, config.num_topic)

    graph_data, sequences, _ = preprocess(
        (raw_sequences, seq_labels), (items, item_embedding))

    config.num_item = len(items)
    config.num_seq = len(sequences)
    config.item_embedding_dim = graph_data.x_item.size(1)

    trainer = Trainer(graph_data, config)
    losses = trainer.fit()
    h_item, h_seq = trainer.eval()
    topics = group_topics(h_seq, num_topic=config.num_topic)
    top_items = top_topic_items(
        topics, sequences, num_top_item=5, num_item=config.num_item)

    print('item embedding')
    _ = group_topics(h_item, num_topic=config.num_topic)

    print(topics)
    for topic, top_items_for_topic in enumerate(top_items):
        print(f'Top items for topic {topic}: ' +
              ' '.join([items[index] for index in top_items_for_topic]))
    print(losses[-1])


if __name__ == '__main__':
    main()
