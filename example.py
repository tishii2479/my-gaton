from containers import Config
from util import group_topics, top_topic_items
from trainer import Trainer
from preprocess import preprocess

from toydata import create_labeled_toydata, create_toydata


def get_data(objective: str, num_topic: int):
    if objective == 'topic-modeling':
        (raw_sequences, seq_label), (items,
                                     item_embedding) = create_toydata(num_topic)
    elif objective == 'classification':
        (raw_sequences, seq_label), (items, item_embedding) = create_labeled_toydata(
            num_topic)
    else:
        Exception
    return (raw_sequences, seq_label), (items, item_embedding)


def main():
    # TODO: read from command line
    objective = 'topic-modeling'
    num_topic = 5

    (raw_sequences, seq_label), (items,
                                 item_embedding) = get_data(objective, num_topic)

    graph_data, sequences, _ = preprocess(
        (raw_sequences, seq_label), (items, item_embedding))

    # TODO: read from command line, and initialize
    config = Config()
    config.num_item = len(items)
    config.num_seq = len(sequences)
    config.item_embedding_dim = graph_data.x_item.size(1)
    config.d_model = 300
    config.epochs = 2000
    config.output_dim = 50
    config.num_topic = num_topic
    config.lr = 0.002
    config.verbose = False
    config.dropout = 0.2
    config.l2_lambda = 0
    config.objective = objective

    trainer = Trainer(graph_data, config)
    losses = trainer.fit()
    _, h_seq = trainer.eval()
    topics = group_topics(h_seq, num_topic=config.num_topic)
    top_items = top_topic_items(
        topics, sequences, num_top_item=5, num_item=config.num_item)

    print(topics)
    for topic, top_items_for_topic in enumerate(top_items):
        print(f'Top items for topic {topic}: ' +
              ' '.join([items[index] for index in top_items_for_topic]))
    print(losses[-1])


if __name__ == '__main__':
    main()
