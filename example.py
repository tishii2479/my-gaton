from pyparsing import ExceptionWordUnicode
from containers import Config
from util import group_topics, top_topic_items
from trainer import Trainer
from preprocess import preprocess

from toydata import create_labeled_toydata, create_toydata


def main():
    # TODO: read from command line
    objective = 'classification'
    num_topic = 5

    if objective == 'topic-modeling':
        (raw_sequences, seq_label), (items,
                                     item_embedding) = create_toydata(num_topic)
    elif objective == 'classification':
        (raw_sequences, seq_label), (items, item_embedding) = create_labeled_toydata(
            num_topic)
    else:
        Exception

    graph_data, sequences, item_index_dict = preprocess(
        (raw_sequences, seq_label), (items, item_embedding))

    # TODO: read from command line, and initialize
    config = Config()
    config.num_item = len(items)
    config.num_seq = len(sequences)
    config.item_embedding_dim = graph_data.x_item.size(1)
    config.d_model = 300
    config.epochs = 1000
    config.output_dim = num_topic
    config.lr = 1
    config.verbose = False
    config.dropout = 0.2
    config.objective = objective

    trainer = Trainer(graph_data, config)
    losses = trainer.fit()
    _, h_seq = trainer.eval()
    topics = group_topics(h_seq, num_topic=config.output_dim)
    top_items = top_topic_items(
        topics, sequences, num_top_item=5, num_item=config.num_item)

    print(topics)
    for topic, top_items_for_topic in enumerate(top_items):
        print(f'Top items for topic {topic}: ' +
              ' '.join([items[index] for index in top_items_for_topic]))
    print(losses[-1])


if __name__ == '__main__':
    main()
