from random import randint
from containers import Config
from util import group_topics, top_topic_items
from trainer import Trainer
from preprocess import preprocess


def main():
    documents = []
    for i in range(9):
        for j in range(100):
            s = ''
            l = randint(5, 10)
            for _ in range(l):
                s += chr(ord('a') + i * 3 + randint(0, 2))
            print(s)
            documents.append(s)

    graph_data, sequences, items, item_index_dict = preprocess(
        documents)

    # TODO: read from command line, and initialize
    config = Config()
    config.num_item = len(items)
    config.num_seq = len(sequences)
    config.word_embedding_dim = graph_data.x_item.size(1)
    config.epochs = 3000
    config.output_dim = 9

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
