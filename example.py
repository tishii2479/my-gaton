from random import randint, choice
import torch
from containers import Config
from util import group_topics, top_topic_items
from trainer import Trainer
from preprocess import preprocess


def create_toydata(num_topic: int):
    documents = []
    words = []
    key_words = [[] for _ in range(num_topic)]

    for _ in range(1, 51):
        s = ''
        for _ in range(10):
            s += chr(ord('a') + randint(0, 26))
        words.append(s)

    for i in range(num_topic):
        for j in range(1, 6):
            s = chr(ord('a') + i) * j
            key_words[i].append(s)
            words.append(s)

    for i in range(num_topic):
        for _ in range(20):
            doc = []
            for _ in range(randint(50, 100)):
                doc.append(choice(key_words[i]))
            # for _ in range(randint(15, 20)):
            #     doc.append(choice(words))
            documents.append(doc)
    word_embedding = torch.eye(len(words))

    return documents, (words, word_embedding)


def main():
    # torch.set_printoptions(sci_mode=False)
    num_topic = 5
    raw_sequences, (items, item_embedding) = create_toydata(num_topic)
    graph_data, sequences, item_index_dict = preprocess(
        raw_sequences, (items, item_embedding))

    # TODO: read from command line, and initialize
    config = Config(
        num_item=len(items),
        num_seq=len(sequences),
        item_embedding=graph_data.x_item.size(1),
        d_model=300,
        epochs=100,
        output_dim=num_topic,
        lr=1,
        l2_lambda=0,
        verbose=False,
        dropout=0.2
    )

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
