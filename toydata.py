from random import randint, choice
import torch


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
        for _ in range(5):
            doc = []
            for _ in range(randint(20, 40)):
                doc.append(choice(key_words[i]))
            # for _ in range(randint(15, 20)):
            #     doc.append(choice(words))
            documents.append(doc)
    word_embedding = torch.eye(len(words))

    return (documents, None), (words, word_embedding)


def create_labeled_toydata(num_topic: int):
    (documents, _), (words, word_embedding) = create_toydata(num_topic)
    labels = []
    for i in range(num_topic):
        for _ in range(5):
            labels.append(i)
    return (documents, labels), (words, word_embedding)
