import torch
from torch import Tensor


def group_topics(
    h_seq: Tensor,
    num_topic: int
):
    r'''
    Group topics by choosing the highest probability
    TODO: Make 'how to choose' function injectable
    Return:
        list of sequence indicies for each topic
        shape: (num_topic, len(sequence_indicies))
    '''
    topics = [[] for _ in range(num_topic)]

    for i, h in enumerate(h_seq):
        topic = torch.argmax(h).item()
        topics[topic].append(i)
        print(f'index: {i}, probability: {h}, to: {topic}')

    return topics
