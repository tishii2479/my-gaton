from typing import List
import torch
from torch import Tensor
import numpy as np

from containers import Sequence


def group_topics(
    h_seq: Tensor,
    num_topic: int
) -> List[List[int]]:
    r'''
    Group topics by choosing the highest probability
    Args:
        h_seq:
            hidden representation for sequences
            the value should describe possibility, so the sum should be 1
            shape: (num_seq, num_topic)
        num_topic: number of topics
    Return:
        list of sequence indicies for each topic
        shape: (num_topic, len(sequence_indicies))
    '''
    topic_indicies = [[] for _ in range(num_topic)]

    for i, h in enumerate(h_seq):
        # TODO: Make 'how to choose' function injectable
        # NOTE: it is now `argmax(h)`
        topic = torch.argmax(h).item()
        topic_indicies[topic].append(i)
        print(f'index: {i}, probability: {h}, to: {topic}')

    return topic_indicies


def top_topic_items(
    topic_indicies: List[List[int]],
    sequences: List[Sequence],
    num_top_item: int,
    num_item: int
) -> List[List[int]]:
    r'''
    Args:
        topic_indicies: list of sequence indicies for each topic
            shape: (num_topic, len(sequence_indicies))
        sequences: all sequence data
    Return:
        top items for each topic
            shape: (num_topic, num_top_item)
    '''
    num_topic = len(topic_indicies)
    item_counts = np.zeros((num_topic, num_item))

    for topic, sequence_indicies in enumerate(topic_indicies):
        for index in sequence_indicies:
            for item_index in sequences[index].indicies:
                item_counts[topic][item_index] += 1

    top_items = []
    for topic in range(num_topic):
        # Get item index of top `num_top_item` items which has larget item_count
        top_items_for_topic = list(
            item_counts[topic].argsort()[::-1][:num_top_item])
        top_items.append(top_items_for_topic)
    return top_items
