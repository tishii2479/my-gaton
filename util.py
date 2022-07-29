from typing import List
import torch
from torch import Tensor
import numpy as np

from containers import Sequence

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
        num_top_item: number of item to list
        num_item: number of items in data
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


def top_cluster_items(
    num_cluster: int,
    cluster_labels: List[int],
    sequences: List[Sequence],
    num_top_item: int,
    num_item: int
) -> List[List[int]]:
    r'''
    Args:
        num_cluster: number of clusters
        topic_indicies: list of sequence indicies for each topic
            shape: (num_topic, len(sequence_indicies))
        sequences: all sequence data
        num_top_item: number of item to list
        num_item: number of items in data
    Return:
        top items for each cluster
            shape: (num_topic, num_top_item)
    '''
    item_counts = np.zeros((num_cluster, num_item))

    for i, sequence in enumerate(sequences):
        for item_index in sequence.indicies:
            item_counts[cluster_labels[i]][item_index] += 1

    top_items = []
    for topic in range(num_cluster):
        # Get item index of top `num_top_item` items which has larget item_count
        top_items_for_topic = list(
            item_counts[topic].argsort()[::-1][:num_top_item])
        top_items.append(top_items_for_topic)
    return top_items


def visualize_cluster(
    features: List[Tensor],
    num_cluster: int,
    cluster_labels: List[int]
):
    r'''
    Visualize cluster to 2d
    '''
    pca = PCA(n_components=2)
    pca.fit(features)
    pca_features = pca.fit_transform(features)

    print(pca_features)

    colors = cm.rainbow(np.linspace(0, 1, num_cluster))
    plt.figure()

    for i in range(pca_features.shape[0]):
        plt.scatter(x=pca_features[i, 0], y=pca_features[i, 1],
                    color=colors[cluster_labels[i]])

    plt.show()
