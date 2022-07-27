from typing import List, Tuple
import torch
from torch import Tensor

from containers import BipartiteGraphData, Sequence, Item


def calc_seq_feature(
    sequences: List[Sequence],
    num_item: int
):
    x_seq = torch.zeros(len(sequences), num_item)
    for seq_index, seq in enumerate(sequences):
        for item_index in seq.indicies:
            x_seq[seq_index][item_index] += 1 / len(seq.indicies)
    return x_seq


def preprocess(
    raw_sequences: List[List[Item]],
    items: List[Tuple[Item, Tensor]]
) -> Tuple[BipartiteGraphData, List[Sequence], List[Item], dict[Item, int]]:
    r'''
    Preprocess data

    Args:
        sequences: raw representation of sequences, should be list of Item
        items:
            tuple of Item, and tensor of item embedding

    Return:
        graph_data, sequences, items, item_index_dict
    '''
    item_list, x_item = items
    sequences, item_index_dict = preprocess_sequences(raw_sequences, item_list)
    edge_index, edge_weight = construct_graph(sequences)

    x_seq = calc_seq_feature(sequences, len(item_list))

    graph_data = BipartiteGraphData(
        torch.tensor(edge_index, dtype=torch.long),
        torch.Tensor(edge_weight),
        x_item,
        x_seq
    )

    return graph_data, sequences, item_index_dict


def preprocess_sequences(
    raw_sequences: List[List[Item]],
    item_list: List[Item]
) -> Tuple[List[Sequence], dict]:
    r'''
    Preprocess raw sequence (document, purchase history) to Sequence
    Return:
        sequences, items, item_index_dict
    '''
    item_index_dict = {}

    for i, item in enumerate(item_list):
        item_index_dict[item] = i

    sequences = []
    for sequence in raw_sequences:
        indicies = [item_index_dict[e] for e in sequence]
        sequences.append(Sequence(sequence, indicies))

    return sequences, item_index_dict


def construct_graph(
    sequences: List[Sequence]
) -> Tuple[List[List[int]], List[int]]:
    r'''
    Construct graph from sequences

    Return:
        edge_index, edge_weight
            shape: 
                edge_index: (2, num_edge)
                edge_weight: (num_edges,)
    '''
    edge_dict = {}
    for seq_index, sequence in enumerate(sequences):
        for item_index in sequence.indicies:
            edge = (seq_index, item_index)
            if edge not in edge_dict:
                edge_dict[edge] = 0
            edge_dict[edge] += 1

    edge_seq, edge_item, edge_weight = [], [], []
    for (seq_index, item_index), count in edge_dict.items():
        edge_seq.append(seq_index)
        edge_item.append(item_index)
        # Calc frequency
        weight = count / len(sequences[seq_index].sequence)
        edge_weight.append(weight)

    return [edge_seq, edge_item], edge_weight
