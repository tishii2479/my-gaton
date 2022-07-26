from typing import List, Tuple
import torch

from containers import BipartiteGraphData, Sequence, Item


def preprocess(
    sequences: List
) -> Tuple[BipartiteGraphData, List[Sequence], List[Item], dict[Item, int]]:
    r'''
    Preprocess data

    Return:
        graph_data, sequences, items, item_index_dict
    '''
    sequences, items, item_index_dict = preprocess_sequences(sequences)
    edge_index, edge_weight = construct_graph(sequences)

    # TODO: Get embedding for item
    x_item = torch.eye(len(items))

    # TODO: Move to somewhere else
    x_seq = torch.zeros(len(sequences), len(items))
    for seq_index, seq in enumerate(sequences):
        for item_index in seq.indicies:
            x_seq[seq_index][item_index] += 1 / len(seq.indicies)

    graph_data = BipartiteGraphData(
        torch.tensor(edge_index, dtype=torch.long),
        torch.Tensor(edge_weight),
        x_item,
        x_seq
    )

    return graph_data, sequences, items, item_index_dict


def preprocess_sequences(
    raw_sequences: List[List[Item]]
) -> Tuple[List[Sequence], List[Item], dict]:
    r'''
    Preprocess raw sequence (document, purchase history) to Sequence
    Return:
        sequences, items, item_index_dict
    '''
    item_set = set()
    for sequence in raw_sequences:
        for item in sequence:
            item_set.add(item)

    items = list(item_set)
    item_index_dict = {}

    for i, item in enumerate(items):
        item_index_dict[item] = i

    sequences = []
    for sequence in raw_sequences:
        indicies = [item_index_dict[e] for e in sequence]
        sequences.append(Sequence(sequence, indicies))

    return sequences, items, item_index_dict


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
