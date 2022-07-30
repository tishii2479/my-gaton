from gensim.models import word2vec
import torch
from torch_geometric.datasets import MovieLens
import pandas as pd
import numpy as np


def create_movielens_data():
    dataset = MovieLens('data')[0]

    movie_embeddings = dataset['movie']['x']
    movies_df = pd.read_csv('data/raw/ml-latest-small/movies.csv')
    titles = movies_df['title'].values
    genres = movies_df['genres'].values

    movies = [titles[i] + genres[i] for i in range(len(titles))]

    user_size = dataset['user']['num_nodes']

    edge_index = dataset['user', 'rates', 'movie'].edge_index
    raw_history = [[] for _ in range(user_size)]

    for user_index, movie_index in zip(edge_index[0], edge_index[1]):
        raw_history[user_index].append(movies[movie_index.item()])

    return (raw_history, None), (movies, movie_embeddings)


def create_hm_data():
    sequences = pd.read_csv('data/hm/purchase_history.csv')
    items = pd.read_csv('data/hm/items.csv', dtype={'article_id': str})

    raw_sequences = [sequence.split(' ')
                     for sequence in sequences.sequence.values[:1000]]
    seq_labels = None

    item_names = items.name.values
    item_ids = items.article_id.values

    item_list = [item_ids[i] for i in range(len(item_ids))]

    word2vec_model = word2vec.Word2Vec.load('weights/word2vec.model')

    print(f'calculating item embedding, size: {len(items)}')
    item_embedding = torch.Tensor(
        np.array([word2vec_model.wv[id] for id in items.article_id.values]))
    print('calculating item embedding is done.')

    return (raw_sequences, seq_labels), (item_list, item_embedding)
