from torch_geometric.datasets import MovieLens
import pandas as pd


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
