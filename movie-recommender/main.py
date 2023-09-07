import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

ratings = pd.read_csv("./ratings.csv")
print(ratings.head())

movies = pd.read_csv("./movies.csv")
print(movies.head())

n_ratings = len(ratings)
n_movies = len(ratings["movieId"].unique())
n_users = len(ratings["userId"].unique())

print(f"Number of ratings: {n_ratings}")
print(f"Number of unique movies: {n_movies}")
print(f"Number of unique users: {n_users}")
print(f"Average ratings per user: {round(n_ratings/n_users, 2)}")
print(f"Average ratings per movie: {round(n_ratings/n_movies, 2)}")

# Create set of users and their rating counts
user_freq = ratings[["userId", "movieId"]].groupby("userId").count().reset_index()
user_freq.columns = ["userId", "n_ratings"]
print(user_freq.head())

# Find average movie rating
mean_ratings = ratings.groupby("movieId")[["rating"]].mean()

# Find lowest rated movies
lowest_rated = mean_ratings["rating"].idxmin()
print(movies.loc[movies["movieId"] == lowest_rated])

# Find highest rated movies
highest_rated = mean_ratings["rating"].idxmax()
print(movies.loc[movies["movieId"] == highest_rated])

# Find people who rated highest movie
print(ratings[ratings["movieId"] == highest_rated])

# Find people who rated lowest movie
print(ratings[ratings["movieId"] == lowest_rated])

# Find average rating and review count for reach movie
movie_stats: pd.DataFrame = ratings.groupby("movieId")[["rating"]].agg(
    ["count", "mean"]
)
movie_stats.columns = movie_stats.columns.droplevel()

print(movie_stats.head())


# Create user-item matrix
def create_matrix(df: pd.DataFrame):
    N = len(df["userId"].unique())
    M = len(df["movieId"].unique())

    # Map ids to indices
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))

    # Map indices to ids
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df["userId"]]
    movie_index = [movie_mapper[i] for i in df["movieId"]]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper


X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

"""
Find similar movies using KNN
"""


def find_similar_movies(
    movie_ids, X: csr_matrix, k, metric="cosine", show_distance=False
):
    neighbor_ids = []
    neighbor_distances = []

    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    movie_inds = [movie_mapper[movie_id] for movie_id in movie_ids]
    movie_vecs = [X[movie_ind] for movie_ind in movie_inds]
    movie_vec = sum(movie_vecs) / len(movie_vecs)
    movie_vec = movie_vec.reshape(1, -1)
    dist, neighbor = kNN.kneighbors(movie_vec, return_distance=show_distance)

    for i in range(0, k):
        n = neighbor.item(i)
        neighbor_ids.append(movie_inv_mapper[n])
        neighbor_distances.append(dist[0][i])

    # Remove the requested movie
    # neighbor_ids.pop(0)
    # neighbor_distances.pop(0)
    return (neighbor_ids, neighbor_distances)


all_movie_titles = dict(zip(movies["movieId"], movies["title"]))


movie_ids = [4993, 79132]

(similar_ids, distances) = find_similar_movies(movie_ids, X, k=10, show_distance=True)
movie_title = [all_movie_titles[movie_id] for movie_id in movie_ids]

print(f"Since you watched {movie_title}")
for i in range(len(similar_ids)):
    print(distances[i], all_movie_titles[similar_ids[i]])
