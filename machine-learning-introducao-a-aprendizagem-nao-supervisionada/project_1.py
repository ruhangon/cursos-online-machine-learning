from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

uri = "https://raw.githubusercontent.com/oyurimatheus/clusterirng/master/movies/movies.csv"
dataset = pd.read_csv(uri)

# print(dataset.info())
# print(dataset.head())

# extract dummies of genres column
genres = dataset.genres.str.get_dummies()

movies_data = pd.concat([dataset, genres], axis=1)

# print(movies_data.head())

scaler = StandardScaler()

scaled_genres = scaler.fit_transform(genres)
# print(scaled_genres)

model = KMeans(n_clusters=3)
model.fit(scaled_genres)

# visualize the results of k-means
# print(model.labels_)

# print(genres.columns)

# print the centers of each group
# print(model.cluster_centers_)

groups = pd.DataFrame(model.cluster_centers_, columns=genres.columns)
groups = groups.transpose()

# print(groups)

titles = dataset.title
genres = dataset.genres
groups_s = pd.Series(model.labels_)
groups_list = pd.concat([titles, genres, groups_s], axis=1)

# change the names of the columns
groups_list.columns = ["title", "genres", "group"]

# change the index
groups_list.set_index(keys = "title", inplace = True)

# print(groups_list.head(n=10))

group0 = groups_list.query("group==0")
print(group0.head(n=10))

# inertia_ of the model
for n in range(1, 21):
    model = KMeans(n_clusters=n)
    model.fit(scaled_genres)
    print("inertia with %d groups" % n)
    print(model.inertia_)

