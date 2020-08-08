import numpy as np
import sklearn.preprocessing as prep
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


# Access data
with open('data/RNA_seq_data.csv') as file:
    lines = file.read().splitlines()
    data = [line.split(',') for line in lines[1:-1]]
    exclude = ['CR43283']
    data = [i for i in data if i[0] not in exclude]


# Organise data
gene_names = [i[0] for i in data]
all_reads = np.array([i[1:] for i in data]).astype("float")
all_indices = np.arange(len(all_reads))
all_reads_labelled = np.vstack([all_reads.T, all_indices]).T


# Clear data: remove all genes with less than 5 reads accross all conditions
selected_indices = np.where((all_reads_labelled[:, 0] >= 5) & \
                            (all_reads_labelled[:, 1] >= 5) & \
                            (all_reads_labelled[:, 2] >= 5))[0]

selected_reads = all_reads_labelled[selected_indices]


# Store delected genes names for labelling later
labels = np.array([gene_names[int(i)] for i in selected_reads[:,3]])


# Normalise data: bring to 0,1 scale by applying min-max normalizarion
X = prep.RobustScaler().fit_transform(selected_reads[:, 0:3])


# Perform PCA
N = 2
pca = PCA(n_components=N)
components = pca.fit_transform(X)
var_list = pca.explained_variance_ratio_
variance = np.sum(var_list)
print(f"\nVariance explained: {variance:.2f}")
pc1, pc2 = components[:,0], components[:,1]


# Perform t-sne
# tsne = TSNE(n_components=2).fit_transform(components)
# t1, t2 = tsne[:,0], tsne[:,1]
# plt.scatter(t1, t2)





# Perform k means clustering
K = 8
km = KMeans(n_clusters=K, init='random', n_init=10, max_iter=300, tol=1e-04,
            random_state=0)
clusters = km.fit_predict(components)


# # Perform DBSCAN
# test = DBSCAN().fit(components)
# core_samples_mask = np.zeros_like(test.labels_, dtype=bool)
# core_samples_mask[test.core_sample_indices_] = True
# labels_ = test.labels_
# clusters_ = set(labels_)


# fig, ax = plt.subplots()
# for i in clusters_:
#     indices = np.where(clusters == i)
#     ax.scatter(pc1[indices], pc2[indices], s=15)
#     plt.show()
    
# Visualise results
fig, ax = plt.subplots()

for i in range(K):
    indices = np.where(clusters == i)[0]
    ax.scatter(pc1[indices], pc2[indices], s=15)
    
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=50,
            marker='x', c='black', label='centroids')

# for x, y, name in zip(pc1, pc2, labels):
#     ax.annotate(f'{name}', xy=(x,y), fontsize=8)

plt.tight_layout()
plt.show()











# Find optimal number of clusters
# distortions = []
# for i in range(1, 11):
#     km = KMeans(
#         n_clusters=i, init='random',
#         n_init=10, max_iter=300,
#         tol=1e-04, random_state=0
#     )
#     km.fit(components)
#     distortions.append(km.inertia_)

# # plot
# plt.plot(range(1, 11), distortions, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Distortion')
# plt.show()

