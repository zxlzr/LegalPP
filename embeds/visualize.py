import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets, decomposition

# Set chinese font
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Colors for different group of entries
colors = ['b', 'g', 'r', 'c', 'm']


# Load entities
entity_names = []
with open('../data/entities.txt', 'r') as f:
    for line in f.readlines():
        entity_names.append(line.strip())
name2idx = {name: idx for idx, name in enumerate(entity_names)}
print('Loaded entity names.')


# Random choose laws
all_entries = [entity for entity in entity_names if '/' in entity]
law_dict = {}
for entry in all_entries:
    law_name = entry[:entry.find('_')]
    if law_name not in law_dict:
        law_dict[law_name] = []
    law_dict[law_name].append(entry)
laws = random.choices(list(law_dict.keys()), k=len(colors))


# Group points by colors
scatter_groups = [[] for _ in range(len(colors))]
entry_indices = []
for idx in range(len(colors)):
    law_name = laws[idx]
    for entry in law_dict[law_name]:
        scatter_groups[idx].append(len(entry_indices))
        entry_indices.append(name2idx[entry])
print('Randomly choose {} laws & {} entries'.format(
    len(laws), len(entry_indices)))
for i in range(len(laws)):
    print(laws[i], len(scatter_groups[i]))


# Visualization for embeddings
for embed_type in ['raw', 'tuned', 'rgcn', 'gat', 'rgcn_simple',
                   'rgcn_pre_tuned', 'rgcn_pre_raw', 'gat_pre_raw', 'gat_pre_tuned']:
    embed = np.load('emb_{}.npz'.format(embed_type))['embed']
    entry_embed = embed[entry_indices]
    print('Loaded {} embed.'.format(embed_type))

    # Do decomposition using SVD / PCA
    reduce_dim = 200
    # svd = decomposition.TruncatedSVD(n_components=reduce_dim, n_iter=10)
    # result = svd.fit_transform(entry_embed)
    pca = decomposition.PCA(n_components=reduce_dim, random_state=1)
    result = pca.fit_transform(entry_embed)
    print('Decomposition: reduced dim to {}'.format(result.shape))

    # T-SNE visualization
    tsne = manifold.TSNE(n_components=3, n_iter=1200,
                         random_state=1, verbose=True)
    result = tsne.fit_transform(result)
    print('t-sne done.')

    # Show and save image
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatte by group
    for idx, group in enumerate(scatter_groups):
        label = laws[idx]
        if len(label) > 20:
            label = label[:20] + '...'
        ax.scatter(xs=[result[index, 0] for index in group],
                   ys=[result[index, 1] for index in group],
                   zs=[result[index, 2] for index in group],
                   c=colors[idx],
                   label=label,
                   marker='o')
    ax.legend(loc='lower center')
    fig.savefig('tsne/embed_{}.png'.format(embed_type))
    print('Visualization saved to tsne/embed_{}.png'.format(embed_type))
