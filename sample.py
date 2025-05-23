import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import matplotlib.ticker as ticker
from ot.gromov import gromov_wasserstein2
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path

# -----------------------------
# Configurations
# -----------------------------
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 20

# -----------------------------
# Parameters
# -----------------------------
data_dir   = Path('data09-20-pre')
file_path  = data_dir / 'RWMOP22_data.csv'
n_clusters = 50

# -----------------------------
# 1. Load CSV and preprocess
# -----------------------------
df = pd.read_csv(file_path)
# compute total constraint violation
con_cols = [c for c in df.columns if c.startswith('Con_')]
df['CV'] = df[con_cols].clip(lower=0).sum(axis=1)
# sort by series ID and generation
df_sorted = df.sort_values(['ID','Gen']).reset_index(drop=True)
# capture decision variables for layout
X_cols = [c for c in df.columns if c.startswith('X_')]

# -----------------------------
# 2. Build directed graph G
# -----------------------------
G = nx.DiGraph()
prev = None; prev_idx = None
for idx, row in df_sorted.iterrows():
    G.add_node(idx,
               ID = row['ID'],
               Gen= row['Gen'],
               X  = row[X_cols].values.astype(float),
               CV = row['CV'])
    if prev is not None and row['ID']==prev['ID'] and row['Gen']==prev['Gen']+1:
        G.add_edge(prev_idx, idx)
    prev = row; prev_idx = idx

# -----------------------------
# 3. Extract per-series subgraphs
# -----------------------------
id_values = sorted({G.nodes[n]['ID'] for n in G.nodes()})
subgraphs = []
for idv in id_values:
    nodes_i = [n for n in G.nodes() if G.nodes[n]['ID']==idv]
    subgraphs.append(G.subgraph(nodes_i).copy())

# -----------------------------
# 4. Compute shortest-path matrices
# -----------------------------
dist_mats = []
for H in subgraphs:
    D = np.array(nx.floyd_warshall_numpy(H), dtype=float)
    # replace inf by large finite
    if np.isinf(D).any():
        maxf = np.nanmax(D[np.isfinite(D)])
        D[np.isinf(D)] = maxf * 10
    dist_mats.append(D)

# -----------------------------
# 5. Pairwise Gromov–Wasserstein distances
# -----------------------------
n = len(dist_mats)
W = np.zeros((n,n))
for i in range(n):
    for j in range(i+1,n):
        C1 = dist_mats[i]; C2 = dist_mats[j]
        p = np.ones(len(C1))/len(C1)
        q = np.ones(len(C2))/len(C2)
        # compute squared-loss GW cost
        gw_val = gromov_wasserstein2(C1, C2, p, q, loss_fun='square_loss')
        W[i,j] = W[j,i] = float(gw_val)

# -----------------------------
# 6. Cluster series and select medoids
# -----------------------------
cl = AgglomerativeClustering(
    n_clusters=n_clusters,
    metric='precomputed', linkage='average'
).fit(W)
labels = cl.labels_

rep_ids = []
for c in range(n_clusters):
    idxs = np.where(labels==c)[0]
    subW = W[np.ix_(idxs,idxs)]
    med  = idxs[np.argmin(subW.sum(axis=1))]
    rep_ids.append(id_values[med])
print("Representative series IDs:", rep_ids)

# -----------------------------
# 7. Filter original graph to reps
# -----------------------------
keep = [n for n in G.nodes() if G.nodes[n]['ID'] in rep_ids]
G_rep = G.subgraph(keep).copy()

# -----------------------------
# 8. 1D layout + CV visualization
# -----------------------------
nodes = list(G_rep.nodes())
X_all = np.vstack([G_rep.nodes[n]['X'] for n in nodes])
# normalize if domain info exists
# domain_df loading omitted here
# pos_1d via Kamada-Kawai on X_all distances
D = pairwise_distances(X_all, metric='euclidean')
D[D<1e-10] = 1e-10
dist_dict = {ni:{nj:D[i,j] for j,nj in enumerate(nodes)} for i,ni in enumerate(nodes)}
pos_1d = nx.kamada_kawai_layout(G_rep, dist=dist_dict, dim=1)
# assemble 2D positions: x=embedding, y=CV
pos = {n:(pos_1d[n][0], G_rep.nodes[n]['CV']) for n in nodes}

# classify nodes
sink = [n for n in nodes if G_rep.out_degree(n)==0]
final_f = [n for n in sink if G_rep.nodes[n]['CV']==0]
final_i = [n for n in sink if G_rep.nodes[n]['CV']>0]
mid = [n for n in nodes if n not in sink]
mid_f = [n for n in mid if G_rep.nodes[n]['CV']==0]
mid_i = [n for n in mid if G_rep.nodes[n]['CV']>0]

# plot
gg = plt.figure(figsize=(12,10))
ax = gg.gca()
nx.draw_networkx_edges(G_rep, pos, arrowstyle='->', arrowsize=12,
                       edge_color='gray', alpha=0.5)
nx.draw_networkx_nodes(G_rep, pos, nodelist=mid_i, node_size=50,
                       node_color='skyblue', edgecolors='black')
nx.draw_networkx_nodes(G_rep, pos, nodelist=mid_f, node_size=50,
                       node_color='salmon', edgecolors='black')
nx.draw_networkx_nodes(G_rep, pos, nodelist=final_f, node_size=60,
                       node_color='red', edgecolors='black')
nx.draw_networkx_nodes(G_rep, pos, nodelist=final_i, node_size=60,
                       node_color='blue', edgecolors='black')
ax.set_yscale('symlog', linthresh=1e-5)
ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(10))
plt.ylim(-1e-5, 1e4)
plt.title('Representative Series System Graph')
plt.tight_layout()
plt.show()
