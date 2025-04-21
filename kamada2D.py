import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import matplotlib.ticker as ticker

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 20

problem_name = 'RWMOP25'
algo_list = ['data']

domain_df = pd.read_csv('domain_info.csv')
row = domain_df.loc[domain_df['problem'] == problem_name].iloc[0]

lower = np.array([float(v) for v in row['lower'].split(",")])
upper = np.array([float(v) for v in row['upper'].split(",")])
diff = upper - lower

for algo in algo_list:
    print(f"Processing: {algo}")
    data = pd.read_csv(f'data09-20/{problem_name}_{algo}.csv')
    con_cols = [c for c in data.columns if c.startswith('Con_')]
    total = data[con_cols].apply(lambda row: np.sum(np.maximum(0, row)), axis=1)

    target_constraints = ['Con_1']
    sub = data[target_constraints].apply(lambda row: np.sum(np.maximum(0, row)), axis=1)

    data['CV'] = total
    data_sorted = data.sort_values(by=['ID', 'Gen'])
    G = nx.DiGraph()
    X_cols = [c for c in data.columns if c.startswith('X_')]

    for idx, row in data_sorted.iterrows():
        G.add_node(idx, Gen=row['Gen'], ID=row['ID'], X=row[X_cols].values, CV=row['CV'])

    prev_row = None
    prev_idx = None

    for idx, row in data_sorted.iterrows():
        if prev_row is not None and prev_row['ID'] == row['ID'] and row['Gen'] == prev_row['Gen'] + 1:
            G.add_edge(prev_idx, idx)
        prev_row = row
        prev_idx = idx

    vec2nodes = defaultdict(list)
    for n in G.nodes():
        vec = tuple(G.nodes[n]['X'])
        vec2nodes[vec].append(n)

    for vec, node_list in vec2nodes.items():
        if len(node_list) > 1:
            representative = node_list[0]
            duplicates = node_list[1:]
            for dup in duplicates:
                for pred in list(G.predecessors(dup)):
                    if pred != representative and not G.has_edge(pred, representative):
                        G.add_edge(pred, representative)
                for succ in list(G.successors(dup)):
                    if succ != representative and not G.has_edge(representative, succ):
                        G.add_edge(representative, succ)
                G.remove_node(dup)

    nodes = list(G.nodes())
    X_all = np.array([G.nodes[n]['X'] for n in nodes])
    X_all_norm = (X_all - lower) / diff

    dist_matrix = pairwise_distances(X_all_norm, metric='euclidean')
    epsilon = 1e-10
    dist_matrix[dist_matrix < epsilon] = epsilon

    dist_dict = {
        ni: {nj: dist_matrix[i, j] for j, nj in enumerate(nodes)}
        for i, ni in enumerate(nodes)
    }

    pos_1d = nx.kamada_kawai_layout(G, dist=dist_dict, dim=1)
    pos = {n: (pos_1d[n][0], G.nodes[n]['CV']) for n in nodes}

    sink_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
    final_feasible = [n for n in sink_nodes if G.nodes[n]['CV'] == 0]
    final_infeasible = [n for n in sink_nodes if G.nodes[n]['CV'] > 0]
    other_nodes = [n for n in nodes if n not in sink_nodes]
    midle_feasible = [n for n in other_nodes if G.nodes[n]['CV'] == 0]
    midle_infeasible = [n for n in other_nodes if G.nodes[n]['CV'] > 0]

    plt.figure(figsize=(15, 12))
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color='gray', alpha=0.5)

    nx.draw_networkx_nodes(G, pos, nodelist=midle_infeasible, node_size=50, node_color='skyblue', edgecolors='black')
    nx.draw_networkx_nodes(G, pos, nodelist=midle_feasible, node_size=50, node_color='salmon', edgecolors='black')
    nx.draw_networkx_nodes(G, pos, nodelist=final_feasible, node_size=50, node_color='red', edgecolors='black')
    nx.draw_networkx_nodes(G, pos, nodelist=final_infeasible, node_size=50, node_color='blue', edgecolors='black',
                           linewidths=2)

    ax = plt.gca()
    ax.set_yscale('symlog', linthresh=1e-4)
    log_formatter = ticker.LogFormatterSciNotation(base=10)
    ax.yaxis.set_major_formatter(log_formatter)

    ax.tick_params(axis='y', which='both', labelleft=True)
    ax.axis('on')
    plt.ylim(bottom=-1e-4, top=1e5)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.05)

    plt.show()