import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
# Read data from CSV files
data = pd.read_csv("../../results/edges/Train_edges.csv")
data_all = pd.read_csv("../../results/edges/All_edges.csv")
link_gt_path = "../../results/links/test_links_sci_ai.json"
with open(link_gt_path,'r') as f:
    link_gt = json.load(f)
link_gt_path_ai_sci = "../../results/links/test_links_ai_sci.json"
with open(link_gt_path_ai_sci,'r') as f:
    link_gt_ai_sci = json.load(f)
def compute_top_k_katz_links(graph, start_node, k=5, alpha=0.1):
    A = nx.to_numpy_array(graph, weight='weight')
    n = A.shape[0]
    I = np.eye(n)
    katz_similarity_matrix = np.linalg.inv(I - alpha * A) - I
    nodes = list(graph.nodes)
    node_index = {i: nodes[i] for i in range(n)}
    reverse_index = {v: k for k, v in node_index.items()}
    start_node_idx = reverse_index[start_node]
    katz_scores_from_start = {}
    for j in range(n):
        if j != start_node_idx:
            target_node = node_index[j]
            if start_node.split("_")[0] != target_node.split("_")[0]:
                katz_scores_from_start[target_node] = katz_similarity_matrix[start_node_idx, j]
    top_k_links = sorted(katz_scores_from_start.items(), key=lambda x: x[1], reverse=True)[:k]
    top_k_links_formatted = [(start_node, target, score) for target, score in top_k_links]
    return top_k_links_formatted

G = nx.Graph()
U = data_all['start'].unique()
V = data_all['end'].unique()
G.add_nodes_from(U, bipartite=0)
G.add_nodes_from(V, bipartite=1)
total_node = np.unique(data['start'].tolist() + data['end'].tolist())
node_U = np.unique([a[0] for a in link_gt])
node_V = np.unique([a[0] for a in link_gt_ai_sci])

def csv_to_list(edges):
    start = edges['start'].tolist()
    end = edges['end'].tolist()
    return [(start[i], end[i]) for i in range(len(start))]

weighted_edges = list(data.itertuples(index=False, name=None))
G.add_weighted_edges_from(weighted_edges)
def parse_target_set(link_gt):
    set_pool = {}
    for edge in link_gt:
        start, end = edge
        if start not in set_pool.keys():
            set_pool[start] = []
        set_pool[start].append(end)
    for key in set_pool.keys():
        set_pool[key] = np.unique(set_pool[key])
    return set_pool
def parse_source_ratio(link_gt,node):
    return len([i for i in link_gt if i[0] == node]) / len(link_gt)
edges_sci_ai = []
edges_ai_sci = []
test_U = [i[0] for i in link_gt]
test_V = [i[1] for i in link_gt]
gt_pool_sci_ai = parse_target_set(link_gt)
gt_pool_ai_sci = parse_target_set(link_gt_ai_sci)
for k in [1,3,5,10]:
    edges_dict_sci_ai = {}
    edges_dict_ai_sci = {}
    print(f"Experimenting for {k}")
    for u in tqdm(node_U):
        top_k_links = compute_top_k_katz_links(G, start_node=u, k=k, alpha=0.1)
        for start, target, score in top_k_links:
            assert start == u
            if start not in edges_dict_sci_ai.keys():
                edges_dict_sci_ai[start] = []
            edges_dict_sci_ai[start].append(target)
    for v in tqdm(node_V):
        top_k_links = compute_top_k_katz_links(G, start_node=v, k=k, alpha=0.1)
        for start, target, score in top_k_links:
            assert start == v
            if start not in edges_dict_ai_sci.keys():
                edges_dict_ai_sci[start] = []
            edges_dict_ai_sci[start].append(target)
    precision_sci_ai = 0
    recall_sci_ai = 0
    for start in edges_dict_sci_ai.keys():
        if start in gt_pool_sci_ai.keys():
            gt_pool = gt_pool_sci_ai[start]
            pred = edges_dict_sci_ai[start]
            precision_sci_ai += (len([p for p in pred if p in gt_pool]) / len(pred)) * parse_source_ratio(link_gt,start)
            recall_sci_ai += (len([gt for gt in gt_pool if gt in pred]) / len(gt_pool)) * parse_source_ratio(link_gt,start)
    
    precision_ai_sci = 0
    recall_ai_sci = 0
    ratio_sum = 0
    for start in edges_dict_ai_sci.keys():
        if start in gt_pool_ai_sci.keys():
            gt_pool = gt_pool_ai_sci[start]
            pred = edges_dict_ai_sci[start]
            precision_ai_sci += (len([p for p in pred if p in gt_pool]) / len(pred)) * parse_source_ratio(link_gt_ai_sci,start)
            recall_ai_sci += (len([gt for gt in gt_pool if gt in pred]) / len(gt_pool)) * parse_source_ratio(link_gt_ai_sci,start)
            ratio_sum += parse_source_ratio(link_gt_ai_sci,start)
    new_links_path_sci_ai = f"your_path.json"
    new_links_path_ai_sci = f"your_path.json"
    new_link_sci_ai = []
    new_link_ai_sci = []
    for start in edges_dict_sci_ai.keys():
        pred = edges_dict_sci_ai[start]
        new_link_sci_ai.extend([(start,p) for p in pred])
    for start in edges_dict_ai_sci.keys():
        pred = edges_dict_ai_sci[start]
        new_link_ai_sci.extend([(start,p) for p in pred])
    with open(new_links_path_sci_ai,"w") as f:
        json.dump(new_link_sci_ai,f)
    with open(new_links_path_ai_sci,"w") as f:
        json.dump(new_link_ai_sci,f)
    print(f"Ratio sum: {ratio_sum}")
    print(f"Recall sci->ai: {recall_sci_ai}")
    print(f"Precision sci->sci: {precision_sci_ai}")
    print(f"Recall ai->sci: {recall_ai_sci}")
    print(f"Precision ai->sci: {precision_ai_sci}")
    print("="*100)
