
import json
from collections import Counter
import numpy as np
import pandas as pd
with open("../../results/cluster_labels/cluster_name_Sci.json", "r") as f:
    top_words_sci = json.load(f)
with open("../../results/cluster_labels/cluster_name_AI.json", "r") as f:
    top_words_ai = json.load(f)
with open("../../results/cluster_labels/cluster_labels_scientific_problem.json", "r") as f:
    cluster_labels_sci = json.load(f)
with open("../../results/cluster_labels/cluster_labels_AI_method.json", "r") as f:
    cluster_labels_ai = json.load(f)
with open("../../results/cluster_labels/well_explored_AI_clusters.json", "r") as f:
    well_ai = json.load(f)
with open("../../results/cluster_labels/under_explored_AI_clusters.json", "r") as f:
    under_ai = json.load(f)
with open("../../results/cluster_labels/well_explored_Sci_clusters.json", "r") as f:
    well_sci = json.load(f)
with open("../../results/cluster_labels/under_explored_Sci_clusters.json", "r") as f:
    under_sci = json.load(f)

# Function to parse node names
def parse_name(name):
    idx = int(name.split("_")[-1])
    if "Sci" in name:
        if len(top_words_sci[idx]) > 0:
            return top_words_sci[idx]
        else:
            return []
    else:
        if len(top_words_ai[idx]) > 0:
            return top_words_ai[idx]
        else:
            return []
        
def parse_name_size(name):
    idx = int(name.split("_")[-1])
    if "Sci" in name:
        return len([i for i in cluster_labels_sci if i == idx])
    else:
        return len([i for i in cluster_labels_ai if i == idx])

def count_new_links(train_links,pred_links):
    train_link_set = set(train_links)
    pred_link_set = set(pred_links)
    diff = pred_link_set - train_link_set
    for i in diff:
        assert i in list(pred_link_set)
    return diff

def get_human_newlinks(train_links):
    new_links_path = f"../../results/links/test_links_sci_ai.json"
    with open(new_links_path,'r') as f:
        link_pred = json.load(f)
    print(f"Human set has {len(link_pred)} links")
    link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0])]
    diff = count_new_links(train_links,link_pred)
    print(f"Human sci->ai: {len(diff)}")

def get_human_newlinks_merged(train_links):
    new_links_path = f"your_path.json"
    new_links_path_ai_sci = f"your_path.json"
    with open(new_links_path,'r') as f:
        link_pred = json.load(f)
    with open(new_links_path_ai_sci,'r') as f:
        link_pred_ai_sci = json.load(f)

    link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0]) and "N/A" not in parse_name(i[1])]
    link_pred_ai_sci = [(i[1],i[0]) for i in link_pred_ai_sci if "N/A" not in parse_name(i[0]) and "N/A" not in parse_name(i[1])]
    link_pred = link_pred + link_pred_ai_sci
    new_links = count_new_links(train_links,link_pred)
    print(f"Pred set has {len(link_pred)} links")
    print(f"LLM new links: {len(new_links)}")
    link_pred = [p for p in link_pred if p in new_links]
    print(f"New link pred set: {len(link_pred)}")
    counter_sci_ai = Counter(link_pred)
    most_common_pred = counter_sci_ai.most_common()
    rows = []
    for top_idx in range(len(most_common_pred)):
        start, end = most_common_pred[top_idx][0]
        frequency = most_common_pred[top_idx][1]
        if "N/A" not in start:
            rows.append({
            'start_name': parse_name(start),
            'end_name': parse_name(end),
            'frequency': frequency,
            'start_size': parse_name_size(start),
            'end_size': parse_name_size(end),
        })
    node2vec_sci_ai_df = pd.DataFrame(rows, columns=["start_name", "end_name", "frequency", "start_size", "end_size"])
    assert len(node2vec_sci_ai_df) == len(new_links)

def get_human_newlinks_ai_sci(train_links):
    new_links_path = f"../../results/links/test_links_sci_ai.json"
    with open(new_links_path,'r') as f:
        link_pred = json.load(f)
    print(f"Human set has {len(link_pred)} links")
    link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0])]
    diff = count_new_links(train_links,link_pred)
    print(f"Human ai->sci: {len(diff)}")

def get_LLM_newlinks_merged(train_links):
    for k in ["oneshot","threeshot","fiveshot","tenshot"]:
        print(f"Running experiment for {k}")
        new_links_path = f"your_path.json"
        new_links_path_ai_sci = f"your_path.json"
        with open(new_links_path,'r') as f:
            link_pred = json.load(f)
        with open(new_links_path_ai_sci,'r') as f:
            link_pred_ai_sci = json.load(f)

        link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0]) and "N/A" not in parse_name(i[1])]
        link_pred_ai_sci = [(i[1],i[0]) for i in link_pred_ai_sci if "N/A" not in parse_name(i[0]) and "N/A" not in parse_name(i[1])]
        link_pred = link_pred + link_pred_ai_sci
        new_links = count_new_links(train_links,link_pred)
        print(f"Pred set has {len(link_pred)} links")
        print(f"LLM new links: {len(new_links)}")
        link_pred = [p for p in link_pred if p in new_links]
        print(f"New link pred set: {len(link_pred)}")
        counter_sci_ai = Counter(link_pred)
        most_common_pred = counter_sci_ai.most_common()
        rows = []
        for top_idx in range(len(most_common_pred)):
            start, end = most_common_pred[top_idx][0]
            frequency = most_common_pred[top_idx][1]
            if "N/A" not in start:
                rows.append({
                'start_name': parse_name(start),
                'end_name': parse_name(end),
                'frequency': frequency,
                'start_size': parse_name_size(start),
                'end_size': parse_name_size(end),
            })
        node2vec_sci_ai_df = pd.DataFrame(rows, columns=["start_name", "end_name", "frequency", "start_size", "end_size"])
        assert len(node2vec_sci_ai_df) == len(new_links)


def get_LLM_newlinks(train_links):
    for i in ["oneshot","threeshot","fiveshot","tenshot"]:
        print(f"Running experiment for {i}")
        new_links_path = f"your_path.json"
        with open(new_links_path,'r') as f:
            link_pred = json.load(f)
        links = []
        print(f"Pred set has {len(link_pred)} links")
        link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0])]
        new_links = count_new_links(train_links,link_pred)
        print(f"LLM sci->ai new links: {len(new_links)}")


def get_LLM_newlinks_ai_sci(train_links):
    for i in ["oneshot","threeshot","fiveshot","tenshot"]:
        print(f"Running experiment for {i}")
        new_links_path = f"../../results/links/new_links/your_path.json"
        with open(new_links_path,'r') as f:
            link_pred = json.load(f)
        links = []
        print(f"Pred set has {len(link_pred)} links")
        link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0])]
        new_links = count_new_links(train_links,link_pred)
        print(f"LLM ai->sci new links: {len(new_links)}")

def get_node2vec_newlinks(train_links):
    for k in [1,3,5,10]:
        new_links_path_sci_ai = f"../../results/links/newlinks/Node2vec_sci_ai_{k}.json"
        with open(new_links_path_sci_ai,'r') as f:
            link_pred = json.load(f)
        links = []
        print(f"Pred set has {len(link_pred)} links")
        link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0])]
        new_links = count_new_links(train_links,link_pred)
        print(f"Node2vec with k = {k} has {len(new_links)} new Sci -> AI links")
        node2vec_sci_ai = []
        for new_link in link_pred:
            start, end = new_link
            if "N/A" not in parse_name(start) and new_link in new_links:
                node2vec_sci_ai.append({
                    'start_name': parse_name(start),
                    'end_name': parse_name(end),
                    'start_size': parse_name_size(start),
                    'end_size': parse_name_size(end),
                })
        print(len(node2vec_sci_ai))

def get_node2vec_newlinks_ai_sci(train_links):
    for k in [1,3,5,10]:
        new_links_path_ai_sci = f"../../results/links/newlinks/Node2vec_ai_sci_{k}.json"
        with open(new_links_path_ai_sci,'r') as f:
            link_pred = json.load(f)
        links = []
        print(f"Pred set has {len(link_pred)} links")
        link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0])]
        new_links = count_new_links(train_links,link_pred)
        print(f"Node2vec with k = {k} has {len(new_links)} new AI -> Sci links")

def get_LLM_cluster_newlinks_ai_sci(train_links):
    for k in ["oneshot","threeshot","fiveshot","tenshot"]:
        new_links_path_ai_sci = f"../../results/links/newlinks/LLM(Cluster)_sci_ai_{k}.json"
        with open(new_links_path_ai_sci,'r') as f:
            link_pred = json.load(f)
        links = []
        print(f"Pred set has {len(link_pred)} links")
        link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0])]
        new_links = count_new_links(train_links,link_pred)
        #print(new_links)
        print(f"LLM cluster with k = {k} has {len(new_links)} new AI -> Sci links")
        parse_link_well_under_sci_LLM(new_links)

def read(path):
    with open(path,'r') as f:
        link = json.load(f)
    link = [(i[0],i[1]) for i in link]
    return link

def parse_link_well_under_sci(links):
    well_clusters_sci = [top_words_sci[i] for i in well_sci]
    well_clusters_ai = [top_words_ai[i] for i in well_ai]
    under_clusters_sci = [top_words_sci[i] for i in under_sci]
    under_clusters_ai = [top_words_ai[i] for i in under_ai]
    start = links['start_name'].tolist()
    end = links['end_name'].tolist()
    links_list = [(start[i],end[i]) for i in range(len(start))]
    well_well_links = [link for link in links_list if (link[0] in well_clusters_sci or link[0] in well_clusters_ai) and (link[1] in well_clusters_sci or link[1] in well_clusters_ai)]
    under_under_links = [link for link in links_list if (link[0] in under_clusters_sci or link[0] in under_clusters_ai) and (link[1] in under_clusters_sci or link[1] in under_clusters_ai)]
    well_under_links = [link for link in links_list if (link[0] in well_clusters_sci or link[0] in well_clusters_ai) and (link[1] in under_clusters_sci or link[1] in under_clusters_ai)]
    under_well_links = [link for link in links_list if (link[0] in under_clusters_sci or link[0] in under_clusters_ai) and (link[1] in well_clusters_sci or link[1] in well_clusters_ai)]
    print(f"Total links: {len(links_list)}")
    print(f"Well-well links: {len(well_well_links)}")
    print(f"Under-under links: {len(under_under_links)}")
    print(f"Well-under links: {len(well_under_links)}")
    print(f"Under-well links: {len(under_well_links)}")


def parse_link_well_under_sci_LLM(links):
    well_clusters_sci = [top_words_sci[i] for i in well_sci]
    well_clusters_ai = [top_words_ai[i] for i in well_ai]
    under_clusters_sci = [top_words_sci[i] for i in under_sci]
    under_clusters_ai = [top_words_ai[i] for i in under_ai]
    def parse_name(string):
        idx = int(string.split("_")[-1])
        if "Sci" in string:
            return top_words_sci[idx]
        else:
            return top_words_ai[idx]
    links_list = [(parse_name(i[0]),parse_name(i[1])) for i in links]
    well_well_links = [link for link in links_list if (link[0] in well_clusters_sci or link[0] in well_clusters_ai) and (link[1] in well_clusters_sci or link[1] in well_clusters_ai)]
    under_under_links = [link for link in links_list if (link[0] in under_clusters_sci or link[0] in under_clusters_ai) and (link[1] in under_clusters_sci or link[1] in under_clusters_ai)]
    well_under_links = [link for link in links_list if (link[0] in well_clusters_sci or link[0] in well_clusters_ai) and (link[1] in under_clusters_sci or link[1] in under_clusters_ai)]
    under_well_links = [link for link in links_list if (link[0] in under_clusters_sci or link[0] in under_clusters_ai) and (link[1] in well_clusters_sci or link[1] in well_clusters_ai)]
    print(f"Total links: {len(links_list)}")
    print(f"Well-well links: {len(well_well_links)}")
    print(f"Under-under links: {len(under_under_links)}")
    print(f"Well-under links: {len(well_under_links)}")
    print(f"Under-well links: {len(under_well_links)}")

link_gt_train_path = "../../results/links/train_links_sci_ai.json"
with open(link_gt_train_path,'r') as f:
    link_gt_train = json.load(f)
print(f"Number of ground truth train links: {len(link_gt_train)}")
link_gt_train = [(i[0],i[1]) for i in link_gt_train]

link_gt_train_ai_sci_path = "../../results/links/train_links_ai_sci.json"

with open(link_gt_train_ai_sci_path,'r') as f:
    link_gt_train_ai_sci = json.load(f)

print(f"Number of ground truth train links: {len(link_gt_train_ai_sci)}")
link_gt_train_ai_sci = [(i[0],i[1]) for i in link_gt_train_ai_sci]

link_gt_test = read("../../results/links/test_links_sci_ai.json")

get_LLM_cluster_newlinks_ai_sci(link_gt_train_ai_sci)
    