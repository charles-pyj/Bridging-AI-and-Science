from pyvis.network import Network
import networkx as nx
import pandas as pd
import json
import numpy as np
import matplotlib.colors as mcolors

B = nx.Graph()
with open("../results/cluster_labels/cluster_labels_scientific_problem.json","r") as f:
    cluster_labels_sci = json.load(f)
with open("../results/cluster_labels/cluster_labels_AI_method.json","r") as f:
    cluster_labels_ai = json.load(f)
with open("../results/cluster_labels/cluster_name_Sci.json", "r") as f:
    top_words_sci = json.load(f)
with open("../results/cluster_labels/cluster_name_AI.json", "r") as f:
    top_words_ai = json.load(f)

sci_df = pd.read_csv("../results/tables/Science_Clusters.csv")
ai_df = pd.read_csv("../results/tables/AI_Clusters.csv")

def generate_degree_dict(df, prefix="Sci"):
    degree_dict = {}
    idx = df["cluster_idx"]
    degree = df["degree"]
    for i in range(len(idx)):
        degree_dict[f"{prefix}_{idx[i]}"] = degree[i]
    return degree_dict

degree_sci = generate_degree_dict(sci_df, "Sci")
degree_ai  = generate_degree_dict(ai_df,  "AI")
color_sci_str = "rgba(255,195,0,1)"   # yellow
color_ai_str  = "rgba(142,68,173,1)"  # purple

def parse_name(name):
    """
    Return a label string for each cluster.
    If the label has more than 3 words, split it into two lines.
    """
    idx = int(name.split("_")[-1])
    if "Sci" in name:
        words_list = top_words_sci[idx] if idx < len(top_words_sci) else []
    else:
        words_list = top_words_ai[idx] if idx < len(top_words_ai) else []
    if isinstance(words_list, list):
        joined_str = " ".join(words_list)
    else:
        joined_str = words_list
    split_words = joined_str.split()
    if len(split_words) >= 20:
        line1 = " ".join(split_words[:2])
        line2 = " ".join(split_words[2:])
        return f"{line1}\n{line2}"
    else:
        return joined_str

def parse_size_degree(name):
    """Use the 'degree' dictionary to size nodes."""
    if "Sci" in name:
        return degree_sci.get(name, 1)
    else:
        return degree_ai.get(name, 1)
edges_all = pd.read_csv("../results/instruction_embedding/edges/kdd_sci_ai_all.csv")
start = edges_all["start"].tolist()
end   = edges_all["end"].tolist()
weights = edges_all["weight"].tolist()

threshold = 3
edges_data = [(start[i], end[i], weights[i]) 
              for i in range(len(edges_all)) 
              if weights[i] > threshold]

start_filtered = [ed[0] for ed in edges_data]
end_filtered   = [ed[1] for ed in edges_data]

for node_name in set(start_filtered + end_filtered):
    size_val = np.sqrt(parse_size_degree(node_name))
    # label_text = " "
    if "Sci" in node_name:
        if size_val > 2:
            label_text = parse_name(node_name)
        else:
            label_text = " "
    else:
        if size_val > 2:
            label_text = parse_name(node_name)
        else:
            label_text = " "

    # Determine node color
    if "Sci" in node_name:
        node_color = color_sci_str
    else:
        node_color = color_ai_str
    B.add_node(
        node_name,
        label=label_text,
        size=2 * size_val,
        color=node_color,
        mass=0.8,
        font={"color": node_color}  # text color for the label
    )
net = Network(notebook=True, height="1000px", width="1800px")
net.from_nx(B)
for node in net.nodes:
    font_dict = node.get("font", {})
    old_color = font_dict.get("color", "#000000")

    font_dict["size"]        = 5
    font_dict["face"]        = "arial"
    font_dict["strokeWidth"] = 1
    font_dict["strokeColor"] = "rgba(153,153,153,0.4)"
    font_dict["color"]       = old_color

    node["font"]   = font_dict
light_grey = "rgba(200, 200, 200, 0.7)"
for (src, dst, w) in edges_data:
    net.add_edge(
        src,
        dst,
        width=2,
        color=light_grey
    )
net.force_atlas_2based(
    gravity       = -8,
    overlap       = 0.5,
    spring_length = 5,
    spring_strength = 0.05
)
net.inherit_edge_colors(False)
net.toggle_physics(False)
html = net.generate_html(name="./index.html", local=False, notebook=False)
with open("./kdd_labeled.html", "w") as file:
    file.write(html)

print("HTML file has been saved as 'index_kdd.html'")

label_start = [parse_name(id) for id in start]
label_end = [parse_name(id) for id in end]
edges_all['start_label'] = label_start
edges_all['end_label'] = label_end

edges_all.to_csv("./edges_all.csv",index=False)