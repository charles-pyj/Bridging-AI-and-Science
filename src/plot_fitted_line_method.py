import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import linregress
from adjustText import adjust_text
xlim = 1.01 * 0.8
ylim = 1.05 * 0.32
df = pd.read_csv("../results/tables/AI_cluster.csv")
with open("../results/cluster_labels/cluster_name_AI.json","r") as f:
    GPT_summary = json.load(f)
print(GPT_summary[279])
x = df['size_total'].values
y = df['sci_size_total'].values
ind = df['cluster_idx'].values
summary = df['GPT_summarization'].values
indices = [i for i in range(len(summary)) if summary[i] not in ["default","default"]]
assert len(indices) == len(summary)
x_new = [x[i] for i in indices]
y_new = [y[i] for i in indices]
summary = [summary[i] for i in indices]
plt.figure(figsize=(15, 10))
sns.regplot(x=x, y=y, data=df, fit_reg=False, scatter_kws={'color': '#8E44AD','s':75})
slope, intercept, r_value, p_value, std_err = linregress(x, y)
x_vals = np.linspace(min(y), max(x)*xlim, 100)
y_vals = intercept + slope * x_vals
t_val = 1.96
ci = t_val * std_err
y_vals_upper = y_vals + ci * x_vals
y_vals_lower = y_vals - ci * x_vals
lower_indices = [ind[i] for i in indices if y[i] < intercept + (slope-ci) * x[i]]
higher_indices = [ind[i] for i in indices if y[i] > intercept + (slope+ci) * x[i]]
print(f"Confidence: {ci}")
print(f"Slope: {slope}")
print(f"Std: {std_err}")
print(f"intercept: {intercept}")
plt.plot(x_vals, y_vals, color='black', label=f'Regression Line: y = {intercept:.2f} + {slope:.2f}x')
plt.fill_between(x_vals, y_vals_lower, y_vals_upper, color='grey', alpha=0.2, label='95% Confidence Interval')
texts = []
plt.xlabel('# of publications in each cluster',fontsize=20)
plt.ylabel('# of AI4Science publications in each cluster',fontsize=20)
for i in range(len(x)):
        if ((x[i] > 65 and y[i] > 50 and y[i] < max(y)*ylim) or (x[i] > 230 and y[i] < 50 and y[i] < max(y)*ylim) or x[i]>500) and "Computational" not in GPT_summary[ind[i]] and "Deep Learning Models" not in GPT_summary[ind[i]] and "Genomic" not in GPT_summary[ind[i]] and "Machine Learning" != GPT_summary[ind[i]]:
            text = df['GPT_summarization'][i]
            label_color = 'black'
            texts.append(plt.text(x[i]+3, y[i], text, fontsize=5, ha='left', color=label_color))
plt.ylim([min(y), max(y) * ylim])
plt.xlim([min(x), max(x) * xlim])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
adjust_text(texts,
            force_points=0.002,
            force_text=0.002, )
indices = [i for i in range(len(df)) if (df['size_total'][i] > max(x) * xlim) or (df['sci_size_total'][i] > max(y)*ylim)]
