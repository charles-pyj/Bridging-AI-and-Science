import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from adjustText import adjust_text
from scipy.stats import linregress
df = pd.read_csv("../results/tables/Science_Clusters.csv")
with open("../results/cluster_labels/cluster_name_Sci.json","r") as f:
    GPT_summary = json.load(f)
x = df['size_total'].values
y = df['AI_size_total'].values
ind = df['cluster_idx'].values
plt.figure(figsize=(15, 10))
sns.regplot(x=x, y=y, data=df, fit_reg=False, scatter_kws={'color': '#FFC300','s':75})
slope, intercept, r_value, p_value, std_err = linregress(x, y)
ylim = 1.05*0.52
xlim = 1.05
x_vals = np.linspace(min(x), max(x)*xlim, 100)
y_vals = intercept + slope * x_vals
t_val = 1.96
ci = t_val * std_err
print(f"Stats: ",slope, intercept,std_err)
y_vals_upper = y_vals + ci * x_vals
y_vals_lower = y_vals - ci * x_vals
lower_indices = [ind[i] for i in range(len(y)) if y[i] < intercept + (slope-ci) * x[i]]
higher_indices = [ind[i] for i in range(len(y)) if y[i] > intercept + (slope+ci) * x[i]]
print(f"Confidence: {ci}")
print(f"Slope: {slope}")
plt.plot(x_vals, y_vals, color='black', label=f'Regression Line: y = {intercept:.2f} + {slope:.2f}x')
plt.fill_between(x_vals, y_vals_lower, y_vals_upper, color='grey', alpha=0.2, label='95% Confidence Interval')
slope, intercept, r_value, p_value, std_err = linregress(x, y)
texts = []
print(len(x))
print(len(ind))
plt.xlabel('# of publications in each cluster',fontsize=20)
plt.ylabel('# of AI4Science publications in each cluster',fontsize=20)
for i in range(len(x)):
    ratio = 3
    if ((x[i] > 200 and y[i] > 50 and y[i] < max(y) * ylim) or (x[i] > 670 and y[i] < 50 and y[i] < max(y) * ylim)): 
        text = df['GPT_summarization'][i]
        label_color = 'black'
        texts.append(plt.text(x[i]+3, y[i], text, fontsize=5, ha='left', color=label_color))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim([min(y), max(y)* ylim  ])
plt.xlim([min(x) , max(x) * xlim ])
indices = [i for i in range(len(df)) if (df['size_total'][i] > max(x) * xlim) or (df['AI_size_total'][i] > max(y)*ylim)]

print(len(indices))
for i in indices:
    print(df.iloc[i]['GPT_summarization'])
    print(df.iloc[i]['size_total'],df.iloc[i]['AI_size_total'])
adjust_text(texts,
            force_points=0.005,
            force_text=0.005, )