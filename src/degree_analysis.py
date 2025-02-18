import json
import pandas as pd
import numpy as np

def get_degree(prefix,indices,degree_dict):
    degrees = []
    for i in indices:
        if f"{prefix}_{i}" in degree_dict:
            degrees.append(degree_dict[f"{prefix}_{i}"])

    print(f"Mean degree: {np.mean(degrees)} of {len(degrees)} clusters")

with open("../../results/cluster_labels/cluster_labels_scientific_problem.json","r") as f:
    cluster_labels_sci = json.load(f)
with open("../../results/cluster_labels/cluster_labels_AI_method.json","r") as f:
    cluster_labels_ai = json.load(f)
with open("../../results/cluster_labels/under_explored_Sci_clusters.json","r") as f:
    under_sci_indices = json.load(f)
with open("../../results/cluster_labels/well_explored_Sci_clusters.json","r") as f:
    over_sci_indices = json.load(f)
with open("../../results/cluster_labels/under_explored_AI_clusters.json","r") as f:
    under_ai_indices = json.load(f)
with open("../../results/cluster_labels/well_explored_AI_clusters.json","r") as f:
    over_ai_indices = json.load(f)
def power_law_fit(
        degrees=None,
        title='',
        xlabel='x',
        ylabel='y',
        savepath = None
    ):
    import powerlaw
    import matplotlib.pyplot as plt
    fit = powerlaw.Fit(degrees, xmin=1)    
    print(f"Alpha (exponent): {fit.alpha}")
    print(f"Xmin (min value for fit): {fit.xmin}")
    print(f"KS Statistic: {fit.power_law.KS()}")
    R, p_value = fit.distribution_compare('power_law', 'exponential')
    print(f"Log-Likelihood Ratio R: {R}")
    print(f"p-value of Log-Likelihood Ratio Test: {p_value}")
    R_log, p_log = fit.distribution_compare('power_law', 'lognormal')
    print(f"Log-Likelihood Ratio R (Log-Normal): {R_log}")
    print(f"p-value of Log-Likelihood Ratio Test (Log-Normal): {p_log}")
    fig, ax = plt.subplots(figsize=(6, 4))
    alpha = fit.alpha
    xmin = fit.xmin
    C = (alpha - 1) / (xmin ** (alpha - 1))
    custom_label = f'C={C:.2f}, alpha={alpha:.2f}, xmin={xmin}'
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    pdf = counts / sum(counts)
    ax.scatter(unique_degrees, pdf, color='b', label=custom_label, s=20, alpha=0.6)
    fit.power_law.plot_pdf(color='r', linestyle='--', ax=ax, label=f'Power Law fit\nalpha={fit.alpha:.2f}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if savepath:
        plt.savefig(savepath)


def plot_degrees_log(degrees, title, savepath, kde_color='red'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    data = degrees
    log_data = np.log10(data)
    plt.figure(figsize=(10, 6))
    log_data_df = pd.DataFrame(log_data, columns=["Log Degree"])
    sns.histplot(log_data, bins=30, color='#5A9', edgecolor='black',kde=False, stat="density")
    sns.kdeplot(log_data, color='crimson',linestyle="--")
    plt.title(title, fontsize=20, fontweight='bold')
    plt.xlabel('Log(Degree)', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='-', linewidth=0.3, color='gray')
    plt.savefig(savepath,format="pdf",bbox_inches='tight',dpi=500)

def qq_plot_and_ks_test(
    data, 
    title='',
    marker_size=2,
    savepath = None
):
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(rc={'figure.figsize':(3,3)})
    log_data = np.log(data)
    _, ax = plt.subplots()
    stats.probplot(
        log_data, 
        dist="norm", 
        plot=ax
    )
    ax.set_title(title)
    ax.set_xlabel('Theoretical Quantiles',fontsize=15)
    ax.set_ylabel('Ordered novelty scores (log)',fontsize=15)
    ax.get_lines()[0].set_markersize(marker_size)
    kstest_result = stats.kstest(data, 'lognorm', stats.lognorm.fit(data))
    shape, loc, scale = stats.lognorm.fit(data)
    ax.text(0.05, 0.95, f"KS test p-value: {kstest_result.pvalue:.2f}", 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))
    plt.savefig(savepath,format="pdf",bbox_inches='tight')

# qq_plot_and_ks_test(degrees_dict_sci, title='Q-Q Plot and KS Test',savepath="../visualizations/QQ_ks_sci.pdf")
# qq_plot_and_ks_test(degrees_dict_ai, title='Q-Q Plot and KS Test',savepath="../visualizations/QQ_ks_ai.pdf")

def edges_to_degree(df,indices = None):
    def parse_name(name):
        return int(name.split("_")[-1])
    degree_dict = {}
    start = df['start'].tolist()
    end = df['end'].tolist()
    for i in np.unique(start):
        if indices == None:
            degree_dict[i] = len(df[df['start'] == i])
        else:
            if parse_name(i) in indices:
                degree_dict[i] = len(df[df['start'] == i])
    return degree_dict

sci_df = pd.read_csv("../../results/tables/Science_Clusters.csv")
degree = sci_df['degree']
ai_df = pd.read_csv("../../results/tables/AI_Clusters.csv")
AI_degree = ai_df['degree']
degree = [d for d in degree if d > 0]
AI_degree = [d for d in AI_degree if d > 0]

# qq_plot_and_ks_test(degree, title='Q-Q Plot and KS Test',savepath="../visualizations/QQ_ks_sci_kdd.pdf")
# qq_plot_and_ks_test(AI_degree, title='Q-Q Plot and KS Test',savepath="../visualizations/QQ_ks_ai_kdd.pdf")