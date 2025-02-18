## This folder contains the AI4Science Dataset proposed in our work



## Directory Structure

- **./**: Contains human annotation source files for scientific journal papers and AI conference papers as well annotation results of 2 human annotators.

| File Name                             | Description                                                  |
| ------------------------------------  | ------------------------------------------------------------ |
| AI4Science_Dataset.csv | Contains paper metadata (title, abstract, mesh terms, venue, etc.), their extraction results, whether they are classified as AI4Science works and the clusters they belong to. |
| Science_clusters.csv | Contains scientific problem clusters information that include cluster index, cluster label, cluster top words, total paper size, total AI4Science paper size and number of AI4Science contributions from either Science/AI communities |
| AI_clusters.csv    | Contains AI method clusters information that include cluster index, cluster label, cluster top words, total paper size, total AI4Science paper size and number of AI4Science contributions from either Science/AI communities |

The extracted information includes: scientific problem keywords, scientific problem definition, scientific problem discipline, AI method keywords, AI method definition and AI method usage.

- **./2D_Projection**: Contains paper meta data files for both projection maps, their corresponding extractions and high-resolution images for 2D projection visualizations.
- **./Bipartite_Linked_Graph**: Contains all edges used to produce the bipartite linked graph and an interactive figure visualizing the interactive figure.



