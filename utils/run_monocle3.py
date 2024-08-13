import os
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.functions import SignatureTranslatedFunction

import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()

import rpy2.robjects.pandas2ri

rpy2.robjects.pandas2ri.activate()

from rpy2.robjects.packages import importr

base = importr("base")

seurat = importr("Seurat")
sceasy = importr("sceasy")
seuratWrappers = importr("SeuratWrappers")
monocle3 = importr("monocle3")

sceasy.convertFormat = SignatureTranslatedFunction(sceasy.convertFormat,
                                                   init_prm_translate={"from_format": "from"})


# Function to run monocle on preprocessed data
def run_monocle3(adata, root_clusters="root", terminal_states=None, cluster_key="cluster_key",
                 temp_file_name="run_monocle3_temp.h5ad"):
    # Special stuff to make this work
    adata_copy = adata.copy()
    del adata_copy.raw
    # del adata_copy.var
    # del adata_copy.obs
    del adata_copy.uns
    del adata_copy.layers

    adata_copy.write(temp_file_name)

    # Convert anndata object to seurat obj and then to CSD
    seuratObj = sceasy.convertFormat(temp_file_name, from_format="anndata", to="seurat")
    monocleObj = seuratWrappers.as_cell_data_set(seuratObj)

    # Run monocle3 (non DDRTree)
    monocleObj = monocle3.cluster_cells(monocleObj, reduction_method="UMAP")
    monocleObj = monocle3.learn_graph(monocleObj, use_partition=True)
    monocleObj = monocle3.order_cells(monocleObj, reduction_method="UMAP",
                                      root_cells=adata.obs.loc[adata.obs[cluster_key] == root_clusters].index.values)

    # Extract graph
    principal_graph = monocleObj.slots["principal_graph"]
    monocle_graph = principal_graph.slots["listData"].rx2["UMAP"]

    # Extract graph coordinates
    principal_graph_aux = monocleObj.slots["principal_graph_aux"]
    monocle_graph_coordinates = principal_graph_aux.slots["listData"].rx2["UMAP"].rx2["dp_mst"]

    monocle_graph_coordinates = pd.DataFrame(monocle_graph_coordinates.T, columns=["coordinate_1", "coordinate_2"])
    monocle_graph_coordinates.index.name = 'Node'

    # Extract monocle pseudotime
    monocle_pst = monocle3.pseudotime(monocleObj)
    monocle_pst = pd.DataFrame(monocle_pst, columns=["Pseudotime"])
    monocle_pst.index.name = 'Cell'

    os.remove(temp_file_name)

    return monocle_pst, monocle_graph_coordinates, monocle_graph
