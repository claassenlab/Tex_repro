import numpy as np
import pandas as pd

import rpy2.robjects as robjects
from rpy2.robjects.functions import SignatureTranslatedFunction

import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()

import rpy2.robjects.pandas2ri

rpy2.robjects.pandas2ri.activate()

from rpy2.robjects.packages import importr

base = importr("base")
slingshot = importr("slingshot")

slingshot.slingshot = SignatureTranslatedFunction(slingshot.slingshot,
                                                  init_prm_translate={"start_clus": "start.clus",
                                                                      "end_clus": "end.clus"})


# Function to run slingshot on preprocessed data
def run_slingshot(adata, terminal_states, root_states, cluster_key="cluster_key", embedding="X_umap"):
    sling_out = slingshot.slingshot(adata.obsm[embedding],
                                    clusterLabels=adata.obs[cluster_key].astype(str).values,
                                    # Clustering provided should always have the start cluster annotated as root
                                    # and end clusters annotated by cell type identity
                                    start_clus=root_states, end_clus=terminal_states)

    # Extract inferred pseudotime, assignment score and curves
    sling_pst = np.array(slingshot.slingPseudotime(sling_out))
    sling_weights = np.array(slingshot.slingCurveWeights(sling_out))
    sling_lineages = np.array(slingshot.slingLineages(sling_out))
    sling_curves = []
    for i in range(np.array(slingshot.slingCurves(sling_out)).shape[0]):
        sling_curves.append(np.array(np.array(slingshot.slingCurves(sling_out))[i][0])[
                                np.array(np.array(slingshot.slingCurves(sling_out))[i][1]) - 1])

    # Generate slingshot data in common format used for comparative analyses
    slingshot_pst_data = pd.DataFrame(sling_pst, columns=[sling_lineages[i][-1] for i in range(sling_pst.shape[1])])
    slingshot_pst_data.index.name = "Cell"

    slingshot_alignment_data = pd.DataFrame(sling_weights,
                                            columns=[sling_lineages[i][-1] for i in range(sling_pst.shape[1])])
    slingshot_alignment_data.index.name = "Cell"

    return slingshot_pst_data, slingshot_alignment_data, sling_curves
