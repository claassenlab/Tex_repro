import argparse
import json
import os
from datetime import datetime
from glob import glob

import anndata as ad
import numpy as np
from cy2path import infer_cytopath_lineages, sample_state_probability
from dtaidistance import dtw_ndim
from sklearn.metrics import silhouette_score
from tqdm.auto import tqdm


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Cluster Markov chains.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input folder containing the Markov chains.",
    )
    parser.add_argument(
        "--anndata", type=str, required=True, help="Input AnnData file."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"output/lineage_clustering/{datetime.now().strftime('%Y%m%d_%H%M')}/",
        help="Output folder to save the clustering results.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=[0],
        nargs="+",
        help="Number of steps to use from the Markov chains.",
    )
    parser.add_argument(
        "--max_lineages",
        type=int,
        default=10,
        help="Maximum number of lineages to cluster.",
    )
    parser.add_argument(
        "--basis", type=str, default="pca", help="Basis for clustering."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to run.",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Load the data
    adata = ad.read_h5ad(args.anndata)
    if "T_forward" not in adata.obsp:
        try:
            adata.obsp["T_forward"] = adata.uns["T_forward"].copy()
        except KeyError:
            raise ValueError(
                "The input file does not contain the transition matrix "
                "'T_forward' in `obsp` or 'uns'."
            )

    # Run state probability sampling once
    sample_state_probability(
        adata,
        matrix_key="T_forward",
        recalc_matrix=False,
        self_transitions=False,
        init="root_cells",
        max_iter=1000,
        tol=1e-5,
        copy=False,
    )

    for file in tqdm(glob(f"{args.input}/simulations_*.npy"), desc="Processing runs "):
        run_number = file.split("/")[-1].split("_")[1].split(".")[0]
        simulation_params = json.load(open(f"{args.input}/params_{run_number}.json"))
        markov_chains = np.load(file)

        # Iterate over number of steps
        for num_steps in args.num_steps:
            # Create separate output directories if running with different number of steps
            if len(args.num_steps) > 1:
                output_dir = os.path.join(args.output, f"{num_steps}_steps")
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = args.output

            if num_steps == 0:
                num_steps = simulation_params["convergence"]
            markov_chains_ = markov_chains[:, :num_steps]
            adata.uns["markov_chain_sampling"] = {}
            adata.uns["markov_chain_sampling"]["sampling_params"] = {}
            adata.uns["markov_chain_sampling"]["sampling_params"]["num_chains"] = (
                markov_chains_.shape[0]
            )
            adata.uns["markov_chain_sampling"]["state_indices"] = markov_chains_

            # Get coordinates of simulations in the basis
            simulations = adata.obsm[f"X_{args.basis}"][markov_chains_].astype("double")
            distances = dtw_ndim.distance_matrix_fast(simulations)
            silhouette_scores = {}

            # Perform hierarchical clustering with different numbers of lineages
            for n_lineages in tqdm(
                range(2, args.max_lineages + 1),
                total=args.max_lineages - 1,
                desc="Clustering with different numbers of lineages",
            ):
                adata_ = infer_cytopath_lineages(
                    adata,
                    recalc_items=False,
                    recalc_matrix=False,
                    basis=args.basis,
                    num_lineages=n_lineages,
                    method="linkage",
                    distance_func="dtw",
                    differencing=False,
                    copy=True,
                    n_jobs=args.n_jobs,
                )
                silhouette_scores[n_lineages] = silhouette_score(
                    distances,
                    adata_.uns["cytopath"]["lineage_inference_clusters"],
                    metric="precomputed",
                )
                lineage_inference_clusters = adata_.uns["cytopath"][
                    "lineage_inference_clusters"
                ]
                np.save(
                    os.path.join(
                        output_dir,
                        f"lineage_inference_clusters_num_lineages_{n_lineages}_{run_number}.npy",
                    ),
                    lineage_inference_clusters,
                )
            json.dump(
                silhouette_scores,
                open(
                    os.path.join(output_dir, f"silhouette_scores_{run_number}.json"),
                    "w",
                ),
            )


if __name__ == "__main__":
    main()
