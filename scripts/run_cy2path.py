import argparse
import os

import anndata as ad
import numpy as np
import torch
from cy2path import infer_cytopath_lineages


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run a full cy2path analysis.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input file containing the transition matrix.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Number of steps to simulate in the Markov chain.",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-5, help="Tolerance for steady-state convergence"
    )
    parser.add_argument(
        "--num_chains",
        type=int,
        default=1000,
        help="Number of Markov chains to simulate.",
    )
    parser.add_argument(
        "--num_lineages", type=int, default=2, help="Number of lineages to infer."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to run.",
    )
    parser.add_argument(
        "--run_number", type=int, default=0, help="Run number for reproducibility."
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        required=False,
        help="Path for saving the output AnnData object; if None (default), the output "
        "is not saved (useful for, e.g., time and memory requirement assessment).",
    )
    args = parser.parse_args()

    # Load the data
    adata = ad.read_h5ad(args.input)
    if "T_forward" not in adata.obsp:
        try:
            adata.obsp["T_forward"] = adata.uns["T_forward"].copy()
        except KeyError:
            raise ValueError(
                "The input file does not contain the transition matrix "
                "'T_forward' in `obsp` or 'uns'."
            )

    np.random.seed(args.run_number)
    torch.manual_seed(args.run_number)

    # Run cy2path
    infer_cytopath_lineages(
        adata,
        method="linkage",
        recalc_items=True,
        recalc_matrix=False,
        max_iter=args.max_iter,
        tol=args.tol,
        num_chains=args.num_chains,
        num_lineages=args.num_lineages,
        n_jobs=args.n_jobs,
    )
    print(
        f"Number of steps: {adata.uns['state_probability_sampling']['sampling_params']['convergence']}"
    )
    if args.save is not None:
        save_dir = os.path.dirname(args.save)
        os.makedirs(save_dir, exist_ok=True)
        file_name = args.save
        if not file_name.save.endswith(".h5ad"):
            if file_name.endswith("/"):
                file_name = os.path.join(file_name, "adata_cy2path.h5ad")
            else:
                file_name = args.save + ".h5ad"
        adata.write_h5ad(file_name, compression="gzip")


if __name__ == "__main__":
    main()
