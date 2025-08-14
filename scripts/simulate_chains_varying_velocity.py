import argparse
import os
import subprocess
import warnings
from datetime import datetime

import scanpy as sc
from scipy import sparse


def main():
    parser = argparse.ArgumentParser(
        "Simulate Markov chains for varying velocity parameters"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input folder containing all transition matrices.",
    )
    parser.add_argument(
        "--anndata", type=str, required=True, help="Input AnnData file."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"../output/varying_velocity_params/{datetime.now().strftime('%Y%m%d_%H%M')}/",
        help="Output folder to save the simulated Markov chains and clustering results.",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        nargs="+",
        default=[15, 30, 50, 100],
        help="Number of nearest neighbors",
    )
    parser.add_argument(
        "--velocity_mode",
        type=str,
        nargs="+",
        default=["deterministic", "stochastic", "dynamical"],
        choices=["deterministic", "stochastic", "dynamical"],
        help="Velocity modes to use",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Number of steps to simulate in the Markov chain.",
    )
    parser.add_argument(
        "--num_chains",
        type=int,
        default=1000,
        help="Number of Markov chains to simulate.",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=10,
        help="Number of Markov chain simulations to run.",
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

    # Create output directory if it does not exist
    os.makedirs(args.output, exist_ok=True)

    # Load the data
    adata = sc.read_h5ad(args.anndata)

    # Load the transition matrices, store them in the AnnData object,
    # and run exploratory trajectory inference
    for n_neighbors in args.n_neighbors:
        for velocity_mode in args.velocity_mode:
            try:
                transition_matrix = sparse.load_npz(
                    os.path.join(
                        args.input, f"{n_neighbors}_neighbors_{velocity_mode}.npz"
                    )
                )
            except FileNotFoundError:
                warnings.warn(
                    f"No transition matrix found in {args.input} for "
                    f"{n_neighbors} neighbors and {velocity_mode} velocity"
                )
                continue
            adata.obsp["T_forward"] = transition_matrix
            fname = os.path.join(
                args.output, f"adata_{n_neighbors}_neighbors_{velocity_mode}.h5ad"
            )
            adata.write_h5ad(fname, compression="gzip")

            # Run sampling
            markov_chains_dir = os.path.join(
                args.output, "markov_chains", f"{n_neighbors}_neighbors_{velocity_mode}"
            )
            subprocess.run(
                [
                    "python3",
                    "simulate_markov_chains.py",
                    "-i",
                    fname,
                    "-o",
                    markov_chains_dir,
                    "--max_iter",
                    str(args.max_iter),
                    "--num_chains",
                    str(args.num_chains),
                    "--num_simulations",
                    str(args.num_simulations),
                    "--n_jobs",
                    str(args.n_jobs),
                ]
            )

            # Run clustering
            clustering_dir = os.path.join(
                args.output,
                "lineage_clustering",
                f"{n_neighbors}_neighbors_{velocity_mode}",
            )
            subprocess.run(
                [
                    "python3",
                    "cluster_markov_chains.py",
                    "-i",
                    markov_chains_dir,
                    "--anndata",
                    fname,
                    "-o",
                    clustering_dir,
                    "--max_lineages",
                    str(args.max_lineages),
                    "--basis",
                    args.basis,
                    "--n_jobs",
                    str(args.n_jobs),
                    "--num_steps",
                ]
                + [str(num_steps) for num_steps in args.num_steps]
            )

            # Remove AnnData object again
            os.remove(fname)


if __name__ == "__main__":
    main()
