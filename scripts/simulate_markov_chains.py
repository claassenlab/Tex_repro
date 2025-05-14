import argparse
import json
import os
from datetime import datetime

import anndata as ad
import numpy as np
import torch
from cy2path import sample_markov_chains
from tqdm.auto import tqdm


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Simulate Markov chains.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input file containing the transition matrix.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"output/markov_chains/{datetime.now().strftime('%Y%m%d_%H%M')}/",
        help="Output folder to save the simulated Markov chains.",
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
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to run.",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

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

    # Simulate and cluster Markov chains num_simulations times
    for i in tqdm(
        range(args.num_simulations),
        total=args.num_simulations,
        desc="Performing simulations ",
        unit="simulation(s)",
    ):
        np.random.seed(i)
        torch.manual_seed(i)

        # Simulate Markov chains and save to file
        adata_ = sample_markov_chains(
            adata,
            recalc_matrix=False,
            self_transitions=False,
            num_chains=args.num_chains,
            max_iter=args.max_iter,
            convergence="auto",
            copy=True,
            n_jobs=args.n_jobs,
        )
        simulations = adata_.uns["markov_chain_sampling"]["state_indices_max_iter"]
        np.save(os.path.join(args.output, f"simulations_{i}.npy"), simulations)
        json.dump(
            adata_.uns["markov_chain_sampling"]["sampling_params"],
            open(os.path.join(args.output, f"params_{i}.json"), "w"),
        )


if __name__ == "__main__":
    main()
