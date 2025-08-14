import argparse
import os
from datetime import datetime

import scanpy as sc
import scvelo as scv
from scipy import sparse


def main():
    parser = argparse.ArgumentParser(
        prog="Perform RNA velocity analysis with varying parameters and "
        "store the transition matrices."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input AnnData file containing spliced and unspliced matrices.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"../output/transition_matrices/{datetime.now().strftime('%Y%m%d_%H%M')}",
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

    args = parser.parse_args()

    adata = sc.read_h5ad(args.input)

    os.makedirs(args.output, exist_ok=True)

    for n_neighbors in args.n_neighbors:
        # Smooth over k neighbors
        adata_ = scv.pp.neighbors(adata, n_neighbors=n_neighbors, copy=True)
        scv.pp.moments(adata_, n_neighbors=n_neighbors)
        scv.tl.recover_dynamics(adata_, n_jobs=-1)

        for velocity_mode in args.velocity_mode:
            # Estimate RNA velocity
            scv.tl.velocity(adata_, mode=velocity_mode)
            scv.tl.velocity_graph(
                adata_, n_neighbors=n_neighbors, mode_neighbors="connectivities"
            )

            # Terminal states and transition matrix
            scv.tl.terminal_states(adata_)
            transition_matrix = scv.utils.get_transition_matrix(
                adata_, self_transitions=False
            )
            sparse.save_npz(
                os.path.join(args.output, f"{n_neighbors}_neighbors_{velocity_mode}"),
                transition_matrix,
            )


if __name__ == "__main__":
    main()
