import os

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import torch

np.random.seed(42)
torch.manual_seed(42)


def generate_adata(means, directions, edges, points_per_cluster, target_dim=100):
    rng = np.random.default_rng(42)
    labels = []
    data = []
    velocities = []

    # Generate main clusters
    for i, (mean, direction) in enumerate(zip(means, directions)):
        points = rng.multivariate_normal(
            mean=mean, cov=np.eye(2), size=points_per_cluster
        )
        data.append(points)
        labels.extend([f"{i}"] * points_per_cluster)
        velocities.append(
            rng.multivariate_normal(
                mean=direction, cov=0.01 * np.eye(2), size=points_per_cluster
            ),
        )

    # Interpolate data points between main clusters
    for idx_1, idx_2 in edges:
        points = rng.multivariate_normal(
            mean=means[idx_1], cov=np.eye(2), size=points_per_cluster
        )
        points += (np.array(means[idx_2]) - np.array(means[idx_1])) * rng.random(
            points_per_cluster
        )[:, None]
        data.append(points)
        labels.extend([f"{idx_1}>{idx_2}"] * points_per_cluster)

        if means[idx_2][1] > means[idx_1][1]:
            dir_y = 1
        elif means[idx_2][1] == means[idx_1][1]:
            dir_y = 0
        else:
            dir_y = -1

        if means[idx_2][0] > means[idx_1][0]:
            dir_x = 1
        elif means[idx_2][0] == means[idx_1][0]:
            dir_x = 0
        else:
            dir_x = -1
        direction = np.asarray([dir_x, dir_y])
        velocities.append(
            rng.multivariate_normal(
                mean=direction, cov=0.01 * np.eye(2), size=points_per_cluster
            ),
        )

    data = np.concatenate(data)
    velocities = np.concatenate(velocities)

    # Project into higher dimension
    projection_matrix = rng.standard_normal(size=(2, target_dim))
    points_highd = data @ projection_matrix
    points_highd += 10 * rng.multivariate_normal(
        mean=np.zeros(target_dim), cov=np.eye(target_dim), size=len(data)
    )
    velocities_highd = velocities @ projection_matrix

    # Create AnnData object
    adata = sc.AnnData(
        X=points_highd,
        obs=pd.DataFrame({"cluster": labels}),
        obsm={"X_orig": data, "X_velocity_orig": velocities},
        layers={"spliced": points_highd, "velocity": velocities_highd},
    )
    return adata


def main():
    output_path = "../data/simulations/changing_size/"
    os.makedirs(output_path, exist_ok=True)
    root = np.array([0, 0])
    branch_points = [np.array([15, 0]), np.array([25, 10])]
    ends = [np.array([25, 32.5]), np.array([47.5, 10]), np.array([25, -10])]
    means = [
        root,
        0.5 * (branch_points[0] + root),
        branch_points[0],
        0.5 * (branch_points[0] + branch_points[1]),
        branch_points[1],
        branch_points[1] + 1 / 3 * (ends[0] - branch_points[1]),
        branch_points[1] + 2 / 3 * (ends[0] - branch_points[1]),
        ends[0],
        branch_points[1] + 1 / 3 * (ends[1] - branch_points[1]),
        branch_points[1] + 2 / 3 * (ends[1] - branch_points[1]),
        ends[1],
        0.5 * (branch_points[0] + ends[2]),
        ends[2],
    ]
    directions = [
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 1],
        [1, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, -1],
        [1, -1],
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (4, 8),
        (8, 9),
        (9, 10),
        (2, 11),
        (11, 12),
    ]
    for points_per_cluster in [20, 40, 80, 200, 400, 800]:
        adata = generate_adata(means, directions, edges, points_per_cluster)

        # Compute neighbors, UMAP, velocity embedding
        sc.pp.pca(adata)
        sc.pp.neighbors(adata, n_neighbors=30)
        sc.tl.umap(adata)
        scv.tl.velocity_graph(adata, mode_neighbors="connectivities")

        # Compute terminal states and transition matrix
        scv.tl.terminal_states(adata)
        adata.obsp["T_forward"] = scv.utils.get_transition_matrix(
            adata, self_transitions=False
        )
        adata.write_h5ad(os.path.join(output_path, f"adata_{adata.shape[0]}.h5ad"))


if __name__ == "__main__":
    main()
