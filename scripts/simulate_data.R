# Load required libraries
library(argparse)
library(tidyverse)
library(dyngen)
library(anndata)
library(reticulate)
library(parallelly)

# use_condaenv("scanpy")
use_condaenv("cytopath")

set.seed(42)

# Set up argument parser
parser <- ArgumentParser()
parser$add_argument(
  "-n",
  "--num_cells",
  type = "integer",
  default = 5000,
  help = "Number of cells to be simulated"
)
parser$add_argument(
  "-b", "--backbone",
  type = "character",
  default = "linear",
  choices = c(
    "linear",
    "bifurcating",
    "bifurcating_converging",
    "bifurcating_cycle",
    "bifurcating_loop",
    "binary_tree",
    "consecutive_bifurcating",
    "trifurcating",
    "converging",
    "cycle",
    "disconnected"
  ),
  help = "Dynamic process topology for dyngen"
)
parser$add_argument(
  "--num_genes",
  type = "integer",
  default = 500,
  help = "Number of genes"
)
parser$add_argument(
  "-o", "--out_dir",
  type = "character",
  default = "sim_data",
  help = "Output directory"
)

args <- parser$parse_args()

# Create output directory if it doesn't exist
dir.create(args$out_dir, showWarnings = FALSE, recursive = TRUE)

# Set up basic model
backbone <- switch(args$backbone,
  linear = backbone_linear(),
  bifurcating = backbone_bifurcating(),
  bifurcating_converging = backbone_bifurcating_converging(),
  bifurcating_cycle = backbone_bifurcating_cycle(),
  bifurcating_loop = backbone_bifurcating_loop(),
  binary_tree = backbone_binary_tree(num_modifications = 2),
  consecutive_bifurcating = backbone_consecutive_bifurcating(),
  trifurcating = backbone_trifurcating(),
  converging = backbone_converging(),
  cycle = backbone_cycle(),
  disconnected = backbone_disconnected()
)

num_cells <- args$num_cells
num_tfs <- nrow(backbone$module_info)
num_targets <- round((args$num_genes - num_tfs) / 2)
num_hks <- args$num_genes - num_targets - num_tfs

out <-
  initialise_model(
    backbone = backbone,
    num_cells = args$num_cells,
    num_tfs = num_tfs,
    num_targets = num_targets,
    num_hks = num_hks,
    simulation_params = simulation_default(
      total_time = 1000,
      census_interval = 2,
      ssa_algorithm = ssa_etl(tau = 300 / 3600),
      experiment_params = simulation_type_wild_type(
        num_simulations = num_cells / 10
      ),
      compute_rna_velocity = TRUE
    ),
    experiment_params = experiment_snapshot(),
    num_cores = availableCores()
  ) %>%
  generate_dataset()

adata <- as_anndata(out$model)
adata$write_h5ad(
  file.path(
    args$out_dir,
    paste(
      "dyngen_",
      args$backbone,
      "_",
      args$num_cells,
      "_cells_",
      args$num_genes,
      "_genes.h5ad",
      sep = ""
    )
  )
)
