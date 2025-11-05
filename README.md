# Tex_repro
Jupyter notebooks for reproducing the results of [Schleicher et al. 2025](https://doi.org/10.1371/journal.pone.0332406).

The notebooks reproduce the preprocessing and annotation ([01_annotation.ipynb](notebooks/01_annotation.ipynb)), the trajectory analysis with Cytopath ([02_cytopath.ipynb](notebooks/02_cytopath.ipynb)), and the exploratory trajectory analysis ([04_cytopath_convergent.ipynb](notebooks/04_cytopath_convergent.ipynb)).
Furthermore, the figures in the paper are reproduced in [99_figures.ipynb](notebooks/99_figures.ipynb).

All AnnData objects used and created by the code in this repository, containing raw and processed expression data, are available on Zenodo at [10.5281/zenodo.10559456](https://doi.org/10.5281/zenodo.10559456).

## Citation

```
@article{10.1371/journal.pone.0332406,
    doi = {10.1371/journal.pone.0332406},
    author = {Schleicher, Jan T. AND Gupta, Revant AND Cerletti, Dario AND Sandu, Ioana AND Oxenius, Annette AND Claassen, Manfred},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {Exploratory trajectory inference reveals convergent lineages for CD8 T cells in chronic LCMV infection},
    year = {2025},
    month = {09},
    volume = {20},
    url = {https://doi.org/10.1371/journal.pone.0332406},
    pages = {1-25},
    number = {9},
}
```
