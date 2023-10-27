from matplotlib import pyplot as plt
from typing import Tuple, List, Dict, Union
from dataclasses import dataclass
from matplotlib.gridspec import SubplotSpec, GridSpec


@dataclass
class Grid:
    """
    Class for specifying subgrids that will be recursively assembled within outer grids
    """
    nrows: int
    ncols: int
    row_heights: List[int]
    col_widths: List[int]
    sub_grids: List[Dict[str, Union[int, List[int], dict]]] = None
    wspace: int = None
    hspace: int = None

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def make_grid(figsize: Tuple[int, int], grid_spec: Dict[str, Union[int, List[int], dict]]):
    """
    Generate a grid for adding plots
    :param figsize: size of figure in format (width, height)
    :param grid_spec: dictionary specifying grids in nested fashion
    :return: fig, axs
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    outer_grid = Grid.from_dict(grid_spec)
    # make outer grid
    gs = fig.add_gridspec(nrows=outer_grid.nrows, ncols=outer_grid.ncols,
                          height_ratios=outer_grid.row_heights, width_ratios=outer_grid.col_widths,
                          wspace=outer_grid.wspace, hspace=outer_grid.hspace)
    _add_subplots(fig, gs, outer_grid)

    return fig, fig.get_axes()


def _make_subgrid(fig: plt.Figure, subplot_spec: SubplotSpec, grid_spec: Dict[str, Union[int, List[int], dict]]):
    """
    Generate a subgrid, which could have further subgrids
    :param fig: figure to which grid is added
    :param subplot_spec: subgrid object to which another grid is added
    :param grid_spec: dictionary specifying subgrids in nested fashion
    :return:
    """
    subgrid = Grid.from_dict(grid_spec)
    gs = subplot_spec.subgridspec(nrows=subgrid.nrows, ncols=subgrid.ncols,
                                  height_ratios=subgrid.row_heights, width_ratios=subgrid.col_widths,
                                  wspace=subgrid.wspace, hspace=subgrid.hspace)
    _add_subplots(fig, gs, subgrid)


def _add_subplots(fig: plt.Figure, gs: GridSpec, grid: Grid):
    """
    Add subplots and subgrids to a GridSpec
    :param fig: figure to which plots are added
    :param gs: GridSpec object to which plots/subplots are added
    :param grid: current grid
    :return:
    """
    # no further subgrids: add plots
    if grid.sub_grids is None:
        if grid.nrows > 1 and grid.ncols > 1:
            for i in range(grid.nrows):
                for j in range(grid.ncols):
                    fig.add_subplot(gs[i, j])
        elif grid.ncols > 1:
            for i in range(grid.ncols):
                fig.add_subplot(gs[i])
        elif grid.nrows > 1:
            for i in range(grid.nrows):
                fig.add_subplot(gs[i])
    # add further subgrids
    else:
        for i, sub in enumerate(gs):
            if grid.sub_grids[i] is not None:
                _make_subgrid(fig, sub, grid.sub_grids[i])
            else:
                fig.add_subplot(gs[i])
