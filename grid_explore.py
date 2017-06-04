import re

import numpy as np
import pandas as pd
import seaborn as sns


class GridSearchExplorer:
    """Allows explorating of GridSearchCV results with plots."""

    def __init__(self, grid):
        """Grid instantiated and fit to data.

        Note about order of results used for reshaping:
        Grid params are ordered alphabetically by str name,
            then iterated through with itertools.product
            for all combinations.
        The order of iteration determines the order of the flattened
            arrays in grid.cv_results_
        Therefore, best practice is to use same ordering when reshaping.
        """

        self.grid = grid
        self.param_grid = grid.param_grid
        self.param_names = sorted(self.param_grid)

    def _has_sklearn_base(self, obj):
        return any(str(x).startswith("<class 'sklearn")
                   for x in obj.__class__.__mro__)

    def _param_to_label(self, obj):
        if self._has_sklearn_base(obj):
            pattern = r"[.]([^'.]+)'"
            label = re.search(pattern, str(obj.__class__)).group(1)
        else:
            label = obj

        return label

    def _param_grid_to_labels(self, param_vals):
        """df"""

        labels = []

        for param_group in param_vals:
            group_lbls = []
            i = 0  # dupe label counter
            for val in param_group:
                label = self._param_to_label(val)
                if label in group_lbls:
                    i += 1
                    label = label + '_' + str(i)
                group_lbls.append(label)

            labels.append(group_lbls)

        return labels

    def plot(self, x, hue, metric='mean_test_score', ax=None, **kwargs):

        if x==hue:
            raise ValueError('x and hue must be different parameter names.')

        param_keys, param_vals = zip(*sorted(self.param_grid.items()))
        param_grid_labels = self._param_grid_to_labels(param_vals)

        index = pd.MultiIndex.from_product(param_grid_labels,
                                           names=param_keys)

        scores = self.grid.cv_results_[metric]
        data = pd.Series(scores, index=index,
                         name=metric).reset_index()
        fg = sns.factorplot(
            y=metric, x=x, hue=hue, data=data,
            estimator=np.max, ci=False, ax=ax, **kwargs)

        fg.ax.set_title('GridSearchCV results')

        return fg.ax
