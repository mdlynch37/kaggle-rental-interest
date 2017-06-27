import re

import numpy as np
import pandas as pd
import seaborn as sns


class GridSearchExplorer:
    """Allows explorating of GridSearchCV results with plots.

    This object allows visualization with heatmaps and pointplots.
    When the GridSearchCV object has more than 2 dimensions of
    parameter sets, heatmaps and pointplots will have different
    interpretations.

    Pointplots support aggregating data into estimates, and will
    show a confidence interval by default in this case.
    Values across parameter sets not specified as x or hue will
    be estimated with numpy's mean function by default.

    Heatmaps do not support estimates, so parameter sets not
    specified for the x-axis will be put onto the y-axis as
    combinations taken from a pandas MultiIndex.
    """

    def __init__(self, grid):
        """Grid instantiated and fit to data."""

        self.grid = grid
        self.param_grid = grid.param_grid
        self.param_names = sorted(self.param_grid)  # for reference
        self.plot_data = None
        self.kind = None

    def _is_from_sklearn(self, obj):
        return any(str(x).startswith("<class 'sklearn")
                   for x in obj.__class__.__mro__)

    def _grid_val_to_label(self, obj):
        """Converts long sklearn object name to simple class name.

        For GridSearch(clf, params=dict(reducer=[PCA, TruncatedSVD])),
        if reducer is passed to self.plot as hue, legend will show
        - PCA
        - TruncatedSVD
        instead of
        - <class 'sklearn.decomposition.pca.PCA'>
        - <class 'sklearn.decomposition.truncated_svd.TruncatedSVD'>
        """
        if self._is_from_sklearn(obj):
            pattern = r"[.]([^'.]+)'"
            label = re.search(pattern, str(obj.__class__)).group(1)
        else:
            label = obj

        return label

    def _grid_vals_to_labels(self, param_vals):
        """Appends duplicates' labels with numbers 1, 2, etc."""

        labels = []

        for group in param_vals:
            group_lbls = []
            i = 0  # dupe label counter
            for val in group:
                label = self._grid_val_to_label(val)
                if label in group_lbls:
                    i += 1
                    label = str(label) + '_' + str(i)
                group_lbls.append(label)

            labels.append(group_lbls)

        return labels

    # TODO: Change grid-calculated mean data to split*_test_score
    # data points so that pointplot can include confidence intervals
    def plot(self, x, hue=None, metric='mean_test_score',
             kind='heatmap', estimator=np.mean, ax=None, **kwargs):
        """Plots GridSearchCV results onto heatmap or pointplot.

        Parameters
        ----------
        x : str
            Set of params on the x-axis of heatmap or pointplot.
            For heatmap, the other param sets will be plotted
            on the y-axis. If more than one, then multiindex-style
            combinations are shown on y-axis.
        hue : str, optional
            Set of params plotted as lines in pointplot. If None,
            mean of scores for x across other params is in single line.
        metric : str, optional
            Key to metrics of each param combination in grid.cv_results
            Examples:
            - 'mean_test_score' over validation folds
            - 'mean_train_score'
            - 'mean_fit_time'
        kind : {``heatmap``, ``point``}
            The kind of plot to draw.
        estimator : callable that maps vector -> scalar for pointplot
            Calculates estimates for pointplot. Applicable when
            GridSearchCV has more than 2 dimensions of param sets.
        ax : matplotlib Axes, optional
            Axes object to draw the plot onto, otherwise uses the
            current Axes.

        kwargs : key, value pairings
            Other keyword arguments passed to seaborn pointplot

        Returns
        -------
        ax : matplotlib Axes
            Returns the Axes object with the boxplot drawn onto it.
        """
        if x == hue:
            raise ValueError('x and hue must be different parameter names.')
        self.kind = kind

        # Note about order of results used for reshaping:
        # Grid params are ordered alphabetically by str name,
        #     then iterated through with itertools.product
        #     for all combinations.
        # The order of iteration determines the order of the flattened
        #     arrays in grid.cv_results_
        # Therefore, best practice is to use same ordering when reshaping.
        param_keys, param_vals = zip(*sorted(self.param_grid.items()))
        param_grid_labels = self._grid_vals_to_labels(param_vals)

        index = pd.MultiIndex.from_product(param_grid_labels, names=param_keys)

        scores = self.grid.cv_results_[metric]

        if self.kind == 'heatmap':
            if hue is not None:
                raise ValueError('hue not supported by heatmap.')
            # Set up DataFrame into grid format
            self.plot_data = (pd.Series(scores, index=index, name=metric)
                              .unstack(x)
            )
            ax = sns.heatmap(data=self.plot_data, ax=ax, **kwargs)

        elif self.kind == 'point':
            # Convert to long-form for plot
            self.plot_data = (pd.Series(scores, index=index, name=metric)
                              .reset_index()
            )
            ax = sns.pointplot(
                x=x, y=metric, hue=hue, data=self.plot_data, estimator=np.mean,
                ax=ax, **kwargs)

            # If GridSearchCV has more than two parameter sets, remove
            # mean() enclosure from ylabel.
            # In this case, there would not be more than one value to
            # aggregate.
            if self.plot_data.shape[1] <= 3:
                ax.set_ylabel(metric)

        else:
            raise ValueError('{} not a supported plot kind.'.format(kind))

        ax.set_title('GridSearchCV results')

        return ax
