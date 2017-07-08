
import re

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve


def best_xgb_cv_score(eval_hist):
    """Return details of boosting round with the highest CV score.

    Parameters
    ----------
    eval_hist : DataFrame
        Returned from xgboost.cv function.

    Returns
    -------
    results : str, formatted
    """
    is_lowest  = (eval_hist['test-mlogloss-mean'] ==
                  eval_hist['test-mlogloss-mean'].min())

    # If tie between two, take first.
    best_data   = eval_hist.loc[is_lowest].iloc[0]

    best_round  = best_data.name
    best_score  = best_data.at['test-mlogloss-mean']
    best_std    = best_data.at['test-mlogloss-std']
    train_score = best_data.at['train-mlogloss-mean']
    train_std   = best_data.at['train-mlogloss-std']

    cv_title    = 'Best CV score (round {}): '.format(best_round)
    results     = ('{:<28}{:.4f} ± {:.4f} (mean ± std. dev.)\n'
                   .format(cv_title, best_score, best_std))

    train_title = 'Train score (round {}): '.format(best_round)
    results    += ('{:<28}{:.4f} ± {:.4f} (mean ± std. dev.)\n'
                   .format(train_title, train_score, train_std))

    return results


def best_grid_score(grid):
    """Returns details of estimator with highest CV score.

    Parameters
    ----------
    grid : fitted GridSearchCV

    Returns
    -------
    results : str, formatted
    """
    best_score = grid.best_score_

    res = pd.DataFrame(grid.cv_results_)
    best_res    = res.loc[res.mean_test_score == best_score].iloc[0]

    best_std    = best_res['std_test_score']
    train_score = best_res['mean_train_score']
    train_std   = best_res['std_train_score']

    results = []

    cv_title    = 'Best grid CV score: '
    results.append('{:<20}{:.4f} ± {:.4f} (mean ± std. dev.)'
                   .format(cv_title, best_score, best_std))

    train_title = 'Train score: '
    results.append('{:<20}{:.4f} ± {:.4f} (mean ± std. dev.)'
                   .format(train_title, train_score, train_std))

    results.append('\n*** For parameters: ***')

    # Sorted for deterministic output
    for param, val in sorted(grid.best_params_.items()):
        results.append('{}={}'.format(param, val))
    return '\n'.join(results)


# Needs testing, not needed anymore
def filter_grid_results(cv_results, param_dict, any_match=False):
    bool_data = []
    for key, val in param_dict.items():
        # Converted to frame for edge-case of one parameter.
        # Would not work with pd.concat in this case.
        col_name = 'param_' + key
        bool_col = cv_results[col_name].apply(lambda x: np.isclose(x, val))
        bool_data.append(bool_col.to_frame())
    match_df = pd.concat(bool_data, axis=1)

    if any_match is False:
        is_match = match_df.all(1)
    else:
        is_match = match_df.any(1)

    return cv_results[is_match]


def plot_xgb_boosting_curve(eval_hist, ax=None, figsize=None):
    """Plot of loss at each boosting round of XGBoost estimator.

    Parameters
    ----------
    eval_hist : DataFrame
        Returned from XGBoost cross-validation function (xgb.cv).

    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the
        current Axes.

    figsize : tuple, optional
        Only applicable if ax is passed.

    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the boosting curve drawn onto it.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.set_title('XGBoost CV Boosting Curve')
    ax.set_xlabel('boost_round')
    ax.set_ylabel('mlogloss-mean')

    train_scores_mean = eval_hist['train-mlogloss-mean']
    test_scores_mean  = eval_hist['test-mlogloss-mean']

    train_scores_std  = eval_hist['train-mlogloss-std']
    test_scores_std   = eval_hist['test-mlogloss-std']

    x = list(eval_hist.index)
    ax.plot(x, train_scores_mean, '-', color='r',
            label='Training loss',
    )
    ax.plot(x, test_scores_mean, '-', color='g',
            label='Cross-validation loss',
    )
    ax.fill_between(x, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std,
                    alpha=0.1, color='r'
    )
    ax.fill_between(x, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std,
                    alpha=0.1, color='g'
    )
    ax.legend()

    return ax


def plot_learning_curve(estimator, X, y, estimator_name=None,
                        train_color='r', cv_color='g',
                        ax=None, figsize=None,
                        scoring=None, n_jobs=1,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Adapted from:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    Parameters
    ----------
    estimator : object type that implements the 'fit' and 'predict' methods
        An object of that type which is cloned for each validation.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    estimator_name : str, optional
        If passed, use this name as part of title, otherwise use __name__
        attribute from estimator.
        Used, for example, when the estimator is a Pipeline and the
        final estimator should be in the title.

    {train_color, cv_color} : str, default 'r' and 'g' resp.
        Any matplotlib color.

    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the
        current Axes.

    figsize : tuple, optional
        Only applicable if ax is passed.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Note: Length of array will be number of folds.
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the learning curve drawn onto
        it.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlabel('Training examples')
    ax.set_ylabel('Score')

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, scoring=scoring, n_jobs=n_jobs,
        train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1, color=train_color
    )
    ax.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1, color=cv_color
    )
    ax.plot(train_sizes, train_scores_mean, 'o-', color=train_color,
            label='Training score')
    ax.plot(train_sizes, test_scores_mean, 'o-', color=cv_color,
            label='Cross-validation score')

    ax.legend(loc='best')

    if estimator_name is None:
        estimator_name = type(estimator).__name__
    ax.set_title('Learning Curve for {}'.format(estimator_name))

    return ax


def plot_validation_curve(estimator, X, y, param_name, param_range,
                          estimator_name=None, ax=None, figsize=None,
                          cv=None, scoring=None, n_jobs=1):
    """Plots vaildation curve for given hyperparamters.

    Adapted from:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    param_name : string
        Name of the parameter that will be varied.

    param_range : array-like, shape (n_values,)
        The values of the parameter that will be evaluated.

    estimator_name : str, optional
        If passed, use this name as part of title, otherwise use __name__
        attribute from estimator.
        Used, for example, when the estimator is a Pipeline and the
        final estimator should be in the title.

    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the
        current Axes.

    figsize : tuple, optional
        Only applicable if ax is passed.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the validation curve drawn onto
        it.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    train_scores, valid_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=n_jobs)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    if estimator_name is None:
        estimator_name = type(estimator).__name__
    ax.set_title('Validation Curve for {}'.format(estimator_name))

    ax.set_xlabel(param_name)
    ax.set_ylabel('Score')

    lw = 2

    ax.plot(param_range, train_scores_mean, label='Training score',
            color='darkorange', lw=lw)
    ax.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2,
                    color='darkorange', lw=lw)

    ax.plot(param_range, valid_scores_mean, label='Cross-validation score',
            color='navy', lw=lw)
    ax.fill_between(param_range, valid_scores_mean - valid_scores_std,
                    valid_scores_mean + valid_scores_std, alpha=0.2,
                    color='navy', lw=lw)

    ax.legend(loc='best')

    return ax


# TODO: For sklearn labels use __name__ instead of convoluted extraction
# TODO: Add condition that grid must have at least 2 dims
# TODO: Finish level_order implementation (swap level on MultiIndex)
# TODO: Format long float tick labels as scientific notation
# TODO: Unittest
# TODO: Plot data to/as DataFrame
class GridSearchExplorer:
    """Allows explorating of GridSearchCV results with plots.

    This object allows visualization with heatmaps and pointplots.
    When the GridSearchCV object has more than 2 dimensions of
    parameter sets, heatmaps and pointplots will have different
    interpretations. This does not support param_grid if list of dicts

    Pointplots support aggregating data into estimates, and will
    show a confidence interval by default in this case.
    Values across parameter sets not specified as x or hue will
    be estimated with numpy's mean function by default.

    Heatmaps do not support estimates, so parameter sets not
    specified for the x-axis will be put onto the y-axis as
    combinations taken from a pandas MultiIndex.
    """

    def __init__(self, grid, estimator_name=None):
        """Grid instantiated and fit to data.

        Parameters
        ----------
        grid : GridSearchCV

        estimator_name : str, optional
            If passed, use this name as part of title, otherwise use __name__
            attribute from estimator.
            Used, for example, when the estimator is a Pipeline and the
            final estimator should be in the title.
        """

        self.grid = grid
        self.param_grid = grid.param_grid
        self.param_names = sorted(self.param_grid)  # for reference
        self.plot_data = None

        if estimator_name is None:
            self.estimator_name = type(grid.best_estimator_).__name__
        else:
            self.estimator_name = estimator_name

    def _is_from_sklearn(self, obj):
        """Determines if obj is an sklearn object."""
        return any(str(x).startswith("<class 'sklearn")
                   for x in obj.__class__.__mro__)

    def _extract_sklearn_name(self, obj):
        pattern = r"[.]([^'.]+)'"
        return re.search(pattern, str(obj.__class__)).group(1)

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
            label = self._extract_sklearn_name(obj)
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

    def plot(self, x, hue=None, metric='mean_test_score', kind='heatmap',
             cbar_label=True, annot=True, fmt='.3f', ax=None, figsize=None,
             **kwargs):
        """Plots GridSearchCV results onto heatmap or pointplot.

        Parameters
        ----------
        x : str
            Parameter whose values go on the x-axis.
            For heatmap, the other parameter's values sets will be plotted
            on the y-axis. If the grid has a third parameter, then
            multiindex-style combinations are shown on y-axis.

        hue : str, optional
            Set of params plotted as lines in pointplot. If None,
            mean of scores for x across other params is in single line.
            Ignored if kind='heatmap'

        metric : str, default is 'mean_test_score'
            Any metric that is a key in grid.cv_results.
            Examples:
            - 'mean_test_score' over validation folds
            - 'mean_train_score'
            Special case: 'test_less_train'
                The difference between 'mean_test_score' and
                'mean_train_score'. Used to evaluate overfitting.

        kind : {'heatmap', 'point'}
            The kind of plot to draw.

        cbar_label : bool, default True
            If True, label the heatmap colorbar with metric.
            Ignored if kind='heatmap'.

        annot : bool or rectangular dataset, optional
            If True, write the data value in each cell. If an array-like
            with the same shape as ``data``, then use this to annotate
            the heatmap instead of the raw data.
            Ignored if kind='heatmap'.

        fmt : string, default is '.3f'
            String formatting code to use when adding annotations.
            Ignored if kind='heatmap'.

        ax : matplotlib Axes, optional
            Axes object to draw the plot onto, otherwise uses the
            current Axes.

        figsize : tuple, optional
            Only applicable if ax is passed.

        # TODO: level_order : str, optional
            Allows custom ordering of y-axis levels when there are more
            than 2 grid dimensions.

        kwargs : key, value pairings
            Keyword arguments passed to seaborn plotting function.

        Returns
        -------
        ax : matplotlib Axes
            Returns the Axes object with the heatmap or pointplot drawn
            onto it.
        """
        if x == hue:
            raise ValueError('x and hue must be different parameter names.')

        try:
            scores = self.grid.cv_results_[metric]
        except KeyError as e:
            if str(e) == "'test_less_train'":
                if kind != 'heatmap':
                    raise ValueError(
                        'Metric test_less_train only available for heatmap.')
            else:
                raise e

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.set_title('GridSearchCV scores for {}'.format(self.estimator_name))

        # Note about order of results used for reshaping:
        # Grid params are ordered alphabetically by str name,
        #     then iterated through with itertools.product
        #     for all combinations.
        # The order of iteration determines the order of the flattened
        #     arrays in grid.cv_results_
        sorted_by_key = sorted(self.param_grid.items())  # by key
        param_keys, param_vals = zip(*sorted_by_key)

        param_grid_labels = self._grid_vals_to_labels(param_vals)
        index = pd.MultiIndex.from_product(param_grid_labels, names=param_keys)

        if kind == 'heatmap':
            if hue is not None:
                raise ValueError('hue not supported by heatmap.')

            if metric == 'test_less_train':
                mean_test_scores = self.grid.cv_results_['mean_test_score']
                mean_train_scores = self.grid.cv_results_['mean_train_score']
                scores = mean_test_scores - mean_train_scores

            # Set up DataFrame into grid format
            self.plot_data = (
                pd.Series(scores, index=index)
                  .unstack(x)
            )
            # for lo-to-hi y-axis
            try:
                self.plot_data = self.plot_data.sort_index(ascending=False)
            except TypeError as e:
                # For grid parameters that are mixed types
                if 'unorderable types' not in str(e):
                    raise e

            sns.heatmap(data=self.plot_data, ax=ax, annot=annot, fmt=fmt,
                        **kwargs)

            if cbar_label:
                ax.collections[0].colorbar.set_label(metric)

        elif kind == 'point':
            # Convert to long-form for pointplot
            self.plot_data = (pd.Series(scores, index=index, name=metric)
                              .reset_index()
            )
            sns.pointplot(
                x=x, y=metric, hue=hue, data=self.plot_data, ax=ax, **kwargs
            )

            # If GridSearchCV has two parameter sets, remove
            # mean() enclosure from ylabel.
            # In this case, there would not be more than one value to
            # aggregate.
            if self.plot_data.shape[1] <= 3:
                ax.set_ylabel(metric)
        else:
            raise ValueError('{} not a supported plot kind.'.format(kind))

        return ax

    def plot_cv_train_comparison(self, x, figsize=(12, 4),
                                 annot=True, fmt='.3f',
                                 **kwargs):
        """
        Generate a figure of 3 heatmaps that compare train and test
        (or cross-validation) scores of the grid of parameters.

        The leftmost heatmap shows the difference between
        corresponding cells on the other two. One of the other
        heatmaps shows training scores, i.e. the mean score on the
        training sets generated by the k-folds. The other shows mean
        scores on the test sets during that cv process.

        By having a representation of the gap between training
        and CV scores, the best parameter combination can be determined
        by the one that best balances the bias and variance of the model.

        Note: Beward misleading shades; range of scores might very small.

        Parameters
        ----------
        x : str
            Parameter whose values go on the x-axes of the heatmaps.
            The other parameter's values will be plotted
            on the y-axis. If the grid has a third parameter, then
            multiindex-style combinations are shown on y-axes.

        figsize : tuple, default is (12, 4)
            Passed to plt.subplots function.

        annot : bool or rectangular dataset, optional
            If True, write the data value in each cell. If an array-like
            with the same shape as ``data``, then use this to annotate
            the heatmap instead of the raw data.

        fmt : string, default is '.3f'
            String formatting code to use when adding annotations.

        kwargs : key, value pairings
            Keyword arguments passed to seaborn plotting function.

        Returns
        -------
        fig : matplotlib Figure with 3 Axes
            Returns the Figure object with the heatmaps drawn onto it.
        """
        comparison_data = {}
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=figsize)
        kind = 'heatmap'

        self.plot(x=x, kind=kind, metric='test_less_train', ax=ax1,
                  cbar_label=False, annot=annot, fmt=fmt, **kwargs)
        fig.suptitle(ax1.get_title(), fontsize='x-large', y=1.05)
        ax1.set_title('test_less_train')
        comparison_data['test_less_train'] = self.plot_data

        self.plot(x=x, kind=kind, metric='mean_test_score', ax=ax2,
                  cbar_label=False, annot=annot, fmt=fmt, **kwargs)
        ax2.set_ylabel('')
        ax2.set_title('mean_test_score')
        comparison_data['mean_test_score'] = self.plot_data

        self.plot(x=x, kind=kind, metric='mean_train_score', ax=ax3,
                  cbar_label=False, annot=annot, fmt=fmt, **kwargs)
        ax3.set_ylabel('')
        ax3.set_title('mean_train_score')
        comparison_data['mean_train_score'] = self.plot_data

        fig.suptitle('Learning Curve for {}'.format(self.estimator_name),
                     fontsize='x-large', y=1.05)
        self.plot_data = comparison_data

        return fig
