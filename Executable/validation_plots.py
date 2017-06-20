"""
Sources:
http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html


"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import ShuffleSplit


def best_grid_results(grid):
    results = []
    results.append('Best score: {:.5f}\n'.format(grid.best_score_))
    results.append('*** For parameters: ***')
    for param, val in grid.best_params_.items():
        results.append('{}: {}'.format(param, val))
    return '\n'.join(results)


def plot_learning_curve(estimator, X, y, ylim=None, cv=None, scoring=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the 'fit' and 'predict' methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    fig, ax = plt.subplots()

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs,
        train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color='r')
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color='g')
    ax.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label='Training score')
    ax.plot(train_sizes, test_scores_mean, 'o-', color='g',
             label='Cross-validation score')

    ax.legend(loc='best')
    return ax


def plot_validation_curve(estimator, X, y, param_name, param_range,
                          cv=None, scoring='accuracy', n_jobs=1):
    """Plots vaildation curve for given hyperparamters.

    Parameters
    ----------
    estimator : sklearn Classifier
    param_name :  str
    param_range : iterable
    kwargs : passed to matplotlib plot

    Returns
    -------
    plt : matplotlib plot
    """
    fig, ax = plt.subplots()

    train_scores, valid_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=n_jobs)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)


    ax.set_title('Validation Curve with {}'.format(type(estimator).__name__))
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