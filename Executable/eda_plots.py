
import re
import seaborn as sns
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

from main import read_rental_interest


# TODO: Add support for unordered x values
# TODO: Add more detailed message for value out of bins exception
# TODO: consider warning instead of exception
class ConditionalProbabilities:

    """Nested histogram or countplot of conditional probabilities.

    Essentailly, the conditional variable is normalized, useful in
    cases where it is a target variable with unbalanced class labels.

    If nested bars are equal, x and conditional variable are perfectly
    independent.

    The plot then becomes a representation of the effect of the x
    variable on the conditional within the distribution of x.
    The combination of the two allows interpretation of effect relative
    to how prevelant x variable is in the dataset. The normalization of
    hue variable prevents class imbalance from making these
    interpretations difficult.
    """

    def __init__(self, x, data, cond, bins=None, cond_bins=None, **kwargs):
        """
        Parameters
        ----------
        x : str
            Name of column in ``df`` that is the 'affecting' variable.
            If continuous, bins must be specified.

        df : DataFrame

        cond : str
            Name of column in ``df`` nested within x variables, that
            is the 'affected' variable.
            Equivalent to 'hue' plot parameter in seaborn.
            If continuous, cond_bins must be specified.

        bins, cond_bins : int or sequence of scalars, optional
            Histogram bins required for continuous data, passed to
            pd.cut with include_lowest=True.
        """
        data = data.copy()

        if bins is not None:
            data[x] = pd.cut(
                data[x], bins=bins, include_lowest=True
            )
            if data[x].isnull().any():
                raise ValueError(
                    'Some x values fall outside x_bins.'
                )

        if cond_bins is not None:
            data[cond] = pd.cut(
                data[cond], bins=cond_bins,
                include_lowest=True
            )
            if data[cond].isnull().any():
                raise ValueError(
                    'Some cond values fall outside bins.'
                )

        self.data = data
        self.x = x
        self.cond = cond
        self.bins = bins
        self.cond_bins = cond_bins
        self.conditions = self.data[self.cond].unique()

    @property
    def cond_prob_data(self):
        """Long-form DataFrame of conditional probabilities."""

        return (self.data.groupby(self.cond)[self.x]
                    .value_counts(normalize=True)
                    .rename('cond_prob')
                    .reset_index()
                    .sort_values(self.x)
        )

    def plot(self, palette=None, ax=None, **kwargs):
        """

        Parameters
        ----------
        palette : seaborn color palette or dict, optional
            Colors to use for the different levels of the hue
            variable. Should be something that can be interpreted
            by color_palette, or a dictionary mapping hue levels to
            matplotlib colors.

            Default palette is 'Reds' hues for ordinal cond variable.
            Number of shades matches number of cond bins or categories,
            and is shift up one gradation so that lightest is not too
            light.

        ax : matplotlib Axes, optional
            Axes object to draw the plot onto, otherwise uses the
            current Axes.

        kwargs : key, value mappings
            Other keyword arguments are passed to ``sns.barplot``

        Returns
        -------
        ax : matplotlib Axes
            Returns the Axes object with the boxplot drawn onto it.
        """
        if palette is None:
            palette = sns.color_palette('Reds', len(self.conditions)+1)[1:]

        ax = sns.barplot(
            x=self.x, y='cond_prob', hue=self.cond, data=self.cond_prob_data,
            palette=palette, ax=ax, **kwargs
        )

        ax.set_ylabel('P( {} | {} )'.format(self.x, self.cond))
        ax.legend(title=self.cond)
        title = 'Normalized {} by {}'.format(self.cond, self.x)
        title += '\n*nested bars are even if no effect'
        ax.set_title(title, y=1.02)

        return ax

    @property
    def chi2_pval(self):
        """Calculate p-value for chi2 test from contigency table."""
        obs = self.data.groupby(self.cond)[self.x].value_counts()
        contigency_table = obs.unstack().fillna(0)
        _, p, *_ = stats.chi2_contingency(contigency_table)
        return p

    @property
    def chi2_pval_msg(self):
        return 'chi2 p-val = {:.4g}'.format(self.chi2_pval)


def plot_interest_pie(interest_series, **pie_kwargs):
    """Plots proportion of interest levels in listing data.

    Adapted from Matplotlib example:
    https://matplotlib.org/_sources/examples/pie_and_polar_charts/pie_demo_features.rst.txt
    """
    labels = 'Low (1)', 'Medium (2)', 'High (3)'
    cnts = interest_series.value_counts()
    explode = [.05] * 3

    with sns.color_palette(palette=sns.color_palette('Reds', 5)[:4]):
        plt.pie(cnts, labels=labels, autopct='%1.1f%%', startangle=90,
                explode=explode, **pie_kwargs)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')
    plt.title('Proportion of Listings by Interest Level', y=1.05)

    return plt.gca()


def plot_count_comparison(x, df_tr, df_te):
    """Compare counts of x between train and test data."""

    df_tr = df_tr[x].to_frame().assign(dataset='test')
    df_te = df_te[x].to_frame().assign(dataset='train')

    data = pd.concat([df_tr, df_te])

    return sns.countplot(x=x, hue='dataset', data=data)
