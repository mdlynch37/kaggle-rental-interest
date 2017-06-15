
import re
import seaborn as sns
import numpy as np
from scipy import stats
import pandas as pd

from main import read_rental_interest


def plot_prob_x_for_hue(x, data, hue,
                        bins=None, hue_bins=None, chi2_msg=True,
                        exp_xlabels=False, palette=None, ax=None,
                        **kwargs):

    def chi2_p_val(obs_col, types_col, df):
        obs = (df.groupby(types_col)[obs_col]
               .apply(val_cnts, df[obs_col].unique())
               .unstack()
        )
        _, p, *_ = stats.chi2_contingency(obs)

        return '{} chi2 p-val = {:.4f} (rounded)'.format(types_col, p)

    def val_cnts(col,categories):
        """ Test"""
        # When grouping by a categorical feature, there could be
        # empty groups for categorical values with no rows
        if col.empty:
            return col

        cnts=col.value_counts()
        for cat in categories:
            if cat not in cnts.index:
                cnts.loc[cat]=0
        return cnts

    if bins is not None:
        data = data.copy()
        data[x] = pd.cut(data[x], bins=bins)

    if hue_bins is not None:
        if bins is not None:
            data = data.copy()
        data[hue] = pd.cut(data[hue], bins=hue_bins).astype(str)
        # Ensure numerical ordering is preserved
        d = {}
        for bin in data[hue].unique():
            lo, hi = bin.split(',')
            d[bin] = lo[1:] + ' -' + hi[:-1]
        data[hue] = data[hue].replace(d)


    if chi2_msg:
        print(chi2_p_val(hue, x, data))

    # normed for class (hue) imbalance
    prob_x = (data.groupby(hue)[x]
              .value_counts(normalize=True)
              .rename('prob')
              .reset_index()
              .sort_values(x))

    if palette is None:
        palette = sns.color_palette('Reds', len(data[hue].unique())+1)[1:]

    ax = sns.barplot(x=x, y='prob', hue=hue, data=prob_x,
                     palette=palette, ax=ax, **kwargs)

    if bins and exp_xlabels:
        xtickslables = []
        for lbl in ax.get_xticklabels():
            lo, hi = re.findall(r'[0-9.]+', lbl.get_text())
            lo, hi = np.exp(float(lo)), np.exp(float(hi))
            xtickslables.append('{:.0f} - {:.0f}'.format(lo, hi))
        ax.set_xticklabels(xtickslables)

    # Rotate xlabel if length cause overlapping in plot.
    max_xtick_len = prob_x[x].astype(str).map(len).max()
    if max_xtick_len > 2:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    ax.set_ylabel('p( {} | {} )'.format(x, hue))
    ax.set_title('Normalized Interest by {}'.format(x))

    return ax

def set_bedbath_types(df, test=False):

    df = df.copy()
    if test:  # separate columns for mutual exclusivity test
        a, b, c, d = ('Type'+str(i) for i in range(4))
    else:
        a = b = c = d = 'Type'

    df['Type'] = 'reasonable'

    no_bed = df.bedrooms == 0
    studio = no_bed & (df.bathrooms==1)
    bath_gt_bed = (df.bathrooms>df.bedrooms) & ~studio
    df.loc[bath_gt_bed, a] = 'bath > bed, except studio'

    no_bath = df.bathrooms == 0
    bed_gt_3bath = (df.bedrooms > df.bathrooms*3) & ~no_bath
    df.loc[bed_gt_3bath, b] = 'bed > bath*3'

    only_bath_missing = ~no_bed & no_bath
    df.loc[only_bath_missing, c] = 'bath == 0, with bedrooms'

    all_missing = no_bath & no_bed
    df.loc[all_missing, d] = 'both missing'

    return df

if __name__ == '__main__':
    DAT_DIR = '../Data/'
    TRAIN_FP = ''.join([DAT_DIR, 'train.json'])


    # see if activity of agents affects interest
    df = read_rental_interest(TRAIN_FP)
    prior = df.interest_level.mean()
    avg_int = (df.groupby('manager_id')['interest_level']
               .apply(exp_int, prior)
               .rename('avg_int')
    )
    data = pd.concat([n_posts,avg_int],axis=1)
    bins = [0, 1, 2, 3, 4, 6, 8, 12, 16, 25, 30, data.n_posts.max()]

    x = 'n_posts'
    hue = 'avg_int'

    plot_prob_x_for_hue(x=x, data=data, hue=hue, bins=bins)
    plt.show()
