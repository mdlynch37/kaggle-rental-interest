import pandas as pd
import pickle

from sklearn.base import BaseEstimator, TransformerMixin


SEED = 42

def read_rental_interest(fp, frac=None, random_state=None,
                         encode_labels=True):
    """Reads Two Sigma Connect: Rental Listing Inquiries json."""

    if fp.split('.')[-1] == 'pkl':
        return load_pickle(fp)

    df=(pd.read_json(fp)
        .set_index('listing_id')
    )
    df.created = pd.to_datetime(df.created)

    # encode for automatic ordering when plotting
    if True and 'interest_level' in df.columns:
        df.interest_level = df.interest_level.replace(
            ['low', 'medium', 'high'], [1, 2, 3]
        )
    # stratified sample
    if frac is not None:
        df = (df.groupby('interest_level')
             .apply(lambda x:x.sample(frac=frac,random_state=random_state))
        )

    return df


def save_submission(preds, fp):
    """Ensures correct format and class ordering."""
    submit = (pd.DataFrame(preds,
                           columns=['high', 'low', 'medium'],
                           index=X_sub.index)
              .loc[:, ['high', 'medium', 'low']])

    # ensure label ordering correct
    assert submit.high.sum() < submit.medium.sum() < submit.low.sum(), (
        'Interest level probpip abilities are mismatched')

    submit.to_csv(fp)

    return submit

def dump_pickle(obj, fp):
    with open(fp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(fp):
    with open(fp, 'rb') as f:
        obj = pickle.load(f)
    return obj

class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Source: http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


