
import pickle

import pandas as pd


def to_min_secs(seconds):
    """Returns tuple of min, seconds."""
    seconds = round(seconds)
    return seconds//60, seconds % 60


def stratified_sample(df, y, n=None, frac=None, random_state=None):
    """Sample from DataFrame, stratifying on y column."""

    n_unique_y = len(df[y].unique())
    if n is not None and n % n_unique_y:
            raise ValueError(
                'n for sample must match number of unique '
                'stratification values ({}).'.format(n_unique_y)
            )
    grp_n = int(n/3)
    df = df.groupby(y).apply(
        lambda x: x.sample(n=grp_n, frac=frac, random_state=random_state)
    )
    return df


def read_rental_interest(fp, n=None, frac=None, random_state=None,
                         encode_labels=True):
    """Reads RentalHop listing json or .pkl data file.

    Parameters
    ----------
    fp : str
        Filepath to .json or .pkl

    n : int, optional
        Number of items from axis to return. Cannot be used with `frac`.
        Default = 1 if `frac` = None.

    frac : float, optional
        Fraction of axis items to return. Cannot be used with `n`.

    random_state : int, default None
        Random seed for stratified sample.

    encode_labels : bool, default True
        If True, encode 'high', 'medium' and 'low' as 3, 2, 1 resp.

    Returns
    -------
    df : DataFrame
    """
    if fp.split('.')[-1] == 'pkl':
        df = load_pickle(fp)
    else:
        df = (pd.read_json(fp)
                .set_index('listing_id')
        )
        df.created = pd.to_datetime(df.created)

    # stratified sample
    if n is not None or frac is not None:
        df = stratified_sample(
            df, 'interest_level', n=n, frac=frac, random_state=random_state
        )

    # encode for automatic ordering when plotting
    if encode_labels and 'interest_level' in df.columns:
        try:  # Catch case where pickled DataFrame already encoded.
            df.interest_level = df.interest_level.replace(
                ['low', 'medium', 'high'], [1, 2, 3]
            )
        except TypeError as e:
            msg = "Cannot compare types 'ndarray(dtype=int64)' and 'str'"
            if str(e) != msg:
                raise e

    return df


def save_submission(preds, index, fp):
    """Ensures correct format and class ordering."""
    submit = (pd.DataFrame(preds,
                           columns=['low', 'medium', 'high'],
                           index=index)
              .loc[:, ['high', 'medium', 'low']]  # right order
    )

    # ensure label ordering correct
    assert submit.high.sum() < submit.medium.sum() < submit.low.sum(), (
        'Interest level probabilities are mismatched')

    submit.to_csv(fp)


def dump_pickle(obj, fp):
    with open(fp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(fp):
    with open(fp, 'rb') as f:
        obj = pickle.load(f)
    return obj
