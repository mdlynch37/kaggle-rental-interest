import pickle

import pandas as pd

SEED = 42


def to_min_secs(seconds):
    """Returns tuple of min, seconds."""
    seconds = round(seconds)
    return seconds//60, seconds % 60


def read_rental_interest(fp, frac=None, random_state=None,
                         encode_labels=True):
    """Reads RentalHop listing json or .pkl data file.

    Parameters
    ----------
    fp : str
    frac : float, default None
        Specficies size of stratified sample of read dataset.
        Ignored for for .pkl files.
    random_state : int, default None
        Random seed for stratified sample.
        Ignored for for .pkl files.
    encode_labels : bool, default True
        If True, encode 'high', 'medium' and 'low' as 3, 2, 1 resp.
        Ignored for for .pkl files.

    Returns
    -------
    df : pandas DataFrame
    """
    if fp.split('.')[-1] == 'pkl':
        df = load_pickle(fp)
    else:
        df=(pd.read_json(fp)
            .set_index('listing_id')
        )
        df.created = pd.to_datetime(df.created)

        # encode for automatic ordering when plotting
        if encode_labels and 'interest_level' in df.columns:
            df.interest_level = df.interest_level.replace(
                ['low', 'medium', 'high'], [1, 2, 3]
            )
        # stratified sample
        if frac is not None:
            df = (df.groupby('interest_level')
                    .apply(lambda x:x.sample(frac=frac,random_state=random_state))
            )

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



