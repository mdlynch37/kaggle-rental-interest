import pickle

import pandas as pd

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



