import pandas as pd


SEED = 42

def read_rental_interest(fp, frac=None, random_state=None,
                         encode_labels=True):
    """Reads Two Sigma Connect: Rental Listing Inquiries json."""

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

