import numpy as np
from scipy import stats


def outl_dropped_msg(name, pre_len, post_len, k=None):
    """Return info on outliers dropped.

    Parameters
    ----------
    name : str
        Name of feature whose outliers were dropped
    {pre_len, post_len} : int
        Number of items before and after outliers were dropped.
    k : float
        Tukey constant if Tukey's range test used to drop ouliers.
    """
    diff = pre_len - post_len
    info = 'Dropped {} {} outliers ({:.2%})'.format(
        diff, name, diff/pre_len
    )
    if k is not None:
        info += ' with Tukey test constant {}'.format(k)

    return info


def is_outl_val(col, k=1.5, log=False):
    """Detects outliers using customized Tukey's range test.

    Classifies outlier as a point that lies above Q3 + k*IQR or
    below Q1 - K*IQR, where IQR is interquartile range and k is a
    positive constant, roughly given as 1.5 for an outlier, or 3
    for a far out point.

    Parameters
    ----------
    col : pandas.Series
        Specific feature to be assessed.
    k : float, default is 1.5
        Tukey constant for Tukey's range test.
    log : bool, default is False
        If True, apply natural logarithm to values before applying
        Tukey's range test.

    Returns
    -------
    result : pandas.Series
        Of boolean values, true if value is outlier
    """
    if log:
        col = np.log(col+1)
    q1, q3 = col.quantile(q=.25), col.quantile(q=.75)
    step = k * (q3-q1)
    lo, hi = q1-step, q3+step

    result = (col<lo) | (col>hi)
    return result


def is_price_outl(price_col, k=3, log=True):
    """Return boolean Series from using modified Tukey range test."""
    price = np.log(price_col) if log else price_col
    return is_outl_val(price_col, k=k)


def drop_price_outl(df, k=3, log=True, msg=True):
    """Drop outliers deterined by Tukey's range test."""
    pre_len = len(df)
    df = df.loc[~is_price_outl(df.price, k=k, log=log)]
    post_len = len(df)
    if msg:
        print(outl_dropped_msg('price', pre_len, post_len, k))

    return df


def is_bath_outl(bathrooms):
    return ~bathrooms.isin([1, 2, 3])


def drop_bath_outl(df, msg=True):
    pre_len = len(df)
    df = df.loc[~is_bath_outl(df.bathrooms)]
    post_len = len(df)
    if msg:
        print(outl_dropped_msg('bathroom', pre_len, post_len))

    return df


def is_bed_outl(bedrooms):
    return bedrooms > 4


def drop_bed_outl(df, msg=True):
    pre_len = len(df)
    df = df.loc[~is_bed_outl(df.bedrooms)]
    post_len = len(df)
    if msg:
        print(outl_dropped_msg('bedroom', pre_len, post_len))

    return df


def drop_bbp_outl(df, msg=True):
    """Drop all outliers from bedroom, bathroom, price features.

    Uses thresholds for determining outliers
    """
    pre_len = len(df)
    df = drop_price_outl(df, k=3.5, log=True, msg=False)
    df = drop_bath_outl(df, msg=False)
    df = drop_bed_outl(df, msg=False)
    post_len = len(df)
    if msg:
        print(outl_dropped_msg(
            'bed, bath and price', pre_len, post_len)
        )

    return df


def is_lat_outl(lat):
    return ((lat == 0) |
            (lat < 40) |  # left limit
            (lat > 40.95)   # right limit
    )


def is_long_outl(long):
    return ((long == 0) |
            (-73.7 < long) |  # top limit
            (-74.05 > long))   # bottom limit


def is_geo_outl(lat_col, long_col):
    return is_lat_outl(lat_col) | is_long_outl(long_col)


def drop_geo_outl(df, msg=True):
    pre_len = len(df)
    df = df.loc[~is_geo_outl(df.latitude, df.longitude)]
    post_len = len(df)
    if msg:
        print(outl_dropped_msg('geo-coordinate', pre_len, post_len))

    return df


def imp_missing_baths(df):
    """Imputes bathroom values of 0."""

    def impute_median(gr_df):
        median = np.median(gr_df.bathrooms)
        return gr_df.replace(dict(bathrooms={0: median}))

    return df.groupby('bedrooms', group_keys=False).apply(impute_mode)
