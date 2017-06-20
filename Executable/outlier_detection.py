import numpy as np
from scipy import stats


def outl_dropped_msg(name, pre_len, post_len, k=None):
    """Return info on outliers dropped."""
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
        Specific feature to be assessed
    k : int, default is 1.5
    log : bool, default is False

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


def is_price_outl(price, k=3, log=True):
    """Return boolean Series of outliers using modified Tukey method"""
    price = np.log(price) if log else price
    return is_outl_val(price, k=k)


def drop_price_outl(df, k=3, log=True, msg=True):
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
    return (lat == 0) | (40.95 < lat) | (lat < 40.4)

def is_long_outl(long):
    return (long == 0) | (-73.7 < long) | (long < -74.05)

def is_geo_outl(lat, long):
    return is_lat_outl(lat) | is_long_outl(long)


def drop_geo_outl(df, msg=True):
    pre_len = len(df)
    df = df.loc[~is_geo_outl(df.latitude, df.longitude)]
    post_len = len(df)
    if msg:
        print(outl_dropped_msg('geo-coordinate', pre_len, post_len))

    return df


def imp_missing_baths(df):
    """Imputes bathroom values of 0."""

    def impute_mode(gr_df):
        median = np.median(gr_df.bathrooms)
        return gr_df.replace(dict(bathrooms={0: median}))

    return df.groupby('bedrooms', group_keys=False).apply(impute_mode)



