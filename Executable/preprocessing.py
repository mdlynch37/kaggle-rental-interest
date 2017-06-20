from copy import deepcopy
import re

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer
from pandas.util.testing import assert_series_equal
import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper

from outlier_detection import *


def exp_int(col, prior):
    """Expected interest level with Bayesian prior."""
    return (col.sum()+prior)/(len(col)+1)

def combine_mappers(mappers, df_out=True, input_df=True):
    mapper_features = []
    _ = [mapper_features.extend(deepcopy(x.features)) for x in mappers]

    mapper = DataFrameMapper(
        mapper_features, df_out=df_out, input_df=input_df
    )
    return mapper

def get_word_cnt(doc):
    return len(re.findall(r'\w+', doc))

class LogTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self

    def transform(self, X):
        return np.log(X)

class SqrtTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self

    def transform(self, X):
        return np.sqrt(X)

class LenExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            result = X.apply(len).values.reshape(-1, 1)
        else:
            result = X.applymap(len).values

        return result

class WordCntExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self

    def transform(self, X):

        if isinstance(X, pd.Series):
            result = X.apply(get_word_cnt).values.reshape(-1, 1)
        else:
            result = X.applymap(get_word_cnt).values

        return result

class DayExtractor(LabelBinarizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, ser):

        if not isinstance(ser, pd.Series):
            raise TypeError('AggTotalExtractor only accepts Series.')

        return super().fit(ser.dt.weekday_name)

    def transform(self, ser):
        return super().transform(ser.dt.weekday_name)

class WeekendExtractor(BaseEstimator, TransformerMixin):

    def fit(self, ser):

        if not isinstance(ser, pd.Series):
            raise TypeError('AggTotalExtractor only accepts Series.')

        return self

    def transform(self, ser):
        return ser.dt.weekday > 4


class GroupSumExtractor(BaseEstimator, TransformerMixin):
    """Transform values into counts of those values.

    Uses right-merge of counts into counted values.

    Example use-case: 'manager_id' for a rental listing to useful
    aggregate of how many postings the listing's manager
    has made. This transformer is analogous to an instance-based
    learning algorithm like kNN which stores data from training.

    When fit and transform with training data, test data
    transformation includes counts from training data in
    aggregation. Normalizing counts at the end of each
    transformation keeps counts in the same scale for predictions.
    """
    def __init__(self, normalize=True):
        self.normalize = normalize


    def fit(self, ser, y=None):
        if not isinstance(ser, pd.Series):
            raise TypeError('GroupSumExtractor only accepts Series.')
        self.fitted_cnts = ser.value_counts()
        self.fitted_n = len(ser)

        return self

    def transform(self, ser, y=None):
        if not isinstance(ser, pd.Series):
            raise TypeError('GroupSumExtractor only accepts Series.')
        check_is_fitted(self, 'fitted_cnts')

        new_cnts = ser.value_counts()
        n = len(ser)

        # Heuristic for determining transformation on fitted data
        if new_cnts.equals(self.fitted_cnts):
            cnts = self.fitted_cnts
        else:
            cnts = self.fitted_cnts.add(new_cnts, fill_value=0)
            n = self.fitted_n + len(ser)


        if self.normalize:
            cnts = cnts/n
        merged = (pd.merge(ser.to_frame(), cnts.to_frame(), how='left',
                           left_on=ser.name, right_index=True)
                  .iloc[:, 1]
        )
        assert not merged.isnull().any(), (
            'Series from merge should not have any missing values.')

        # Shape for valid input to other feature transformers,
        # important when pipelining.
        return merged.values.reshape(-1, 1)


class AverageInterestExtractor(BaseEstimator, TransformerMixin):
    """Transform values into counts of those values.

    Uses right-merge of counts into counted values.

    Example use-case: 'manager_id' for a rental listing to useful
    aggregate of how many postings the listing's manager
    has made. This transformer is analogous to an instance-based
    learning algorithm like kNN which stores data from training.

    When fit and transform with training data, test data
    transformation includes counts from training data in
    aggregation. Normalizing counts at the end of each
    transformation keeps counts in the same scale for predictions.
    """
    pass
    # def __init__(self):
    #     self.normalize = normalize
    #
    # def fit(self, df, y=None):
    #
    #     self.fitted_cnts = df.value_counts()
    #     self.fitted_n = len(ser)
    #
    #     return self
    #
    # def transform(self, ser, y=None):
    #     if not isinstance(ser, pd.Series):
    #         raise TypeError('GroupSumExtractor only accepts Series.')
    #     check_is_fitted(self, 'fitted_cnts')
    #
    #     new_cnts = ser.value_counts()
    #
    #     # Heuristic for determining transformation on fitted data
    #     if new_cnts.equals(self.fitted_cnts):
    #         cnts = self.fitted_cnts
    #     else:
    #         cnts = self.fitted_cnts.add(new_cnts, fill_value=0)
    #         n = self.fitted_n + len(ser)
    #
    #
    #     if self.normalize:
    #         cnts = cnts/n
    #     merged = (pd.merge(ser.to_frame(), cnts.to_frame(), how='left',
    #                        left_on=ser.name, right_index=True)
    #               .iloc[:, 1]
    #     )
    #     assert not merged.isnull().any(), (
    #         'Series from merge should not have any missing values.')
    #
    #     # Shape for valid input to other feature transformers,
    #     # important when pipelining.
    #     return merged.values.reshape(-1, 1)

class BoolFlagTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, val, operator='eq'):
        self.val = val
        self.operator = operator

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        if self.operator == 'eq':
            bool_arr = self.val == X
        if self.operator == 'gt':
            bool_arr = self.val > X
        if self.operator == 'lt':
            bool_arr = self.val < X

        return bool_arr


class BedBathImputer(BaseEstimator, TransformerMixin):
    """Imputes missing bedrooms and bathrooms"""

    def __init__(self, imp_val=False):
        self.imp_val = imp_val

    def fit(self, df, y=None):
        if self.imp_val is False:
            self.bed_med = np.median(df.bedrooms)
            self.bath_med = np.median(df.bathrooms)
            self.grp_bath_med = df.groupby('bedrooms')['bathrooms'].median()
        return self

    # 313 imputed with .35 dataset
    def transform(self, df, y=None):
        df = df.copy()
        no_bed = df.bedrooms == 0
        no_bath = df.bathrooms == 0
        all_miss = no_bath & no_bed
        only_bath = no_bath | ~no_bed

        if self.imp_val is False:
            df.loc[all_miss, 'bedrooms'] = self.bed_med
            df.loc[all_miss, 'bathrooms'] = self.bath_med

            gb = df.groupby('bedrooms')
            for n_beds, gr_df in gb:
                idx = gr_df.loc[gr_df.bathrooms==0].index
                df.loc[idx, 'bathrooms'] = self.grp_bath_med[n_beds]
        else:
            df.loc[all_miss, 'bedrooms'] = self.imp_val
            df.loc[only_bath, 'bathrooms'] = self.imp_val

        return df


class LatLongImputer(BaseEstimator, TransformerMixin):

    def __init__(self, imp_val=False):
        self.imp_val = imp_val

    def fit(self, df, y=None):

        if self.imp_val is False:
            self.lat_mean = df.latitude.mean()
            self.long_mean = df.longitude.mean()

        return self

    # 68 imputed with .35 dataset
    def transform(self, df, y=None):
        df = df.copy()

        if self.imp_val is False:
            lat_val, long_val = self.lat_mean, self.long_mean
        else:
            lat_val = long_val = self.imp_val

        df.loc[is_lat_outl(df.latitude), 'latitude'] = lat_val
        df.loc[is_long_outl(df.longitude), 'longitude'] = long_val

        return df

class PriceOutlierDropper(BaseEstimator, TransformerMixin):

    def __init__(self, tukey=False):
        self.tukey = tukey

    def fit(self, df, y):
        return self

    def transform(self, df, y):
        df = df.copy()

        if self.tukey:
            is_outl = is_price_outl(df.price, k=tukey, log=True)
        else:
            # price outliers, based on values determined through analysis
            is_outl = (df.price < 600) | (df.price > 1e6)
        return df.loc[~is_outl]
