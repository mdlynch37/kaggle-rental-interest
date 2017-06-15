from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer
from pandas.util.testing import assert_series_equal
import numpy as np
import pandas as pd


def exp_int(col, prior):
    """Expected interest level with Bayesian prior."""
    return (col.sum()+prior)/(len(col)+1)

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

class ExtractBinarizeDay(LabelBinarizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, ser):

        if not isinstance(ser, pd.Series):
            raise TypeError('AggTotalExtractor only accepts Series.')

        return super().fit(ser.dt.weekday_name)

    def transform(self, ser):
        return super().transform(ser.dt.weekday_name)

class AggTotalExtractor(BaseEstimator, TransformerMixin):

    def fit(self, ser):
        if not isinstance(ser, pd.Series):
            raise TypeError('AggTotalExtractor only accepts Series.')
        self.fitted_cnts_ = ser.value_counts()
        return self

    def transform(self, ser):
        check_is_fitted(self, 'fitted_cnts_')
        new_cnts = ser.value_counts()
        if new_cnts.equals(self.fitted_cnts_):
            cnts = self.fitted_cnts_
        else:
            cnts = self.fitted_cnts_.add(new_cnts, fill_value=0)

        result = (pd.merge(ser.to_frame(), cnts.to_frame(), how='left',
                           left_on=ser.name, right_index=True)
                  .iloc[:, 1]
                  .fillna(1)
                  .astype(int)
                  .rename(ser.name)
        )
        return result.values.reshape(-1, 1)
