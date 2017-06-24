from copy import deepcopy
import pdb
import warnings
from imp import reload
import time
from pprint import pprint
import random
from copy import deepcopy
import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import (StratifiedShuffleSplit, train_test_split,
                                     validation_curve)
from grid_explore import GridSearchExplorer
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, make_scorer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import (LabelBinarizer, LabelEncoder,
                                   MinMaxScaler, StandardScaler,
                                   OneHotEncoder, Imputer)
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from IPython.display import display
from sklearn_pandas import DataFrameMapper
import xgboost as xgb
from pyglmnet import GLM

import pdir

from outlier_detection import *
from preprocessing import *
from eda_plots import *
from validation_plots import *
from validation_plots import best_grid_results
from main import *
from main import read_rental_interest


def exp_int(col, prior):
    """Expected interest level with Bayesian prior."""
    return (col.sum()+prior)/(len(col)+1)

def concat_mappers(mappers, df_out=True, input_df=True):
    mapmerge_mappersper_features = []
    _ = [mapper_features.extend(deepcopy(x.features)) for x in mappers]

    mapper = DataFrameMapper(
        mapper_features, df_out=df_out, input_df=input_df
    )
    return mapper

def concat_pipelines(pipelines):
    steps = it.chain(*(deepcopy(pipe.steps) for pipe in pipelines))
    return Pipeline(steps)

def get_word_cnt(doc):
    # TODO: Could be more efficient than slow regex engine
    return len(re.findall(r'\w+', doc))

class LogTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.log(X)

class ToFrame(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pd.DataFrame(X, columns=self.columns)


class SqrtTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.sqrt(X)

class LenExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.vectorize(len)(X)


class WordCntExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.vectorize(get_word_cnt)(X)

class DayBinarizer(LabelBinarizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, y, _=None):
        if not isinstance(y, pd.Series):
            raise TypeError('DayBinarizer only accepts Series.')
        return super().fit(y.dt.weekday_name)

    def transform(self, y, _=None):
        if not isinstance(y, pd.Series):
            raise TypeError('DayBinarizer only accepts Series.')
        return super().transform(y.dt.weekday_name)

class WeekendExtractor(TransformerMixin):

    def fit(self, y):
        if not isinstance(y, pd.Series):
            raise TypeError('WeekendExtractor only accepts Series.')

        return self

    def transform(self, y):
        if not isinstance(y, pd.Series):
            raise TypeError('WeekendExtractor only accepts Series.')

        return y.dt.weekday > 4


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


    def fit(self, df, y=None):
        if not isinstance(df, pd.DataFrame) or df.shape[1] != 1:
            raise TypeError(
                'GroupSumExtractor only accepts DataFrame with one column.')
        self.fitted_cnts = df.iloc[:, 0].value_counts()
        self.fitted_n = len(df)

        return self

    def transform(self, df, y=None):
        if not isinstance(df, pd.DataFrame) or df.shape[1] != 1:
            raise TypeError(
                'GroupSumExtractor only accepts DataFrame with one column.')
        check_is_fitted(self, 'fitted_cnts')

        df = df.copy()

        new_cnts = df.iloc[:, 0].value_counts()
        n = len(df)

        # Heuristic for determining transformation on fitted data
        if new_cnts.equals(self.fitted_cnts):
            cnts = self.fitted_cnts
        else:
            cnts = self.fitted_cnts.add(new_cnts, fill_value=0)
            n += self.fitted_n


        if self.normalize:
            cnts = cnts/n
        merged = (pd.merge(df, cnts.to_frame('cnts'), left_on=df.columns[0],
                           right_index=True)
                    .loc[:, ['cnts']]
        )
        assert not merged.isnull().any().any(), (
            'DataFrame from merge should not have any missing values.')

        # Shape for valid input to other feature transformers,
        # important when pipelining.
        return merged


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
        return self

    # 313 imputed with .35 dataset
    def transform(self, df, y=None):
        df = df.copy()
        no_bed = df.bedrooms == 0
        no_bath = df.bathrooms == 0
        all_miss = no_bath & no_bed

        if self.imp_val is False:
            bed_med = np.median(df.bedrooms)
            bath_med = np.median(df.bathrooms)
            grp_bath_med = df.groupby('bedrooms')['bathrooms'].median()

            df.loc[all_miss, 'bedrooms'] = bed_med
            df.loc[all_miss, 'bathrooms'] = bath_med

            gb = df.groupby('bedrooms')
            for n_beds, gr_df in gb:
                idx = gr_df.loc[gr_df.bathrooms==0].index
                df.loc[idx, 'bathrooms'] = grp_bath_med[n_beds]
        else:
            df.loc[all_miss, 'bedrooms'] = self.imp_val
            df.loc[no_bath, 'bathrooms'] = self.imp_val

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

        df.loc[df.latitude==0, 'latitude'] = lat_val
        df.loc[df.longitude==0, 'longitude'] = long_val

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


extractor = FeatureUnion([
    ('dense', Pipeline([
        ('extract', FeatureUnion([
            ('basic', FeatureUnion([
                ('coordinates', make_pipeline(
                    ItemSelector(['latitude', 'longitude']),
                    LatLongImputer(),
                )),
                ('pre_processed', ItemSelector(
                    ['lg_price', 'n_photos', 'n_feats', 'descr_wcnt'])),
                ('rooms', make_pipeline(
                    ItemSelector(['bathrooms', 'bedrooms']),
                    BedBathImputer())),
                ])),

#             ('aggregate', make_pipeline(
#                 FeatureUnion([
#                     ('n_posts', make_pipeline(
#                         ItemSelector(['manager_id']),
#                         GroupSumExtractor())),
#                     ('building_activity', make_pipeline(
#                         ItemSelector(['building_id']),
#                         GroupSumExtractor()))
#                     ]),
#                 LogTransformer(),
#             )),
        ])),
        ('standardize', StandardScaler())
    ])),

    ('sparse', FeatureUnion([
        ('day_names', ItemSelector([
            'created_Friday', 'created_Monday', 'created_Saturday',
            'created_Sunday', 'created_Thursday', 'created_Tuesday',
            'created_Wednesday'
            ])
        ),
        ('flags', ItemSelector(['no_photo', 'no_feats', 'no_desc']))

    ])),
])