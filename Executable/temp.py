
from copy import deepcopy
import pdb
import warnings
from imp import reload
import time
from pprint import pprint
import random

import pandas as pd
import numpy as np

from sklearn.model_selection import (StratifiedShuffleSplit, train_test_split,
                                     validation_curve)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, make_scorer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import (LabelBinarizer, LabelEncoder,
                                   MinMaxScaler, StandardScaler, OneHotEncoder)
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from IPython.display import display
from sklearn_pandas import DataFrameMapper

import pdir

from preprocessing import *
from main import *



pd.set_option('display.float_format', lambda x: '%.5f' % x)

SEED = 42
np.random.seed(SEED)

DAT_DIR = '../Data/'
SUBM_DIR = '../Submissions/'
TEST_DIR = '../Tests/'
TRAIN_FP = ''.join([DAT_DIR, 'train.json'])
TEST_FP = ''.join([DAT_DIR, 'test.json'])
SAMPLE_FP = ''.join([DAT_DIR, 'sample_submission.csv'])
DF_TRAIN_PKL = ''.join([DAT_DIR, 'df_train.pkl'])

scorer_acc = make_scorer(accuracy_score)

# using built-in 'neg_log_loss' scoring param used for simplicity
# source code show exact same make_scorer call, kept for reference
scorer = make_scorer(log_loss, greater_is_better=False,
                     needs_proba=True)

scoring = 'neg_log_loss'



mapper = DataFrameMapper([
    (['bathrooms'], StandardScaler()),
    (['bedrooms'], StandardScaler()),
    (['created'], ExtractBinarizeDay()),
    (['created'], None),
    (['latitude'], MinMaxScaler()),
    (['longitude'], MinMaxScaler()),
    (['price'], [LogTransformer(), StandardScaler()]),
    ('manager_id', [AggTotalExtractor(), LogTransformer(), StandardScaler()],
         {'alias': 'n_posts'}),
    ('building_id', [AggTotalExtractor(), LogTransformer(), StandardScaler()],
         {'alias': 'n_buildings'}),
    (['photos'], [LenExtractor(), SqrtTransformer(), StandardScaler()],
         {'alias': 'sq_n_photos'}),
    (['features'], [LenExtractor(), SqrtTransformer(), StandardScaler()],
         {'alias': 'sq_n_feats'}),
    (['description'], [LenExtractor(), SqrtTransformer(), StandardScaler()],
         {'alias': 'sq_len_descr'}),
], input_df=True, df_out=True)

# booleans
def make_boolean_cols(df):
    df = df.copy()
    # Use min for zero value of standardized data
    df['no_photo'] = df.sq_n_photos == df.sq_n_photos.min()
    df['no_feats'] = df.sq_n_feats == df.sq_n_feats.min()
    df['no_desc'] = df.sq_len_descr == df.sq_n_feats.min()
    df['one_post'] = df.lg_n_posts == df.lg_n_posts
    
    return df


df = read_rental_interest(DF_TRAIN_PKL, read_pkl=True)

X, y = df.drop('interest_level', axis=1), df.interest_level

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.5, random_state=SEED, stratify=y)

mapper = mapper.fit(X_train)
X_train = mapper.transform(X_train)
X_train = make_boolean_cols(X_train)

X_test = mapper.transform(X_test)
X_test = make_boolean_cols(X_test)