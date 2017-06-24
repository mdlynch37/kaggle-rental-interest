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


from sklearn.model_selection import (StratifiedShuffleSplit, train_test_split,
                                     validation_curve)
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
from sklearn_pandas import DataFrameMapper
import xgboost as xgb

import pdir

from outlier_detection import *
from preprocessing import *
from main import *
pd.set_option('display.float_format', lambda x: '%.5f' % x)

SEED = 42
np.random.seed(SEED)

DAT_DIR = '../Data/'
SUBM_DIR = '../Submissions/'
TEST_DIR = '../Tests/'
REPORT_IMG_DIR = '../Report-Images/'

TRAIN_FP = ''.join([DAT_DIR, 'train.json'])
TEST_FP = ''.join([DAT_DIR, 'test.json'])
SAMPLE_FP = ''.join([DAT_DIR, 'sample_submission.csv'])
DF_TRAIN_PKL = ''.join([DAT_DIR, 'df_train.pkl'])
DF_TEST_PKL = ''.join([DAT_DIR, 'df_test.pkl'])

scorer_acc = make_scorer(accuracy_score)

# using built-in 'neg_log_loss' scoring param used for simplicity
# source code show exact same make_scorer call, kept for reference
scorer = make_scorer(log_loss, greater_is_better=False,
                     needs_proba=True)

scoring = 'neg_log_loss'


if __name__ == '__main__':

    def best_grid_results(grid):
        results = []
        results.append('Best score: {:.5f}\n'.format(grid.best_score_))
        results.append('*** For parameters: ***')
        for param, val in grid.best_params_.items():
            results.append('{}: {}'.format(param, val))
        return '\n'.join(results)


    from main import *
    from preprocessing import *

    extractor = Pipeline([
        ('union', FeatureUnion([
            ('basic', make_pipeline(
                FeatureUnion([
                    ('price', make_pipeline(
                        ItemSelector(['price']),
                        LogTransformer())),
                    ('rooms', make_pipeline(
                        ItemSelector(['bathrooms', 'bedrooms']),
                        BedBathImputer())),
                    ('geo_coords', make_pipeline(
                        ItemSelector(['latitude', 'longitude']),
                        Imputer(missing_values=0)))
                ]),
                StandardScaler()
            )),
            ('created', make_pipeline(
                FeatureUnion([
                    ('list_lens', make_pipeline(
                        ItemSelector(['photos', 'features']),
                        LenExtractor())),
                    ('word_cnt', make_pipeline(
                        ItemSelector(['description']),
                        WordCntExtractor())),
                ]),
                SqrtTransformer(),
                StandardScaler()
            )),
            ('aggregate', make_pipeline(
                FeatureUnion([
                    ('n_posts', make_pipeline(
                        ItemSelector(['manager_id']),
                        GroupSumExtractor())),
                    ('building_activity', make_pipeline(
                        ItemSelector(['building_id']),
                        GroupSumExtractor()))
                ]),
                LogTransformer(),
                StandardScaler()
            )),
            ('binarized', make_pipeline(
                ItemSelector('created'),
                DayBinarizer()))
        ])),
    ])


    lr_clf = Pipeline([
        ('lr_clf', LogisticRegression(random_state=SEED, multi_class='multinomial',
                                      warm_start=True, max_iter=1000,
                                      solver='newton-cg'))
    ])

    pipe = concat_pipelines([extractor, lr_clf])
    # pipe.fit(X_train, y_train)

    from main import ItemSelector
    from preprocessing import *

    def feature_prep(df):
        mapper = DataFrameMapper([
                ('price', LogTransformer(),
                    {'alias': 'lg_price'}),
                ('photos',
                     [LenExtractor(), SqrtTransformer()],
                     {'alias': 'n_photos'}),
                ('features',
                     [LenExtractor(), SqrtTransformer()],
                     {'alias': 'n_feats'}),
                ('description',
                     [WordCntExtractor(), SqrtTransformer()],
                     {'alias': 'descr_wcnt'}),

                ('created', DayBinarizer()),
                ('created', WeekendExtractor(),
                    {'alias': 'is_weekend'}),
        ], input_df=True, df_out=True)

        new = mapper.fit_transform(X)
        new.index = df.index

        return pd.concat([df, new], axis=1)

    extractor = FeatureUnion([
        ('dense', Pipeline([
            ('extract', FeatureUnion([
                ('basic', FeatureUnion([
                    ('pre_processed', ItemSelector(
                        ['latitude', 'longitude', 'lg_price',
                         'n_photos', 'n_feats', 'descr_wcnt'])),
                    ('rooms', make_pipeline(
                        ItemSelector(['bathrooms', 'bedrooms']),
                        BedBathImputer())),
                    ])),

                ('aggregate', make_pipeline(
                    FeatureUnion([
                        ('n_posts', make_pipeline(
                            ItemSelector(['manager_id']),
                            GroupSumExtractor())),
                        ('building_activity', make_pipeline(
                            ItemSelector(['building_id']),
                            GroupSumExtractor()))
                        ]),
                    LogTransformer(),
                )),
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
            ('is_weekend', ItemSelector(['is_weekend'])),
        ])),
    ])


    df = read_rental_interest(DF_TRAIN_PKL)

    X, y = df.drop('interest_level', axis=1), df.interest_level

    X = feature_prep(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.35, random_state=SEED, stratify=y)

    pipe = Pipeline([
        ('extractor', extractor),
        ('lr_clf', LogisticRegression(random_state=SEED, multi_class='multinomial',
                                   warm_start=True, max_iter=1000,
                                   solver='newton-cg'))
    ])

    parameters = dict(
        lr_clf__C=np.logspace(-4, 4, 2),
        lr_clf__solver=['newton-cg', 'sag'],
    )

    grid = GridSearchCV(pipe, parameters, n_jobs=-1, scoring=scoring,
                        error_score=np.nan)


    grid.fit(X_train, y_train)

    print(best_grid_results(grid))
