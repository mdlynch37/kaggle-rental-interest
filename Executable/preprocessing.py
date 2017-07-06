import re
from copy import deepcopy

import numpy as np
import pandas as pd
import xgboost as xgb
from IPython.display import display
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import (Imputer, LabelBinarizer, LabelEncoder,
                                   OneHotEncoder, StandardScaler)
from sklearn.utils.validation import check_is_fitted
from sklearn_pandas import DataFrameMapper

from outlier_detection import (drop_bbp_outl, drop_bed_outl, is_lat_outl,
                               is_long_outl, is_price_outl)


def exp_int(col, prior):
    """Expected interest level with Bayesian prior."""
    return (col.sum()+prior)/(len(col)+1)


def concat_mappers(mappers, df_out=True, input_df=True):
    mapper_features = []
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


def feature_prep(df, imp_constant=False):
    """Feature extraction and transformations that can be
    performed before train_test_split, outside of Pipeline.

    This is because these operations are done independently
    on each row.

    Parameters
    ----------
    df : pandas DataFrame
    imp_constant : False, default or int
        Integer passed will fill in missing values for bedrooms,
        bathrooms, latitude, and longitude values.
        Designed with XGBoost's missing_val parameter in mind,
        letting it used an algorithm to guess suitable values.

    Returns
    -------
    df : pandas DataFrame with additional features
    """
    # LogTransformer and SqrtTransformer not necessary for tree-based
    # algorithms, but since they are so cheap, they are left for
    # simplicity's sake

    mapper = DataFrameMapper([

        ('price', LogTransformer(),
            {'alias': 'price_lg'}),
        ('photos',
            [LenExtractor(), SqrtTransformer()],
            {'alias': 'n_photos_sq'}),
        ('features',
            [LenExtractor(), SqrtTransformer()],
            {'alias': 'n_feats_sq'}),
        ('description',
            [WordCntExtractor(), SqrtTransformer()],
            {'alias': 'descr_wcnt_sq'}),

        ('created', DayBinarizer()),

    ], input_df=True, df_out=True)

    new = mapper.fit_transform(df)
    new.index = df.index

    day_cols = dict(
        created_Friday='day_fri',
        created_Monday='day_mon',
        created_Saturday='day_sat',
        created_Sunday='day_sun',
        created_Thursday='day_thu',
        created_Tuesday='day_tue',
        created_Wednesday='day_wed'
    )
    new = new.rename(columns=day_cols)

    if imp_constant is not False:
        if not isinstance(imp_constant, int):
            raise ValueError(
                'imp_constant must be integer to fill missing values'
            )

        df = BedBathImputer(how=imp_constant).fit_transform(df)
        df = LatLongImputer(how=imp_constant).fit_transform(df)

    # Indicator features for logistic regression.
    new['no_photo_sq'] = new.n_photos_sq == 0
    new['no_feats_sq'] = new.n_feats_sq == 0
    new['no_desc_sq'] = new.descr_wcnt_sq == 0

    return pd.concat([df, new], axis=1)


class LogTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.log(X)


class ToFrame(BaseEstimator, TransformerMixin):
    """So that classifier is given names of columns"""
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

    def fit(self, y=None):
        if not isinstance(y, pd.Series):
            raise TypeError('WeekendExtractor only accepts Series.')

        return self

    def transform(self, y=None):
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
    """Imputes missing bedrooms and bathrooms in DataFrame.

    In two cases is data considered missing, each treated differently
    if how='medians' (in this order):
    1. Bedrooms and bathrooms, when both are zeros
    2. Zero-values bathrooms, when bedrooms > 0
    Note: Zero-valued bedrooms with bathrooms > 0 can be studios.

    If any other value but 'medians' is passed, that value will be used
    to impute missing values by the same criteria above.

    For case 1, the impute val for both bed and bath is the global
    median from the fitted dataset.
    For case 2, the impute val for bath will depend on its
    corresponding bed value. The impute val will be the median
    taken from rows with the same bed value (using groupby groups),
    also taken from the fitted dataset.
    """
    def __init__(self, how='medians'):
        """
        Parameters
        ----------
        how : 'medians', or any other value for simple impute
            Zero is redundant and will raise exception.
        """
        # Missing values bedrooms and bathrooms will always be 0
        if how == 0:
            raise ValueError('Imputation value of zero is redundant.')
        self.how = how

    def fit(self, df, y=None):
        if self.how == 'medians':
            self.bed_median = np.median(df.bedrooms)
            self.bath_median = np.median(df.bathrooms)
            self.grp_bath_median = df.groupby('bedrooms')['bathrooms'].median()
        else:
            self.imp_constant = self.how

        self.fitted = True
        return self

    # 313 imputed with .35 dataset
    def transform(self, df, y=None):
        check_is_fitted(self, 'fitted')

        df = df.copy()
        no_bed = df.bedrooms == 0
        no_bath = df.bathrooms == 0
        both_miss = no_bath & no_bed

        if self.how is 'medians':
            # Case 1:
            df.loc[both_miss, 'bedrooms'] = self.bed_median
            df.loc[both_miss, 'bathrooms'] = self.bath_median

            # Case 2:
            # For each group of rows, grouped by bedrooms values,
            # check for missing bathrooms. If there are some, get
            # median from fitted dataset's corresponding group.
            gb = df.groupby('bedrooms')
            for n_beds, bed_grp in gb:
                grp_missing = bed_grp.bathrooms == 0
                if not grp_missing.any():
                    continue

                # If bedroom val not in fitted data, thus no matching
                # group use group for nearest bedroom value in fitted
                # dataset.
                try:
                    imp_val = self.grp_bath_median[n_beds]
                except KeyError:
                    # Group keys (index in series) are always sorted
                    # ascending, thus when two values are equally near
                    # the lower one is taken.
                    nearest_idx = (np.abs(self.grp_bath_median.index - n_beds)
                                   .argmin())
                    imp_val = self.grp_bath_median[nearest_idx]

                # Get index of missing bathrooms for bedroom group and
                # do imputation in dataset
                idx = bed_grp.loc[grp_missing].index
                df.loc[idx, 'bathrooms'] = imp_val
        else:
            # Case 1
            df.loc[both_miss, 'bedrooms'] = self.imp_constant
            # Case 2
            df.loc[no_bath, 'bathrooms'] = self.imp_constant

        # Updated for assertion check
        no_bed = df.bedrooms == 0
        no_bath = df.bathrooms == 0
        both_miss = no_bath & no_bed

        # Edge-case would be if how='medians' and impute value
        # from fitted group medians is zero.
        # This could happen if there are so few bedroom values in
        # the fitted dataset that any of those values having no
        # bathrooms could cause a zero-valued median.
        assert not (no_bath & both_miss).any(), (
            'Impute failed. Potential but rare edge case details in code.')

        return df


class LatLongImputer(BaseEstimator, TransformerMixin):
    """Imputes missing latitude and longitude values in DataFrame."""

    def __init__(self, how='mean', broad=False):
        """
        Parameters
        ----------
        how : 'mean', or any other value for simple impute
            Zero is redundant and will raise exception.
        broad : bool, detault False
            If broad is True, use broader outlier definition
            used when plotting. If false, only consider zeros
            missing values.
        """
        if how == 0:
            raise ValueError('Imputation value of zero is redundant.')
        self.how = how
        self.broad = broad

    def fit(self, df, y=None):
        if self.how == 'mean':
            self.lat_mean = df.latitude.mean()
            self.long_mean = df.longitude.mean()
        else:
            self.imp_constant = self.how

        self.fitted = True

        return self

    # 68 imputed with .35 dataset with broad definition
    def transform(self, df, y=None):
        check_is_fitted(self, 'fitted')
        df = df.copy()

        if self.how == 'mean':
            lat_val, long_val = self.lat_mean, self.long_mean
        else:
            lat_val = long_val = self.imp_constant

        if self.broad:
            lat_outl  = is_lat_outl(df.latitude)
            long_outl = is_long_outl(df.longitude)
        else:
            lat_outl  = df.latitude == 0
            long_outl = df.longitude == 0

        df.loc[lat_outl, 'latitude'] = lat_val
        df.loc[long_outl, 'longitude'] = long_val

        return df


class PriceOutlierDropper(BaseEstimator, TransformerMixin):

    def __init__(self, tukey=False):
        self.tukey = tukey

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df = df.copy()

        if self.tukey:
            is_outl = is_price_outl(df.price, k=tukey, log=True)
        else:
            # price outliers, based on values determined through analysis
            is_outl = (df.price < 600) | (df.price > 1e6)
        return df.loc[~is_outl]


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Source:
    http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        return data[self.key]
