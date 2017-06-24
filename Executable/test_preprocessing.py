# -*- coding: utf-8 -*-

"""
we test preprocessing functions.
"""

import pandas as pd
from pandas.util.testing import assert_series_equal, assert_frame_equal
from numpy.testing import assert_array_equal
import numpy as np

from preprocessing import exp_int, GroupSumExtractor

class TestRentalInterest:

    def test_bayes_prior_exp_interest(self):
        man_ids = ['a', 'a', 'a', 'b', 'b', 'c']
        int_lev = [1, 2, 3, 2, 3, 1]

        df = pd.DataFrame(dict(manager_id=man_ids, interest_level=int_lev))

        prior = 1.2
        exp_int1 = df.groupby('manager_id')['interest_level'].apply(exp_int, prior)

        exp_a = (prior + 1 + 2 + 3)/4
        exp_b = (prior + 2 + 3)/3
        exp_c = (prior + 1)/2

        exp_int2 = pd.Series(dict(a=exp_a, b=exp_b, c=exp_c))

        assert_series_equal(exp_int1, exp_int2, check_names=False)

class TestGroupSumExtractor:

    def test_fit_transform(self):
        train = pd.DataFrame(['a', 'a', 'a', 'b', 'b', 'c'])

        fitted = GroupSumExtractor(normalize=False).fit(train)
        expected = np.array([3, 3, 3, 2, 2, 1]).reshape(-1, 1)
        result = fitted.transform(train)
        assert_array_equal(result.values, expected)

        test = pd.DataFrame(['a', 'a', 'd'])
        expected = np.array([5, 5, 1]).reshape(-1, 1)

        result = fitted.transform(test)
        assert_array_equal(result.values, expected)

    def test_fit_transform_normed(self):
        train = pd.DataFrame(['a', 'a', 'a', 'b', 'b', 'c'])
        fitted = GroupSumExtractor(normalize=True).fit(train)
        fitted_size = len(train)
        cnts1 = dict(a=3, b=2, c=1)
        arr = np.array([3, 3, 3, 2, 2, 1])

        normed = arr/fitted_size
        expected = normed.reshape(-1, 1)

        result = fitted.transform(train)
        assert_array_equal(result.values, expected)

        test = pd.DataFrame(['a', 'a', 'd'])
        total_size = fitted_size + len(test)
        cnts2 = dict(a=2, d=1)
        cnts_tot = dict(a=3+2, b=2, c=1, d=1)
        arr2 = np.array([5, 5, 1])
        normed = arr2/total_size
        expected = normed.reshape(-1, 1)

        result = fitted.transform(test)
        assert_array_equal(result.values, expected)

class _TestAverageInterestExtractor:

    def test_fit_transform(self):
        data = dict(
            manager_id='abbccc',
            interest_level=[3, 1, 3, 2, 2, 3]
        )
        train_listings = pd.DataFrame(data)
        fitted = AverageInterestExtractor().fit(train_listings)

        a_sum, a_num = 3, 1
        a_avg = a_sum / a_num

        b_sum, b_num = (1+3), 2
        b_avg = b_sum / b_num

        c_sum, c_num = (2+2+4), 3
        c_avg = c_sum / c_num

        exp_data = dict(
            manager_id='abbccc',
            avg_interest=[a_avg, b_avg, b_avg, c_avg, c_avg, c_avg]
        )
        expected = pd.DataFrame(exp_data)
        result = fitted.transform(train_listings)

        assert_dataframe_equal(result, expected)

        data = dict(
            manager_id='bbbce',
            interest_level=[1, 2, 2, 2, 1]
        )
        test_listings = pd.DataFrame(data)

        exp_data = dict(
            manager_id='bbbce',
            avg_interest=[b_avg, b_avg, b_avg, c_avg, np.nan]
        )
        expected = pd.DataFrame(exp_data)
        result = fitted.transform(train)

        assert_dataframe_equal(result, expected)








