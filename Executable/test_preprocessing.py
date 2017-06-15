# -*- coding: utf-8 -*-

"""
we test preprocessing functions.
"""

import pandas as pd
from pandas.util.testing import assert_series_equal
from numpy.testing import assert_array_equal

from preprocessing import exp_int, AggTotalExtractor

class TestRentalInterest(object):

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

class TestAggTotalExtractor(object):

    def test_fit_transform(self):
        train = pd.Series(['a', 'a', 'a', 'b', 'b', 'c'], name='col')

        fitted = AggTotalExtractor().fit(train)
        result = fitted.transform(train)
        expected = pd.Series([3, 3, 3, 2, 2, 1], name='col').values.reshape(-1, 1)
        assert_array_equal(result, expected)
        # assert_series_equal(result, expected)

        test = pd.Series(['a', 'a', 'b', 'c', 'd'], name='col')
        result = fitted.transform(test)
        expected = pd.Series([5, 5, 3, 2, 1], name='col').values.reshape(-1, 1)
        assert_array_equal(result, expected)
        # assert_series_equal(result, expected)








