"""
Test exploratory data analysis plot functions.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import pytest

import sys
sys.path.append('../Executable')

from eda_plots import ConditionalProbabilities
from main import read_rental_interest


class TestConditionalProbabilities:

    def test_cond_probs(self):
        dfs = []
        expects = []

        cond_A = (
            ['a']*3 +  # P(a|A) = .75
            ['b']*1    # P(b|A) = .25
        )
        dfs.append(pd.DataFrame(cond_A, columns=['prob_var']).assign(cond='A'))

        exp_A = dict(
            cond_prob=[.75, .25],
            prob_var=['a', 'b']
        )
        expects.append(pd.DataFrame(exp_A).assign(cond='A'))

        cond_B = (
            ['a']*1 +  # P(a|B) = .2
            ['b']*4    # P(b|B) = .8
        )
        dfs.append(pd.DataFrame(cond_B, columns=['prob_var']).assign(cond='B'))

        exp_B = dict(
            cond_prob=[.2, .8],
            prob_var=['a', 'b']
        )
        expects.append(pd.DataFrame(exp_B).assign(cond='B'))

        cond_C = (
            ['a']*3 +  # P(a|C) = .5
            ['b']*3    # P(b|C) = .5
        )
        dfs.append(pd.DataFrame(cond_C, columns=['prob_var']).assign(cond='C'))

        exp_C = dict(
            cond_prob=[.5, .5],
            prob_var=['a', 'b']
        )
        expects.append(pd.DataFrame(exp_C).assign(cond='C'))

        cond_D = (
            ['c']*3     # P(c|D) = 1
        )
        dfs.append(pd.DataFrame(cond_D, columns=['prob_var']).assign(cond='D'))

        exp_D = dict(
            cond_prob=[1],
            prob_var=['c']
        )
        expects.append(pd.DataFrame(exp_D).assign(cond='D'))

        df = pd.concat(dfs)
        expected = pd.concat(expects)

        cond_prob = ConditionalProbabilities(df, prob='prob_var',
                                             condition='cond')

        def sort_reset(df):
            df = df.sort_values(['cond', 'prob_var'])
            df = df.reset_index(drop=True)
            return df

        result = sort_reset(cond_prob.cond_probs)
        expected = sort_reset(expected)
        expected = expected.reindex(columns=result.columns)

        assert_frame_equal(result, expected)
