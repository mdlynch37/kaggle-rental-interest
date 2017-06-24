from sklearn.preprocessing import StandardScaler
from preprocessing import (LogTransformer, WordCntExtractor, SqrtTransformer,
                           GroupSumExtractor, WordCntExtractor, LenExtractor,
                           DayBinarizer, LatLongImputer, BedBathImputer,
                           PriceOutlierDropper, WeekendExtractor,
                           combine_mappers)
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper

from copy import deepcopy


extractor = DataFrameMapper([
    (['bathrooms'], None),
    (['bedrooms'],  None),
    (['latitude'],  None),
    (['longitude'], None),
    ([('price')],     None),

    ('photos', LenExtractor(),
         {'alias': 'n_photos'}),
    ('features', LenExtractor(),
         {'alias': 'n_feats'}),
    ('description', WordCntExtractor(),
         {'alias': 'descr_wcnt'}),

    ('manager_id', GroupSumExtractor(),
         {'alias': 'n_posts'}),
    ('building_id', GroupSumExtractor(),
         {'alias': 'n_buildings'}),

#     ('created', DayBinarizer()),

], input_df=True, df_out=True)

scaler = DataFrameMapper([
    (['bathrooms'], StandardScaler()),
    (['bedrooms'], StandardScaler()),
    (['latitude'], StandardScaler()),
    (['longitude'], StandardScaler()),
    (['price'], [LogTransformer(), StandardScaler()]),
    ('n_photos', [SqrtTransformer(), StandardScaler()]),
    ('n_feats', [SqrtTransformer(), StandardScaler()]),
    ('descr_wcnt', [SqrtTransformer(), StandardScaler()]),
    ('manager_id', [LogTransformer(), StandardScaler()],
         {'alias': 'n_posts'}),
    ('building_id', [LogTransformer(), StandardScaler()],
         {'alias': 'n_buildings'}),
], input_df=True, df_out=True, default=None)




start_mapper = DataFrameMapper([
    (['bathrooms'], None),
    (['bedrooms'],  None),
    (['latitude'],  None),
    (['longitude'], None),
    (['price'],     None),
], input_df=True, df_out=True)

scl_start_mapper = DataFrameMapper([
    (['bathrooms'], StandardScaler()),
    (['bedrooms'], StandardScaler()),
    (['latitude'], StandardScaler()),
    (['longitude'], StandardScaler()),
    (['price'], [LogTransformer(), StandardScaler()]),
], input_df=True, df_out=True)

len_mapper = DataFrameMapper([
    ('photos', [LenExtractor()],
         {'alias': 'n_photos'}),
    ('features', [LenExtractor()],
         {'alias': 'n_feats'}),
    ('description', [WordCntExtractor()],
         {'alias': 'descr_wcnt'}),
], input_df=True, df_out=True)

scl_len_mapper = DataFrameMapper([
    ('photos', [LenExtractor(), SqrtTransformer(), StandardScaler()],
         {'alias': 'n_photos'}),
    ('features', [LenExtractor(), SqrtTransformer(), StandardScaler()],
         {'alias': 'n_feats'}),
    ('description', [WordCntExtractor(), SqrtTransformer(), StandardScaler()],
         {'alias': 'descr_wcnt'}),
], input_df=True, df_out=True)

aggr_mapper = DataFrameMapper([
    ('manager_id', [GroupSumExtractor()],
         {'alias': 'n_posts'}),
    ('building_id', [GroupSumExtractor()],
         {'alias': 'n_buildings'}),
], input_df=True, df_out=True)

scl_aggr_mapper = DataFrameMapper([
    ('manager_id', [GroupSumExtractor(), LogTransformer(), StandardScaler()],
         {'alias': 'n_posts'}),
    ('building_id', [GroupSumExtractor(), LogTransformer(), StandardScaler()],
         {'alias': 'n_buildings'}),
], input_df=True, df_out=True)


date_mapper = DataFrameMapper([
#     ('created', WeekendExtractor(),
#         {'alias': 'is_weekend'}),
    ('created', DayBinarizer()),
], input_df=True, df_out=True)

base_pipe = Pipeline([
#     ('price_outl_drop', PriceOutlierDropper()),
#     ('bb_imputer', BedBathImputer()),
    ('ll_imputer', LatLongImputer()),
])

mapper = combine_mappers(
    [start_mapper, date_mapper, len_mapper, aggr_mapper]
)

pipe = deepcopy(base_pipe)
pipe.steps.append(('mapper', mapper))


scl_mapper = combine_mappers(
    [scl_start_mapper, date_mapper, scl_len_mapper, scl_aggr_mapper]
)

scl_pipe = deepcopy(base_pipe)
scl_pipe.steps.append(('scl_mapper', scl_mapper))
