#
# scl_start_mapper = DataFrameMapper([
#     (['price'], [LogTransformer(), StandardScaler()]),
#     (['bathrooms'], StandardScaler()),
#     (['bedrooms'], StandardScaler()),
#     (['latitude'], StandardScaler()),
#     (['longitude'], StandardScaler()),
#
# ], input_df=True, df_out=True)
#
#
# scl_len_mapper = DataFrameMapper([
#     ('photos', [LenExtractor(), SqrtTransformer(), StandardScaler()],
#          {'alias': 'n_photos'}),
#     ('features', [LenExtractor(), SqrtTransformer(), StandardScaler()],
#          {'alias': 'n_feats'}),
#     ('description', [WordCntExtractor(), SqrtTransformer(), StandardScaler()],
#          {'alias': 'descr_wcnt'}),
# ], input_df=True, df_out=True)
#
#
# scl_aggr_mapper = DataFrameMapper([
#     ('manager_id', [GroupSumExtractor(), LogTransformer(), StandardScaler()],
#          {'alias': 'n_posts'}),
#     ('building_id', [GroupSumExtractor(), LogTransformer(), StandardScaler()],
#          {'alias': 'n_buildings'}),
# ], input_df=True, df_out=True)
#
#
# date_mapper = DataFrameMapper([
# #     ('created', WeekendExtractor(),
# #         {'alias': 'is_weekend'}),
#     ('created', DayBinarizer()),
# ], input_df=True, df_out=True)
#
# base_pipe = Pipeline([
# #     ('price_outl_drop', PriceOutlierDropper()),
#     ('bb_imputer', BedBathImputer()),
# #     ('ll_imputer', LatLongImputer()),
# ])
#
# mapper = merge_mappers(
#     [start_mapper, len_mapper, aggr_mapper]
# )
#
# pipe = deepcopy(base_pipe)
# pipe.steps.append(('mapper', mapper))
#
#
# scl_mapper = merge_mappers(
#     [scl_start_mapper, scl_len_mapper, scl_aggr_mapper]
# )
#
# scl_pipe = deepcopy(base_pipe)
# scl_pipe.steps.append(('scl_mapper', scl_mapper))
#
# with warnings.catch_warnings():
#     warnings.simplefilter('ignore')
#     dat1 = scl_pipe.fit_transform(df)
#
#
#
#
# pipeline = Pipeline([
#     ('union', FeatureUnion([
#         ('basic', make_pipeline(
#             FeatureUnion([
#                 ('price', make_pipeline(
#                     ItemSelector(['price']), LogTransformer())),
#                 ('rooms', make_pipeline(
#                     ItemSelector(['bathrooms', 'bedrooms']), BedBathImputer())),
#                 ('geo_coords', make_pipeline(
#                     ItemSelector(['latitude', 'longitude'])))
#             ]),
#             StandardScaler()
#         )),
#         ('created', make_pipeline(
#             FeatureUnion([
#                 ('list_lens', make_pipeline(
#                     ItemSelector(['photos', 'features']), LenExtractor())),
#                 ('word_cnt', make_pipeline(
#                     ItemSelector(['description']), WordCntExtractor())),
#             ]),
#             SqrtTransformer(),
#             StandardScaler()
#         )),
#         ('aggregate', make_pipeline(
#             FeatureUnion([
#                 ('n_posts', make_pipeline(
#                     ItemSelector('manager_id'), GroupSumExtractor())),
#                 ('building_activity', make_pipeline(
#                     ItemSelector('building_id'), GroupSumExtractor()))
#             ]),
#             LogTransformer(),
#             StandardScaler()
#         ))
#     ])),
# ])
#
# dat2 = pipeline.fit_transform(df)
#
# assert np.allclose(dat1.values, dat2)