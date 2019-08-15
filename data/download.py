import argparse
from os.path import join
import os, requests
from tqdm import tqdm

links_list = {}
links_list[('collections', 'score')] = [
        ('collections/data/model_scores/scores_test.bin', 'https://www.dropbox.com/s/0ljpai1t00rap3v/scores_test.bin?dl=1'),
        ('collections/data/model_scores/scores_train.bin', 'https://www.dropbox.com/s/6e4eoatmiyz2asf/scores_train.bin?dl=1'),
        ('collections/data/model_scores/groundtruth.bin', 'https://www.dropbox.com/s/cszb9zt7j64hm58/groundtruth.bin?dl=1')
        ]

links_list[('collections', 'model')] = [
        ('collections/model.bin', 'https://www.dropbox.com/s/iodg7lvtq89ousk/model.bin?dl=1'),
        ('collections/data/user_features_test.fvecs', 'https://www.dropbox.com/s/qt65bfcbmy1l7xy/user_features_test.fvecs?dl=1'),
        ('collections/data/item_features.fvecs', 'https://www.dropbox.com/s/yetngpvpv6t7ben/item_features.fvecs?dl=1'),
        ('collections/data/user_features_list.txt', 'https://www.dropbox.com/s/sf11ewcv2igtip9/user_features_list.txt?dl=1'),
        ('collections/data/user_features_train.fvecs', 'https://www.dropbox.com/s/jjjrqy01u8kucez/user_features_train.fvecs?dl=1'),
        ('collections/data/item_features_list.txt', 'https://www.dropbox.com/s/xo9h31nk0e8psqq/item_features_list.txt?dl=1'),
        ('collections/data/pairwise/UserBoardPositiveVectorB_train.fvecs', 'https://www.dropbox.com/s/07r2tg1bz7cr5uc/UserBoardPositiveVectorB_train.fvecs?dl=1'),
        ('collections/data/pairwise/CardImagePositive1dImageVisual_train.fvecs', 'https://www.dropbox.com/s/ulb3v7v73mfxrz5/CardImagePositive1dImageVisual_train.fvecs?dl=1'),
        ('collections/data/pairwise/CardComplainImageT2T_test.fvecs', 'https://www.dropbox.com/s/36j6qvi8z0mgvvk/CardComplainImageT2T_test.fvecs?dl=1'),
        ('collections/data/pairwise/CardImagePositive7dImageT2T_test.fvecs', 'https://www.dropbox.com/s/32fw0i7uksyk52i/CardImagePositive7dImageT2T_test.fvecs?dl=1'),
        ('collections/data/pairwise/BoardVisualPositive21dVSImageVisual_test.fvecs', 'https://www.dropbox.com/s/ih4sonm5odtu8rj/BoardVisualPositive21dVSImageVisual_test.fvecs?dl=1'),
        ('collections/data/pairwise/CounterVSObjectId__CT_BoardSubscription#OT_Board#RT_Sum_30d_train.fvecs', 'https://www.dropbox.com/s/umem8ie0ltyplqy/CounterVSObjectId__CT_BoardSubscription%23OT_Board%23RT_Sum_30d_train.fvecs?dl=1'),
        ('collections/data/pairwise/UserCardTextHistoryDssm_train.fvecs', 'https://www.dropbox.com/s/5cedhiw2aemghr5/UserCardTextHistoryDssm_train.fvecs?dl=1'),
        ('collections/data/pairwise/CardLikeImageVisual_train.fvecs', 'https://www.dropbox.com/s/po7s4uig61mauc6/CardLikeImageVisual_train.fvecs?dl=1'),
        ('collections/data/pairwise/CardImagePositive21dImageVisual_train.fvecs', 'https://www.dropbox.com/s/3fgh9lctz1ih382/CardImagePositive21dImageVisual_train.fvecs?dl=1'),
        ('collections/data/pairwise/CardImagePositive1dImageVisual_test.fvecs', 'https://www.dropbox.com/s/1j7w21svlo4yjqq/CardImagePositive1dImageVisual_test.fvecs?dl=1'),
        ('collections/data/pairwise/CounterVSObjectId__CT_BoardSubscription#OT_Board#RT_Sum_30d_test.fvecs', 'https://www.dropbox.com/s/9wp5wnwazttih3b/CounterVSObjectId__CT_BoardSubscription%23OT_Board%23RT_Sum_30d_test.fvecs?dl=1'),
        ('collections/data/pairwise/UserImageBroadThemeT2TPositive_train.fvecs', 'https://www.dropbox.com/s/1won8245hj4te3f/UserImageBroadThemeT2TPositive_train.fvecs?dl=1'),
        ('collections/data/pairwise/UserBoardCardPositives_test.fvecs', 'https://www.dropbox.com/s/us4ibybfzxti2c5/UserBoardCardPositives_test.fvecs?dl=1'),
        ('collections/data/pairwise/UserImageBroadThemeT2TPositive_test.fvecs', 'https://www.dropbox.com/s/7wayl8aev60pfid/UserImageBroadThemeT2TPositive_test.fvecs?dl=1'),
        ('collections/data/pairwise/CardShareImageVisual_test.fvecs', 'https://www.dropbox.com/s/uk1ryblvwfee1qw/CardShareImageVisual_test.fvecs?dl=1'),
        ('collections/data/pairwise/CounterVSObjectId__CT_BoardClick#OT_Board#RT_Sum_21d_train.fvecs', 'https://www.dropbox.com/s/743b8e2dvdruzpt/CounterVSObjectId__CT_BoardClick%23OT_Board%23RT_Sum_21d_train.fvecs?dl=1'),
        ('collections/data/pairwise/ImageT2TPositive7dVSImageT2T_train.fvecs', 'https://www.dropbox.com/s/0kpcy5sy4fg9ust/ImageT2TPositive7dVSImageT2T_train.fvecs?dl=1'),
        ('collections/data/pairwise/UserBoardCardPositives_train.fvecs', 'https://www.dropbox.com/s/d8niaq3x0npq22u/UserBoardCardPositives_train.fvecs?dl=1'),
        ('collections/data/pairwise/CardImagePositive21dImageVisual_test.fvecs', 'https://www.dropbox.com/s/r55jrodptl22um8/CardImagePositive21dImageVisual_test.fvecs?dl=1'),
        ('collections/data/pairwise/CardShareImageT2T_test.fvecs', 'https://www.dropbox.com/s/9dv4p23gn5ize5v/CardShareImageT2T_test.fvecs?dl=1'),
        ('collections/data/pairwise/UserBoardCardClicks_train.fvecs', 'https://www.dropbox.com/s/ayhyszqb3eet3dv/UserBoardCardClicks_train.fvecs?dl=1'),
        ('collections/data/pairwise/CardClickImageVisual_train.fvecs', 'https://www.dropbox.com/s/5ms7a3x78e3v9bj/CardClickImageVisual_train.fvecs?dl=1'),
        ('collections/data/pairwise/CardShareImageVisual_train.fvecs', 'https://www.dropbox.com/s/686kagywb9jp9yi/CardShareImageVisual_train.fvecs?dl=1'),
        ('collections/data/pairwise/UserBoardPositiveVectorB_test.fvecs', 'https://www.dropbox.com/s/s99mla7kjcv2pzc/UserBoardPositiveVectorB_test.fvecs?dl=1'),
        ('collections/data/pairwise/CardLikeImageT2T_train.fvecs', 'https://www.dropbox.com/s/pd9p4mmwqpayfwj/CardLikeImageT2T_train.fvecs?dl=1'),
        ('collections/data/pairwise/UserChannelPositiveVectorB_test.fvecs', 'https://www.dropbox.com/s/4ne3aldholq7hir/UserChannelPositiveVectorB_test.fvecs?dl=1'),
        ('collections/data/pairwise/UserImageBroadThemeVisualPositive_train.fvecs', 'https://www.dropbox.com/s/6il59ele757a6eu/UserImageBroadThemeVisualPositive_train.fvecs?dl=1'),
        ('collections/data/pairwise/CardShareImageT2T_train.fvecs', 'https://www.dropbox.com/s/uiyxpg2lnuf0vjk/CardShareImageT2T_train.fvecs?dl=1'),
        ('collections/data/pairwise/CardAddImageT2T_train.fvecs', 'https://www.dropbox.com/s/1wp06pem6mlcz0m/CardAddImageT2T_train.fvecs?dl=1'),
        ('collections/data/pairwise/UserBoardCardClicks_test.fvecs', 'https://www.dropbox.com/s/l6lytn3lmhlsyai/UserBoardCardClicks_test.fvecs?dl=1'),
        ('collections/data/pairwise/UserBoardPositiveVectorM_test.fvecs', 'https://www.dropbox.com/s/6ggh2lkkc4ajinr/UserBoardPositiveVectorM_test.fvecs?dl=1'),
        ('collections/data/pairwise/UserChannelPositiveVectorB_train.fvecs', 'https://www.dropbox.com/s/3tn72hrfia6w72u/UserChannelPositiveVectorB_train.fvecs?dl=1'),
        ('collections/data/pairwise/BoardVisualPositive1dVSImageVisual_train.fvecs', 'https://www.dropbox.com/s/kfq7ukjogfg7nmg/BoardVisualPositive1dVSImageVisual_train.fvecs?dl=1'),
        ('collections/data/pairwise/CardAddImageVisual_test.fvecs', 'https://www.dropbox.com/s/uutnsi1boijxbsy/CardAddImageVisual_test.fvecs?dl=1'),
        ('collections/data/pairwise/CardImagePositive7dImageT2T_train.fvecs', 'https://www.dropbox.com/s/k2e41j4xsmuvnri/CardImagePositive7dImageT2T_train.fvecs?dl=1'),
        ('collections/data/pairwise/UserBoardPositiveVectorM_train.fvecs', 'https://www.dropbox.com/s/sa0whyrw2b8w8u3/UserBoardPositiveVectorM_train.fvecs?dl=1'),
        ('collections/data/pairwise/UserChannelCardViews_test.fvecs', 'https://www.dropbox.com/s/0uhv092y60hicga/UserChannelCardViews_test.fvecs?dl=1'),
        ('collections/data/pairwise/UserChannelCardViews_train.fvecs', 'https://www.dropbox.com/s/asubbcubjpelwtw/UserChannelCardViews_train.fvecs?dl=1'),
        ('collections/data/pairwise/UserBoardCardViews_train.fvecs', 'https://www.dropbox.com/s/0lyp4hrlzc2lxav/UserBoardCardViews_train.fvecs?dl=1'),
        ('collections/data/pairwise/UserCardTextHistoryDssm_test.fvecs', 'https://www.dropbox.com/s/w2zyhzzjyi4yhds/UserCardTextHistoryDssm_test.fvecs?dl=1'),
        ('collections/data/pairwise/UserBoardCardViews_test.fvecs', 'https://www.dropbox.com/s/obpc73qvtgs04h3/UserBoardCardViews_test.fvecs?dl=1'),
        ('collections/data/pairwise/ImageT2TPositive7dVSImageT2T_test.fvecs', 'https://www.dropbox.com/s/lykz1krhvzzugja/ImageT2TPositive7dVSImageT2T_test.fvecs?dl=1'),
        ('collections/data/pairwise/CardClickImageVisual_test.fvecs', 'https://www.dropbox.com/s/1fzgjfycpe6gid6/CardClickImageVisual_test.fvecs?dl=1'),
        ('collections/data/pairwise/UserChannelPositiveVectorM_test.fvecs', 'https://www.dropbox.com/s/7mbvkunym4x1ixj/UserChannelPositiveVectorM_test.fvecs?dl=1'),
        ('collections/data/pairwise/CardComplainImageVisual_train.fvecs', 'https://www.dropbox.com/s/88welaeu61x2bvf/CardComplainImageVisual_train.fvecs?dl=1'),
        ('collections/data/pairwise/CardLikeImageT2T_test.fvecs', 'https://www.dropbox.com/s/g6v10hvuzq08hah/CardLikeImageT2T_test.fvecs?dl=1'),
        ('collections/data/pairwise/BoardVisualPositive1dVSImageVisual_test.fvecs', 'https://www.dropbox.com/s/yubb0gmmlabhexf/BoardVisualPositive1dVSImageVisual_test.fvecs?dl=1'),
        ('collections/data/pairwise/CardLikeImageVisual_test.fvecs', 'https://www.dropbox.com/s/c22ylzjppedn30o/CardLikeImageVisual_test.fvecs?dl=1'),
        ('collections/data/pairwise/CardAddImageT2T_test.fvecs', 'https://www.dropbox.com/s/b8s9eyje0k2cgsj/CardAddImageT2T_test.fvecs?dl=1'),
        ('collections/data/pairwise/UserImageBroadThemeVisualPositive_test.fvecs', 'https://www.dropbox.com/s/yskg4wg3ptck19w/UserImageBroadThemeVisualPositive_test.fvecs?dl=1'),
        ('collections/data/pairwise/CounterVSObjectId__CT_BoardClick#OT_Board#RT_Sum_21d_test.fvecs', 'https://www.dropbox.com/s/bnevuo44jel1nhe/CounterVSObjectId__CT_BoardClick%23OT_Board%23RT_Sum_21d_test.fvecs?dl=1'),
        ('collections/data/pairwise/CardComplainImageVisual_test.fvecs', 'https://www.dropbox.com/s/c93e4xslxg6s0tj/CardComplainImageVisual_test.fvecs?dl=1'),
        ('collections/data/pairwise/BoardT2TPositive7dVSImageT2T_test.fvecs', 'https://www.dropbox.com/s/oxh6z7apw72ghlk/BoardT2TPositive7dVSImageT2T_test.fvecs?dl=1'),
        ('collections/data/pairwise/BoardT2TPositive7dVSImageT2T_train.fvecs', 'https://www.dropbox.com/s/qynqccpiw4xf5b7/BoardT2TPositive7dVSImageT2T_train.fvecs?dl=1'),
        ('collections/data/pairwise/BoardVisualPositive21dVSImageVisual_train.fvecs', 'https://www.dropbox.com/s/hsop6me8eygzdws/BoardVisualPositive21dVSImageVisual_train.fvecs?dl=1'),
        ('collections/data/pairwise/UserChannelPositiveVectorM_train.fvecs', 'https://www.dropbox.com/s/cjb2fwkkq3gsb0g/UserChannelPositiveVectorM_train.fvecs?dl=1'),
        ('collections/data/pairwise/CardAddImageVisual_train.fvecs', 'https://www.dropbox.com/s/2kncz4pegzb1ysa/CardAddImageVisual_train.fvecs?dl=1'),
        ('collections/data/pairwise/CardComplainImageT2T_train.fvecs', 'https://www.dropbox.com/s/4qknn8wcdplw7vw/CardComplainImageT2T_train.fvecs?dl=1')
        ]

links_list[('video', 'score')] = [
        ('video/data/model_scores/scores_test.bin', 'https://www.dropbox.com/s/cuvfwra5tgfzhc9/scores_test.bin?dl=1'),
        ('video/data/model_scores/scores_train.bin', 'https://www.dropbox.com/s/mnil3kacync6ou0/scores_train.bin?dl=1'),
        ('video/data/model_scores/groundtruth.bin', 'https://www.dropbox.com/s/w8bfprg1y93ixpc/groundtruth.bin?dl=1')
        ]

links_list[('video', 'model')] = [
        ('video/model.bin', 'https://www.dropbox.com/s/l2sj4nny9u7p9d3/model.bin?dl=1'),
        ('video/data/user_features_test.fvecs', 'https://www.dropbox.com/s/3ugen0i258a9odg/user_features_test.fvecs?dl=1'),
        ('video/data/item_features.fvecs', 'https://www.dropbox.com/s/l72wnxpjtib554d/item_features.fvecs?dl=1'),
        ('video/data/user_features_list.txt', 'https://www.dropbox.com/s/tlznc6w1wmnslf9/user_features_list.txt?dl=1'),
        ('video/data/user_features_train.fvecs', 'https://www.dropbox.com/s/udam6p6g37gwvi1/user_features_train.fvecs?dl=1'),
        ('video/data/item_features_list.txt', 'https://www.dropbox.com/s/bjmfzn01xknlili/item_features_list.txt?dl=1'),
        ('video/data/pairwise/deep80_view_video_cv_dot_product_7d_train.fvecs', 'https://www.dropbox.com/s/7e8n7hrfwbsl4no/deep80_view_video_cv_dot_product_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/weighted_view_video_cv_cos_7d_train.fvecs', 'https://www.dropbox.com/s/rl6jl7n73ysr4jp/weighted_view_video_cv_cos_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_views_180d_train.fvecs', 'https://www.dropbox.com/s/d2xeotu74kvwjxr/user_author_views_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep60_view_video_cv_left_norm_7d_train.fvecs', 'https://www.dropbox.com/s/uy12ibrc6imlbss/deep60_view_video_cv_left_norm_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep40_view_video_cv_cos_180d_train.fvecs', 'https://www.dropbox.com/s/5cij8bai8i4bzyt/deep40_view_video_cv_cos_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep40_view_video_cv_dot_product_7d_train.fvecs', 'https://www.dropbox.com/s/3yyp9gjbejczarp/deep40_view_video_cv_dot_product_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep80_views_7d_train.fvecs', 'https://www.dropbox.com/s/m4tv59f323pcoso/user_author_deep80_views_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep80_view_video_cv_left_norm_7d_train.fvecs', 'https://www.dropbox.com/s/qyq3mklv8teifu8/deep80_view_video_cv_left_norm_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep40_view_video_cv_cos_7d_train.fvecs', 'https://www.dropbox.com/s/jzbw2e0wt5jgkvv/deep40_view_video_cv_cos_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_view_time_180d_train.fvecs', 'https://www.dropbox.com/s/uqgugoxqriblbpd/user_author_view_time_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep60_view_video_cv_cos_180d_train.fvecs', 'https://www.dropbox.com/s/m0bvi5jbq74ulty/deep60_view_video_cv_cos_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep80_view_video_cv_cos_180d_train.fvecs', 'https://www.dropbox.com/s/ctgfm8asrewl2vq/deep80_view_video_cv_cos_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep20_view_video_cv_left_norm_7d_train.fvecs', 'https://www.dropbox.com/s/f9d7rub1co0v458/deep20_view_video_cv_left_norm_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep40_view_video_cv_left_norm_7d_train.fvecs', 'https://www.dropbox.com/s/75zj2qcrvsk92mx/deep40_view_video_cv_left_norm_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep20_views_7d_train.fvecs', 'https://www.dropbox.com/s/lr71l4eq3u9fnvz/user_author_deep20_views_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/click_video_cv_cos_7d_train.fvecs', 'https://www.dropbox.com/s/8ha52fffwmxvvma/click_video_cv_cos_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/click_video_cv_dot_product_7d_train.fvecs', 'https://www.dropbox.com/s/195c4ol5fc6x661/click_video_cv_dot_product_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep60_view_video_cv_dot_product_180d_train.fvecs', 'https://www.dropbox.com/s/se6nygsklmkz733/deep60_view_video_cv_dot_product_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep60_view_video_cv_left_norm_180d_train.fvecs', 'https://www.dropbox.com/s/j4z732rj0gzhtqo/deep60_view_video_cv_left_norm_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_view_time_7d_train.fvecs', 'https://www.dropbox.com/s/acudmq3lss9f0qh/user_author_view_time_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_weighted_views_7d_train.fvecs', 'https://www.dropbox.com/s/onwzs8g98bg68nj/user_author_weighted_views_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep60_view_video_cv_cos_7d_train.fvecs', 'https://www.dropbox.com/s/um8suk3aospj7yq/deep60_view_video_cv_cos_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/view_video_cv_cos_7d_train.fvecs', 'https://www.dropbox.com/s/jbwo4f7yes4hy60/view_video_cv_cos_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_view_time_30d_train.fvecs', 'https://www.dropbox.com/s/0s1ykuyuwpqxeoq/user_author_view_time_30d_train.fvecs?dl=1'),
        ('video/data/pairwise/total_view_video_cv_left_norm_180d_train.fvecs', 'https://www.dropbox.com/s/326axlkt40wpf3c/total_view_video_cv_left_norm_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/total_view_video_cv_cos_7d_train.fvecs', 'https://www.dropbox.com/s/h3eyj6uc5b8zov1/total_view_video_cv_cos_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/click_video_cv_left_norm_7d_train.fvecs', 'https://www.dropbox.com/s/g42gntyh333jdn6/click_video_cv_left_norm_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep20_view_video_cv_cos_7d_train.fvecs', 'https://www.dropbox.com/s/4i3kwn962q6yjjb/deep20_view_video_cv_cos_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep20_view_video_cv_left_norm_180d_train.fvecs', 'https://www.dropbox.com/s/g2ur2db93ns7l4j/deep20_view_video_cv_left_norm_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/total_view_video_cv_dot_product_7d_train.fvecs', 'https://www.dropbox.com/s/3ovfcisgah0wfaj/total_view_video_cv_dot_product_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/click_video_cv_cos_180d_train.fvecs', 'https://www.dropbox.com/s/5txbtfxtzde04io/click_video_cv_cos_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/total_view_video_cv_cos_180d_train.fvecs', 'https://www.dropbox.com/s/8rpxvhqbgsv9ll1/total_view_video_cv_cos_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep80_view_video_cv_dot_product_180d_train.fvecs', 'https://www.dropbox.com/s/j7wgbjv4dwogmc1/deep80_view_video_cv_dot_product_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_views_7d_train.fvecs', 'https://www.dropbox.com/s/4lp4b3jgylv0qyb/user_author_views_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_clicks_180d_train.fvecs', 'https://www.dropbox.com/s/abp9lh37z0ofeka/user_author_clicks_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep20_views_180d_train.fvecs', 'https://www.dropbox.com/s/cff7v54mjkheccf/user_author_deep20_views_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/view_video_cv_dot_product_180d_train.fvecs', 'https://www.dropbox.com/s/dr5hjeoqhol70h9/view_video_cv_dot_product_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/weighted_view_video_cv_left_norm_7d_train.fvecs', 'https://www.dropbox.com/s/0vs5iggz3ydgs2r/weighted_view_video_cv_left_norm_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/view_video_cv_cos_180d_train.fvecs', 'https://www.dropbox.com/s/1o5bl94igoluxqj/view_video_cv_cos_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_clicks_30d_train.fvecs', 'https://www.dropbox.com/s/q6nrxpojtvcy1ad/user_author_clicks_30d_train.fvecs?dl=1'),
        ('video/data/pairwise/view_video_cv_dot_product_7d_train.fvecs', 'https://www.dropbox.com/s/ub8t4sr5tcjpmgu/view_video_cv_dot_product_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep20_view_video_cv_dot_product_7d_train.fvecs', 'https://www.dropbox.com/s/uasr576gltm3bi0/deep20_view_video_cv_dot_product_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep20_view_video_cv_dot_product_180d_train.fvecs', 'https://www.dropbox.com/s/pjnfjwlt69zan8e/deep20_view_video_cv_dot_product_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep40_views_7d_train.fvecs', 'https://www.dropbox.com/s/u9zz09qroj0wlu3/user_author_deep40_views_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep60_views_180d_train.fvecs', 'https://www.dropbox.com/s/17lxawsz8y4f8rw/user_author_deep60_views_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep40_views_180d_train.fvecs', 'https://www.dropbox.com/s/taqoqt0h5vka5e1/user_author_deep40_views_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_clicks_7d_train.fvecs', 'https://www.dropbox.com/s/ofzh1oax3xoff10/user_author_clicks_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_weighted_views_180d_train.fvecs', 'https://www.dropbox.com/s/dducgme863yl3qe/user_author_weighted_views_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/total_view_video_cv_dot_product_180d_train.fvecs', 'https://www.dropbox.com/s/1yd1tex1m3qjmqa/total_view_video_cv_dot_product_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/weighted_view_video_cv_dot_product_7d_train.fvecs', 'https://www.dropbox.com/s/pshuv44ss4cv1wt/weighted_view_video_cv_dot_product_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/total_view_video_cv_left_norm_7d_train.fvecs', 'https://www.dropbox.com/s/dxhshqfle7qifbo/total_view_video_cv_left_norm_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/view_video_cv_left_norm_180d_train.fvecs', 'https://www.dropbox.com/s/1gamz1h2jdr30f8/view_video_cv_left_norm_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/weighted_view_video_cv_cos_180d_train.fvecs', 'https://www.dropbox.com/s/k5y7i7afe84ji08/weighted_view_video_cv_cos_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep60_views_30d_train.fvecs', 'https://www.dropbox.com/s/n5b0k43ve3si8wi/user_author_deep60_views_30d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep40_view_video_cv_dot_product_180d_train.fvecs', 'https://www.dropbox.com/s/za2mmvy69lpwnll/deep40_view_video_cv_dot_product_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/weighted_view_video_cv_left_norm_180d_train.fvecs', 'https://www.dropbox.com/s/5cuuj0iatmpqxzo/weighted_view_video_cv_left_norm_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/title_history_dssm_train.fvecs', 'https://www.dropbox.com/s/c94uzngsy4dhgpt/title_history_dssm_train.fvecs?dl=1'),
        ('video/data/pairwise/view_video_cv_left_norm_7d_train.fvecs', 'https://www.dropbox.com/s/j1gax6fxp9b5jss/view_video_cv_left_norm_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep80_view_video_cv_left_norm_180d_train.fvecs', 'https://www.dropbox.com/s/s4mo08zp7m3ediu/deep80_view_video_cv_left_norm_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/click_video_cv_left_norm_180d_train.fvecs', 'https://www.dropbox.com/s/ys2z2ad6k9yys0h/click_video_cv_left_norm_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep80_views_180d_train.fvecs', 'https://www.dropbox.com/s/0w8ric72wwdy5ti/user_author_deep80_views_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_weighted_views_30d_train.fvecs', 'https://www.dropbox.com/s/i4znwe4pfdhe5bh/user_author_weighted_views_30d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep20_view_video_cv_cos_180d_train.fvecs', 'https://www.dropbox.com/s/tho7kk2dr3ufnp6/deep20_view_video_cv_cos_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep60_view_video_cv_dot_product_7d_train.fvecs', 'https://www.dropbox.com/s/t4b7vxmsyou7xfc/deep60_view_video_cv_dot_product_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/click_video_cv_dot_product_180d_train.fvecs', 'https://www.dropbox.com/s/ls2s8mfzp305ge0/click_video_cv_dot_product_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep60_views_7d_train.fvecs', 'https://www.dropbox.com/s/zzbyqrcyifm14jm/user_author_deep60_views_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep40_view_video_cv_left_norm_180d_train.fvecs', 'https://www.dropbox.com/s/kg3vjgob6fsecqe/deep40_view_video_cv_left_norm_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep80_view_video_cv_cos_7d_train.fvecs', 'https://www.dropbox.com/s/9pk57x8xugs9htz/deep80_view_video_cv_cos_7d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep40_views_30d_train.fvecs', 'https://www.dropbox.com/s/1f99ydq77rjsy77/user_author_deep40_views_30d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep80_views_30d_train.fvecs', 'https://www.dropbox.com/s/igndapo23kvy9cu/user_author_deep80_views_30d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep20_views_30d_train.fvecs', 'https://www.dropbox.com/s/7deo3i1xwib3ijf/user_author_deep20_views_30d_train.fvecs?dl=1'),
        ('video/data/pairwise/user_author_views_30d_train.fvecs', 'https://www.dropbox.com/s/yakwl6aubh7euc4/user_author_views_30d_train.fvecs?dl=1'),
        ('video/data/pairwise/weighted_view_video_cv_dot_product_180d_train.fvecs', 'https://www.dropbox.com/s/lml3p3tq4q5b0lk/weighted_view_video_cv_dot_product_180d_train.fvecs?dl=1'),
        ('video/data/pairwise/deep80_view_video_cv_left_norm_180d_test.fvecs', 'https://www.dropbox.com/s/rb7ha1pfgxpnljb/deep80_view_video_cv_left_norm_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep60_views_30d_test.fvecs', 'https://www.dropbox.com/s/mkfyufp5czy15z6/user_author_deep60_views_30d_test.fvecs?dl=1'),
        ('video/data/pairwise/click_video_cv_left_norm_180d_test.fvecs', 'https://www.dropbox.com/s/qvxqb04dywiu8xl/click_video_cv_left_norm_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_weighted_views_30d_test.fvecs', 'https://www.dropbox.com/s/vnkquplioq71fnt/user_author_weighted_views_30d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep40_view_video_cv_dot_product_7d_test.fvecs', 'https://www.dropbox.com/s/jfwwbloo3qfo13m/deep40_view_video_cv_dot_product_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/total_view_video_cv_cos_7d_test.fvecs', 'https://www.dropbox.com/s/h3jzn9mqf8w2u1f/total_view_video_cv_cos_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/view_video_cv_cos_180d_test.fvecs', 'https://www.dropbox.com/s/nsnvpsa619n3a5w/view_video_cv_cos_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep60_view_video_cv_left_norm_7d_test.fvecs', 'https://www.dropbox.com/s/thl33vfkkwsm4tj/deep60_view_video_cv_left_norm_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/total_view_video_cv_cos_180d_test.fvecs', 'https://www.dropbox.com/s/yey5rac3cn8str1/total_view_video_cv_cos_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/click_video_cv_cos_180d_test.fvecs', 'https://www.dropbox.com/s/4l958d87qh17u6m/click_video_cv_cos_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_clicks_30d_test.fvecs', 'https://www.dropbox.com/s/i4omxanlqw8w6g0/user_author_clicks_30d_test.fvecs?dl=1'),
        ('video/data/pairwise/click_video_cv_dot_product_7d_test.fvecs', 'https://www.dropbox.com/s/rs21vxdizgq9lgc/click_video_cv_dot_product_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep20_views_7d_test.fvecs', 'https://www.dropbox.com/s/xwzlicaal5boeit/user_author_deep20_views_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_view_time_7d_test.fvecs', 'https://www.dropbox.com/s/d6w6s5l356iqhuf/user_author_view_time_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep80_view_video_cv_dot_product_180d_test.fvecs', 'https://www.dropbox.com/s/ynfytha7fgmcvdd/deep80_view_video_cv_dot_product_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_views_180d_test.fvecs', 'https://www.dropbox.com/s/jyy48pgt0tokyc1/user_author_views_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep20_view_video_cv_cos_180d_test.fvecs', 'https://www.dropbox.com/s/61w0r85mpd4tu4m/deep20_view_video_cv_cos_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/click_video_cv_cos_7d_test.fvecs', 'https://www.dropbox.com/s/kei2noogwpjnx3f/click_video_cv_cos_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep40_view_video_cv_dot_product_180d_test.fvecs', 'https://www.dropbox.com/s/j1zojgnvii2tao8/deep40_view_video_cv_dot_product_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/weighted_view_video_cv_dot_product_7d_test.fvecs', 'https://www.dropbox.com/s/9ilgy5n05oqdijd/weighted_view_video_cv_dot_product_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/click_video_cv_left_norm_7d_test.fvecs', 'https://www.dropbox.com/s/szt7hzdqbnstfhf/click_video_cv_left_norm_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep80_view_video_cv_left_norm_7d_test.fvecs', 'https://www.dropbox.com/s/jketbnzpbtnyakh/deep80_view_video_cv_left_norm_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep60_view_video_cv_cos_180d_test.fvecs', 'https://www.dropbox.com/s/q3kbk7ucx0pybs5/deep60_view_video_cv_cos_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep60_view_video_cv_dot_product_7d_test.fvecs', 'https://www.dropbox.com/s/hxnb5f0xjvcgmi7/deep60_view_video_cv_dot_product_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep20_views_30d_test.fvecs', 'https://www.dropbox.com/s/plgqvqopzkawczq/user_author_deep20_views_30d_test.fvecs?dl=1'),
        ('video/data/pairwise/weighted_view_video_cv_left_norm_7d_test.fvecs', 'https://www.dropbox.com/s/q6z3b0aypobp3i8/weighted_view_video_cv_left_norm_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep80_view_video_cv_dot_product_7d_test.fvecs', 'https://www.dropbox.com/s/97mswx7ecg4gncs/deep80_view_video_cv_dot_product_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep20_view_video_cv_dot_product_7d_test.fvecs', 'https://www.dropbox.com/s/mzaacpnrt53fzga/deep20_view_video_cv_dot_product_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep80_views_180d_test.fvecs', 'https://www.dropbox.com/s/qo5spyt379t4kfe/user_author_deep80_views_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep40_views_30d_test.fvecs', 'https://www.dropbox.com/s/wkaq0w552k0ziw8/user_author_deep40_views_30d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep60_view_video_cv_dot_product_180d_test.fvecs', 'https://www.dropbox.com/s/gr6yuroz6zshemw/deep60_view_video_cv_dot_product_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep60_views_7d_test.fvecs', 'https://www.dropbox.com/s/etyve9tr52s6kty/user_author_deep60_views_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep40_views_7d_test.fvecs', 'https://www.dropbox.com/s/nsabcz5rs34n2p6/user_author_deep40_views_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep80_views_7d_test.fvecs', 'https://www.dropbox.com/s/7xeaxl9fy4mp7jt/user_author_deep80_views_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_clicks_180d_test.fvecs', 'https://www.dropbox.com/s/pn6xwthdwtn77wc/user_author_clicks_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/weighted_view_video_cv_cos_180d_test.fvecs', 'https://www.dropbox.com/s/5bgnwgmtgk9a37y/weighted_view_video_cv_cos_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/view_video_cv_left_norm_7d_test.fvecs', 'https://www.dropbox.com/s/dkghgc76hi6mrgq/view_video_cv_left_norm_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep60_view_video_cv_left_norm_180d_test.fvecs', 'https://www.dropbox.com/s/rgnv5f11w1hzcc9/deep60_view_video_cv_left_norm_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/click_video_cv_dot_product_180d_test.fvecs', 'https://www.dropbox.com/s/9v9xabdt8axttis/click_video_cv_dot_product_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/weighted_view_video_cv_cos_7d_test.fvecs', 'https://www.dropbox.com/s/fennsuqx6yx43fr/weighted_view_video_cv_cos_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/total_view_video_cv_dot_product_180d_test.fvecs', 'https://www.dropbox.com/s/mgg0yfe2lqda2z1/total_view_video_cv_dot_product_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_weighted_views_180d_test.fvecs', 'https://www.dropbox.com/s/kosf6uny6kwkd8v/user_author_weighted_views_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep40_view_video_cv_left_norm_7d_test.fvecs', 'https://www.dropbox.com/s/iuc8l554fqlo5fu/deep40_view_video_cv_left_norm_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep20_view_video_cv_left_norm_180d_test.fvecs', 'https://www.dropbox.com/s/mcpj247k0zxyy9a/deep20_view_video_cv_left_norm_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep80_view_video_cv_cos_180d_test.fvecs', 'https://www.dropbox.com/s/rzvwgqmpcb6ym28/deep80_view_video_cv_cos_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/weighted_view_video_cv_dot_product_180d_test.fvecs', 'https://www.dropbox.com/s/fo9yc0ibwi760gw/weighted_view_video_cv_dot_product_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_clicks_7d_test.fvecs', 'https://www.dropbox.com/s/44gzcsdn7h844lz/user_author_clicks_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_views_7d_test.fvecs', 'https://www.dropbox.com/s/xl5x1i9x2idqf2s/user_author_views_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/view_video_cv_cos_7d_test.fvecs', 'https://www.dropbox.com/s/k3cztgbb1g5ofe1/view_video_cv_cos_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/total_view_video_cv_left_norm_180d_test.fvecs', 'https://www.dropbox.com/s/xtyp15oul5xptsd/total_view_video_cv_left_norm_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/weighted_view_video_cv_left_norm_180d_test.fvecs', 'https://www.dropbox.com/s/pfldxpwyqovigz5/weighted_view_video_cv_left_norm_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/view_video_cv_dot_product_7d_test.fvecs', 'https://www.dropbox.com/s/rumywi2j1snzb5r/view_video_cv_dot_product_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep60_views_180d_test.fvecs', 'https://www.dropbox.com/s/qvpuv5twgozvd52/user_author_deep60_views_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep40_view_video_cv_cos_7d_test.fvecs', 'https://www.dropbox.com/s/oeysbed3aa805xy/deep40_view_video_cv_cos_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep40_views_180d_test.fvecs', 'https://www.dropbox.com/s/uob7780flsc71ju/user_author_deep40_views_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/title_history_dssm_test.fvecs', 'https://www.dropbox.com/s/bp4k4qq3sbh17u5/title_history_dssm_test.fvecs?dl=1'),
        ('video/data/pairwise/total_view_video_cv_dot_product_7d_test.fvecs', 'https://www.dropbox.com/s/11ofl9ou6xhniy7/total_view_video_cv_dot_product_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/view_video_cv_dot_product_180d_test.fvecs', 'https://www.dropbox.com/s/sp19m5h4bjr8uaw/view_video_cv_dot_product_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep20_view_video_cv_cos_7d_test.fvecs', 'https://www.dropbox.com/s/8kl1nlylqxohsys/deep20_view_video_cv_cos_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/total_view_video_cv_left_norm_7d_test.fvecs', 'https://www.dropbox.com/s/qb85tcqwnylon0b/total_view_video_cv_left_norm_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_view_time_30d_test.fvecs', 'https://www.dropbox.com/s/xhwgr1doka5pzq7/user_author_view_time_30d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_view_time_180d_test.fvecs', 'https://www.dropbox.com/s/1gwpvp1qey715pr/user_author_view_time_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_views_30d_test.fvecs', 'https://www.dropbox.com/s/p5yyi5biz0btsit/user_author_views_30d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep20_views_180d_test.fvecs', 'https://www.dropbox.com/s/ytf8zphuu7jc14s/user_author_deep20_views_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep20_view_video_cv_dot_product_180d_test.fvecs', 'https://www.dropbox.com/s/d6c0t9686bimk19/deep20_view_video_cv_dot_product_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep20_view_video_cv_left_norm_7d_test.fvecs', 'https://www.dropbox.com/s/zza1hpo746eahhm/deep20_view_video_cv_left_norm_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep40_view_video_cv_left_norm_180d_test.fvecs', 'https://www.dropbox.com/s/5rzk7q4wobhbtac/deep40_view_video_cv_left_norm_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep40_view_video_cv_cos_180d_test.fvecs', 'https://www.dropbox.com/s/tstilf8ngemppcv/deep40_view_video_cv_cos_180d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep60_view_video_cv_cos_7d_test.fvecs', 'https://www.dropbox.com/s/hskltq26kmaalrf/deep60_view_video_cv_cos_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/deep80_view_video_cv_cos_7d_test.fvecs', 'https://www.dropbox.com/s/efq2g5iyt5v3kw2/deep80_view_video_cv_cos_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_weighted_views_7d_test.fvecs', 'https://www.dropbox.com/s/fr1fjaunux686pe/user_author_weighted_views_7d_test.fvecs?dl=1'),
        ('video/data/pairwise/user_author_deep80_views_30d_test.fvecs', 'https://www.dropbox.com/s/p3w50vmxqh0hgx6/user_author_deep80_views_30d_test.fvecs?dl=1'),
        ('video/data/pairwise/view_video_cv_left_norm_180d_test.fvecs', 'https://www.dropbox.com/s/ihebxn3bw9ir13u/view_video_cv_left_norm_180d_test.fvecs?dl=1')
        ]


def download(url, filename, delete_if_interrupted=True, chunk_size=4096):
    """ saves file from url to filename with a fancy progressbar """
    try:
        with open(filename, "wb") as f:
            print("Downloading {} > {}".format(url, filename))
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                total_length = int(total_length)
                with tqdm(total=total_length) as progressbar:
                    for data in response.iter_content(chunk_size=chunk_size):
                        if data:  # filter-out keep-alive chunks
                            f.write(data)
                            progressbar.update(len(data))
    except Exception as e:
        if delete_if_interrupted:
            print("Removing incomplete download {}.".format(filename))
            os.remove(filename)
        raise e
    return filename

def load(dataset, mode):
    if mode == 'score':
        data_path = join(dataset, 'data/model_scores/')
    elif mode == 'model':
        data_path = join(dataset, 'data/pairwise/')

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    for filename, link in links_list[(dataset, mode)]:
        download(link, filename)


parser = argparse.ArgumentParser()
parser.add_argument('dataset', metavar='dataset', type=str, choices=['collections', 'video', 'all'],
                           help='use \'collections\' to download Collections dataset, \'video\' to download Video dataset and \'all\' to download both')
parser.add_argument('mode', metavar='mode', type=str, choices=['score', 'model', 'all'],
                           help='use \'score\' to download only model scores (about 8 Gb per dataset), \'model\' to download the whole model data (~1 Tb) and \'all\' to download both')

args = parser.parse_args()

if args.dataset == 'all':
    datasets = ['collections', 'video']
else:
    datasets = [args.dataset]
if args.mode == 'all':
    modes = ['score', 'model']
else:
    modes = [args.mode]

for dataset in datasets:
    for mode in modes:
        load(dataset, mode)

