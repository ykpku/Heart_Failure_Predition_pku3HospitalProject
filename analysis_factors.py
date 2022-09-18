from copy import deepcopy
from nabular_model import nabular_deep_models


def nabular_deep_models_factors(categorical_features, ys, y_names, train, top_k):
    args_re = [{'feature_select': 50,  # 原本30，取消feature select
                'patience': 50,  # {0, 15, 30, 50, 100}
                'batch': 256,
                'imb': 0,
                'momentum': 0.2,
                'na_nd': 8,
                'n_steps': 5,
                'cat_emb': 4,
                'n_indent': 1,
                'n_shame': 1},
               {'feature_select': 30,  # 原本30，取消feature select
                'patience': 50,  # {0, 15, 30, 50, 100}
                'batch': 512,
                'imb': 0,
                'momentum': 0.1,
                'na_nd': 8,
                'n_steps': 10,
                'cat_emb': 1,
                'n_indent': 5,
                'n_shame': 1},
               # {'feature_select': 0,  # 原本200，取消feature select
               #  'patience': 100,  # {0, 15, 30, 50, 100}
               #  'batch': 64,
               #  'imb': 1,
               #  'momentum': 0.01,
               #  'na_nd': 16,
               #  'n_steps': 10,
               #  'cat_emb': 1,
               #  'n_indent': 1,
               #  'n_shame': 1},
               # {'feature_select': 0,
               #  'patience': 100,  # {0, 15, 30, 50, 100}
               #  'batch': 1024,
               #  'imb': 1,
               #  'momentum': 0.01,
               #  'na_nd': 16,
               #  'n_steps': 10,
               #  'cat_emb': 1,
               #  'n_indent': 1,
               #  'n_shame': 1},
               {'feature_select': 30,  # 原本200，取消feature select
                'patience': 50,  # {0, 15, 30, 50, 100}
                'batch': 256,
                'imb': 0,
                'momentum': 0.2,
                'na_nd': 8,
                'n_steps': 5,
                'cat_emb': 4,
                'n_indent': 1,
                'n_shame': 1}
               ]

    for i in range(len(ys)):
        print("+++++++++++++++++++", y_names[i], "+++++++++++++++++++")
        results, models, features_names = nabular_deep_models(categorical_features, [ys[i]], [y_names[i]], train, args_re[i])
        # print(models[0].feature_importances_)
        feature_name_importance = {}
        for iter_i, ni in enumerate(features_names[0]):
            feature_name_importance[ni] = models[0].feature_importances_[iter_i]
        fni = sorted(feature_name_importance.items(), key=lambda d: d[1], reverse=True)[:top_k]
        print(fni)
        with open("nabular_factors_" + str(top_k) + "_20220520_3.csv", 'a', encoding='utf-8') as f:
            f.write("-------------" + y_names[i] + "-------------\n")
            for item in fni:
                f.write(item[0] + " : " + str(item[1]) + "\n")


def analysis_factors(top_k):
    origin_factors = []
    temp_factors = {}
    with open("nabular_factors_" + str(top_k) + ".csv", 'r', encoding='utf-8') as f:
        for i in f.readlines():
            if i.startswith("--"):
                if len(temp_factors) != 0:
                    origin_factors.append(deepcopy(temp_factors))
                temp_factors = {}
            else:
                line_sp = i.split((" : "))
                name = line_sp[0]
                value = float(line_sp[1])
                temp_factors[name] = value
        origin_factors.append(deepcopy(temp_factors))

    print(origin_factors)
    factor_counts = {}
    for ofs in origin_factors:
        for fks in ofs.keys():
            if fks in factor_counts:
                factor_counts[fks] += 1
            else:
                factor_counts[fks] = 1

    print(sorted(factor_counts.items(), key=lambda d: d[1], reverse=True))
