from feature_engineering import new_feature_engineering
from base_models import base_singletask_models


def get_variable_statistic():
    Y_names_o = ['此次发病至离院过程中是否合并AKI', '本次就诊后7天是否存活', '离院转归']
    features_o, categorical_features_o, value_features_o, cat_onehot, Ys_o, raw_features_o = new_feature_engineering(
        'origin_data/trainset.csv', label_space=Y_names_o)
    print(len(categorical_features_o) + len(value_features_o), len(list(raw_features_o.columns)))
    print(categorical_features_o)
    print(value_features_o)


if __name__ == '__main__':
    # task 1 : base models and nabular deep model
    # features_o, categorical_features_o, value_features_o, categories_o, Ys_o, raw_features_o = feature_engineering(
    #     'trainset.csv')
    # Y_names_o = ['此次发病至离院过程中是否合并AKI', '本次就诊后7天是否存活', '本次就诊后30天是否存活', '本次就诊后30天内是否再入急诊']
    # base_singletask_models(features_o, Ys_o, Y_names_o)
    # base_multitask_models(features_o, Ys_o, Y_names_o)
    # nabular_deep_models(categorical_features_o, Ys_o, Y_names_o, raw_features_o)
    # nabular_deep_models_multi_task(features_o, categorical_features_o, value_features_o, categories_o, Ys_o, Y_names_o, raw_features_o)

    # task 2 : search nabular deep model parameters
    # search_nabular(categorical_features_o, Ys_o, Y_names_o, raw_features_o)
    # search_nabular_multitasks(categorical_features_o, Ys_o, Y_names_o, raw_features_o)

    # task 3 : find influent features
    # nabular_deep_models_factors(categorical_features_o, Ys_o, Y_names_o, raw_features_o, 30)
    # analysis_factors(30)

    # task 4 : new label Y1
    # features_o, categorical_features_o, value_features_o, categories_o, Ys_o, raw_features_o = feature_engineering('trainset.csv', label_space=["离院转归"])
    # Y_names_o = ['离院转归']
    # search_nabular(categorical_features_o, Ys_o, Y_names_o, raw_features_o)
    # features_o, categorical_features_o, value_features_o, categories_o, Ys_o, raw_features_o = feature_engineering(
    #     'origin_data/trainset.csv', label_space=["离院转归"])
    # Y_names_o = ['离院转归']
    # nabular_deep_models_factors(categorical_features_o, Ys_o, Y_names_o, raw_features_o, 30)
    # analysis_factors(30)

    # task 5 : new features
    # Y_names_o = ['此次发病至离院过程中是否合并AKI', '本次就诊后7天是否存活', '本次就诊后30天是否存活', '本次就诊后30天内是否再入急诊', '离院转归']
    # features_o, categorical_features_o, value_features_o, cat_onehot, Ys_o, raw_features_o = new_feature_engineering('origin_data/trainset.csv', label_space=Y_names_o)
    # print(list(cat_onehot.get_feature_names_out(categorical_features_o)) + value_features_o)
    # print(len(list(cat_onehot.get_feature_names_out(categorical_features_o)) + value_features_o))
    # search_nabular(categorical_features_o, Ys_o, Y_names_o, raw_features_o)
    # search the feature size seperately
    # Y_names_o = ['此次发病至离院过程中是否合并AKI', '本次就诊后7天是否存活', '离院转归']
    # features_o, categorical_features_o, value_features_o, cat_onehot, Ys_o, raw_features_o = new_feature_engineering(
    #     'origin_data/trainset.csv', label_space=Y_names_o)
    # search_nabular(categorical_features_o, Ys_o, Y_names_o, raw_features_o, True)
    # nabular_deep_models_factors(categorical_features_o, Ys_o, Y_names_o, raw_features_o, 50)

    # task 6 : split features
    # __split_val_cat(1, 2, 2, 2)

    # task 7 : statistic for variables
    # get_variable_statistic()

    # taks 8: get final results and draw tables or graph
    # 8.1 get several results of baseline models
    Y_names_o = ['此次发病至离院过程中是否合并AKI', '本次就诊后7天是否存活', '离院转归']
    features_o, categorical_features_o, value_features_o, cat_onehot, Ys_o, raw_features_o = new_feature_engineering('origin_data/trainset.csv', label_space=Y_names_o)
    base_singletask_models(features_o, Ys_o, Y_names_o)
    pass

