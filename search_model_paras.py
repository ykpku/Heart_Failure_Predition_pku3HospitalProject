from copy import deepcopy

from nabular_model import nabular_deep_models
from nabular_model import nabular_deep_models_multi_task
from file_utility import __write_search_results


def search_nabular(categorical_features, ys, y_names, trainset, save_logs=False):
    '''
    n_d=64,n_a=64,n_steps=3,gamma=1.3,cat_idxs=cat_idxs,cat_dims=cat_dims,cat_emb_dim=1,n_independent=2,n_shared=4,
    momentum=0.02,lambda_sparse=1e-3,optimizer_fn=torch.optim.Adam,optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size": 50,  # how to use learning rate scheduler "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR, mask_type='entmax',  # "entmax" device_name='cuda'

    eval_metric=['auc'], max_epochs=1000, patience=30, batch_size=64, virtual_batch_size=64, num_workers=0, weights=1, drop_last=False
    '''
    real_spaces = {'feature_select': [0, 30, 50, 100, 200],
                   'patience': {0, 15, 30, 50, 100},  # {0, 15, 30, 50, 100}
                   'batch': {32, 64, 128, 256, 512, 1024},
                   'imb': {0, 1, 2},
                   'momentum': {0.01, 0.02, 0.1, 0.2},
                   'na_nd': {8, 16, 32, 64},
                   'n_steps': {3, 5, 10},
                   'cat_emb': {1, 2, 3, 4},
                   'n_indent': {1, 2, 5},
                   'n_shame': {1, 4, 5}}

    args_spaces = {'feature_select': {0, 10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210},
                   'patience': {50},
                   'batch': {256},
                   'imb': {0},
                   'momentum': {0.2},
                   'na_nd': {8},
                   'n_steps': {5},
                   'cat_emb': {4},
                   'n_indent': {1},
                   'n_shame': {1}}
    file_name_origin = 'SearchFeatures_fs_oldparams_correctfeature'

    for i in range(len(ys)):
        ys_i = [ys[i]]
        y_names_i = [y_names[i]]
        file_name = file_name_origin + "_" + y_names_i[0]
        args = {}
        results_max = []
        args_max = []
        for args_feat in args_spaces['feature_select']:
            args['feature_select'] = args_feat
            for args_patience in args_spaces['patience']:
                args['patience'] = args_patience
                for args_batch in args_spaces['batch']:
                    args['batch'] = args_batch
                    for args_imb in args_spaces['imb']:
                        args['imb'] = args_imb
                        for args_mom in args_spaces['momentum']:
                            args['momentum'] = args_mom
                            for args_na in args_spaces['na_nd']:
                                args['na_nd'] = args_na
                                for args_nst in args_spaces['n_steps']:
                                    args['n_steps'] = args_nst
                                    for args_c in args_spaces['cat_emb']:
                                        args['cat_emb'] = args_c
                                        for args_ni in args_spaces['n_indent']:
                                            args['n_indent'] = args_ni
                                            for args_nsh in args_spaces['n_shame']:
                                                args['n_shame'] = args_nsh

                                                results, _, _ = nabular_deep_models(categorical_features, ys_i, y_names_i, trainset, args)
                                                if len(results_max) == 0:
                                                    item = results[0]
                                                    results_max.append(deepcopy(item))
                                                    args_max.append(deepcopy(args))
                                                    __write_search_results("search_nabular_" + file_name + ".csv",
                                                                           y_names_i,
                                                                           args_max,
                                                                           results_max)
                                                else:
                                                    update = False
                                                    item = results[0]
                                                    if item['auc'] > results_max[0]['auc']:
                                                        results_max[0] = deepcopy(item)
                                                        args_max[0] = deepcopy(args)
                                                        update = True
                                                    if update:
                                                        __write_search_results("search_nabular_" + file_name + ".csv",
                                                                               y_names_i,
                                                                               args_max,
                                                                               results_max)
                                                if save_logs:
                                                    __write_search_results("search_nabular_" + file_name + "_logs.csv",
                                                                           y_names_i,
                                                                           [args],
                                                                           results,
                                                                           wm='a')


def search_nabular_multitasks(categorical_features, ys, y_names, train):
    args_spaces = {'feature_select': [0, 30, 50, 100, 200],
                   'patience': {0, 15, 30, 50, 100},  # {0, 15, 30, 50, 100}
                   'batch': {32, 64, 128, 256, 512, 1024},
                   'imb': {0, 1, 2},
                   'momentum': {0.01, 0.1, 0.2, 0.3},
                   'na_nd': {8, 16, 32, 64},
                   'n_steps': {3, 5, 10},
                   'cat_emb': {1, 2, 3, 4},
                   'n_indent': {1, 3, 5},
                   'n_shame': {1, 3, 5}}
    args = {'feature_select': 0, 'patience': 30, 'batch': 1024, 'n_steps': 3, 'n_indent': 2, 'n_shame': 4, 'imb': 0}
    file_name = 'multitask' + '_patience_' + str(args['patience']) + '_batch_' \
                + str(args['batch']) + '_nsteps_' + str(args['n_steps']) + '_nindent_' + str(args['n_indent']) \
                + '_nshame_' + str(args['n_shame'])
    results_max = []
    args_max = []
    for args_m in args_spaces['momentum']:
        args['momentum'] = args_m
        for args_na in args_spaces['na_nd']:
            args['na_nd'] = args_na
            for args_c in args_spaces['cat_emb']:
                args['cat_emb'] = args_c

                results = nabular_deep_models_multi_task(categorical_features, ys, y_names, train, args)
                if len(results_max) == 0:
                    for item_i, item in enumerate(results):
                        results_max.append(item)
                        args_max.append(args)
                else:
                    for item_i, item in enumerate(results):
                        if item['auc'] > results_max[item_i]['auc']:
                            results_max[item_i] = item
                            args_max[item_i] = args
    with open("search_nabular_" + file_name + ".csv", 'a', encoding='utf-8') as f:
        for i in range(len(y_names)):
            f.write("-------------" + y_names[i] + "-------------\n")
            for ak in args_max[i].keys():
                f.write(ak + " : " + str(args_max[i][ak]) + "\t")
            f.write("\n")
            for rk in results_max[i].keys():
                f.write(rk + " : " + str(results_max[i][rk]) + "\t")
            f.write("\n")
