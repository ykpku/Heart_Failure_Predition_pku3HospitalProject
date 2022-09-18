import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
import torch

from metrics import __metrics
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def nabular_deep_models(categorical_features, ys, y_names, train, args=None):
    if args is None:
        args = {'feature_select': 0,
                'patience': 30,
                'batch': 1024,
                'imb': 0,
                'momentum': 0.2,
                'na_nd': 8,
                'n_steps': 3,
                'cat_emb': 1,
                'n_indent': 2,
                'n_shame': 2}

    np.random.seed(0)
    cat_idxs, cat_dims = [], []

    models = []
    features_list = []
    task_results = []
    for i in range(len(ys)):
        categorical_columns = []
        categorical_dims = {}
        train_cp = train.copy()
        for col in train.columns:
            if col in categorical_features:
                # print(col, train[col].nunique())
                l_enc = preprocessing.LabelEncoder()
                train_cp[col] = l_enc.fit_transform(train[col].values)
                categorical_columns.append(col)
                categorical_dims[col] = len(l_enc.classes_)
        train = train_cp

        if args['feature_select'] != 0:
            #  ------------- feature selection  -------------
            select_f = SelectKBest(mutual_info_classif, k=args['feature_select'])
            train_values = select_f.fit_transform(train, ys[i])
            train_index = select_f.get_support(indices=True)
            train_names = train.columns[train_index]
            train = pd.DataFrame(train_values, columns=train_names)

        features = [col for col in train.columns]
        cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
        cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
        # print(features)
        features_list.append(features)
        model = TabNetClassifier(n_d=args['na_nd'],
                                 n_a=args['na_nd'],
                                 n_steps=args['n_steps'],
                                 gamma=1.3,
                                 cat_idxs=cat_idxs,
                                 cat_dims=cat_dims,
                                 cat_emb_dim=args['cat_emb'],
                                 n_independent=args['n_indent'],
                                 n_shared=args['n_shame'],
                                 momentum=args['momentum'],
                                 lambda_sparse=1e-3,
                                 optimizer_fn=torch.optim.Adam,
                                 optimizer_params=dict(lr=2e-2),
                                 scheduler_params={"step_size": 50,  # how to use learning rate scheduler
                                                   "gamma": 0.9},
                                 scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                 mask_type='entmax',  # "entmax"
                                 device_name='cuda'
                                 )
        models.append(model)
        x_train, x_test, y_train, y_test = train_test_split(train.values, ys[i], train_size=0.7, random_state=0)

        if args['imb'] == 0:
            pass
        elif args['imb'] == 1:
            #  ------------- imbalance sampling  -------------
            from imblearn.combine import SMOTETomek, SMOTEENN
            smote_tomek = SMOTETomek(random_state=0)
            x_train, y_train = smote_tomek.fit_resample(x_train, y_train)
        else:
            #  ------------- imbalance sampling  -------------
            from imblearn.combine import SMOTETomek, SMOTEENN
            smote_tomek = SMOTEENN(random_state=0)
            x_train, y_train = smote_tomek.fit_resample(x_train, y_train)

        print("-------------", y_names[i], "-------------")
        # unsupervised_model = TabNetPretrainer(
        #     optimizer_fn=torch.optim.Adam,
        #     optimizer_params=dict(lr=2e-2),
        #     mask_type='entmax'  # "sparsemax"
        # )
        # unsupervised_model.fit(
        #     X_train=np.array(x_train, dtype='float64'),
        #     eval_set=[np.array(x_test, dtype='float64')],
        #     pretraining_ratio=0.7,
        # )
        virtual_batch = {32: 32, 64: 64, 128: 128, 256: 128, 512: 128, 1024: 128}
        model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], eval_name=['train', 'test'],
                  eval_metric=['auc'], max_epochs=500, patience=args['patience'], batch_size=args['batch'],
                  virtual_batch_size=virtual_batch[args['batch']], num_workers=0, weights=1, drop_last=False)  # from_unsupervised=unsupervised_model
        # plt.plot(model.history['loss'])
        # plt.plot(model.history['train_auc'])
        # plt.plot(model.history['test_auc'])
        # plt.plot(model.history['lr'])
        # plt.show()
        y_pred = model.predict(np.array(x_test, dtype='float32'))
        print("#############", "TebNet", "#############")
        task_results.append(__metrics(y_test, y_pred))
    return task_results, models, features_list


def nabular_deep_models_multi_task(categorical_features, ys, y_names, train, args):
    models_names = ["TebNet"]
    np.random.seed(0)
    if args['feature_select'] == 0:
        # ------------- Simple preprocessing raw features -------------
        categorical_columns = []
        categorical_dims = {}
        for col in train.columns:
            if col in categorical_features:
                print(col, train[col].nunique())
                l_enc = preprocessing.LabelEncoder()
                train[col] = l_enc.fit_transform(train[col].values)
                categorical_columns.append(col)
                categorical_dims[col] = len(l_enc.classes_)

        features = [col for col in train.columns]
        cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
        cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
    else:
        print("ERROR for feature selection!")
        return

    task_results = []
    models = [TabNetMultiTaskClassifier(
        n_d=args['na_nd'],
        n_a=args['na_nd'],
        n_steps=args['n_steps'],
        gamma=1.3,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=args['cat_emb'],
        n_independent=args['n_indent'],
        n_shared=args['n_shame'],
        momentum=args['momentum'],
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 50,  # how to use learning rate scheduler
                          "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',  # "entmax"
        device_name='cuda'
    )]
    Y = np.vstack((ys[0], ys[1], ys[2], ys[3])).T
    x_train, x_test, y_train, y_test = train_test_split(train.values, Y, train_size=0.7, random_state=0)
    for j in range(len(models)):
        model = models[j]
        # unsupervised_model = TabNetPretrainer(
        #     optimizer_fn=torch.optim.Adam,
        #     optimizer_params=dict(lr=2e-2),
        #     mask_type='entmax'  # "sparsemax"
        # )
        # unsupervised_model.fit(
        #     X_train=np.array(x_train, dtype='float64'),
        #     eval_set=[np.array(x_test, dtype='float64')],
        #     pretraining_ratio=0.7,
        # )
        virtual_batch = {32: 32, 64: 64, 128: 128, 256: 128, 512: 128, 1024: 128}
        model.fit(X_train=x_train, y_train=y_train, eval_set=[(x_train, y_train), (x_test, y_test)],
                  eval_name=['train', 'test'], eval_metric=['auc'], max_epochs=500, patience=args['patience'],
                  batch_size=args['batch'], virtual_batch_size=virtual_batch[args['batch']], num_workers=0,
                  weights=0, drop_last=False)
        # plt.plot(model.history['loss'])
        # plt.plot(model.history['train_auc'])
        # plt.plot(model.history['test_auc'])
        # plt.plot(model.history['lr'])
        # plt.show()
        y_pred = model.predict(np.array(x_test, dtype='float32'))
        print(np.array(y_pred, dtype='float32').shape)
        print("#############", models_names[j], "#############")
        for i in range(len(ys)):
            print("-------------", y_names[i], "-------------")
            task_results.append(__metrics(np.array(y_test)[:, i], np.array(y_pred, dtype='float32').T[:, i]))
    return task_results
