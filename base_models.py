import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn import tree, svm, preprocessing
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from metrics import __metrics


def base_singletask_models(features, ys, y_names, model_use_names, run_time=1, preprocess=False):
    models_names = {"LogisticRegression": 0, "DecisionTree": 1, "SVM": 2, "LinearSVC": 3, "RandomForest": 4, "AdaBoost": 5, "MLP": 6, "XGBoost": 7}
    models = [
        LogisticRegression(solver='liblinear', max_iter=10000, class_weight='balanced'),
        tree.DecisionTreeClassifier(criterion="entropy", class_weight='balanced', max_depth=3),
        svm.SVC(class_weight='balanced', C=50),
        svm.LinearSVC(random_state=0, tol=1e-04, class_weight='balanced', C=50),
        RandomForestClassifier(n_estimators=1000, criterion="entropy", n_jobs=8, class_weight='balanced', max_depth=6),
        AdaBoostClassifier(n_estimators=501, learning_rate=0.1),
        MLPClassifier(max_iter=10000, early_stopping=True, hidden_layer_sizes=50),
        XGBClassifier(n_jobs=8, max_depth=6, min_child_weight=11)
    ]
    model_use = [models[models_names[x]] for x in model_use_names]
    results = []
    for i in range(len(ys)):
        if preprocess:
            features = SelectKBest(f_classif, k=200).fit_transform(features, ys[i])
            scaler = preprocessing.MinMaxScaler()
            features = scaler.fit_transform(features)
        task_results = []
        for j in range(run_time):
            x_train, x_test, y_train, y_test = train_test_split(features, ys[i], train_size=0.7, random_state=0, shuffle=True)
            print("-------------", y_names[i], ": {}run time".format(j), "-------------")
            model_results = {}
            for m_i, m in enumerate(model_use):
                m.fit(x_train, y_train)
                y_pred = m.predict(np.array(x_test, dtype='float32'))
                print("#############", model_use_names[m_i], "#############")
                model_results[model_use_names[m_i]] = __metrics(y_test, y_pred)
            task_results.append(model_results)
        results.append(task_results)
    return results





def base_multitask_models(features, ys, y_names):
    multi_label_names = ["DecisionTree", "RandomForest", "MLP"]
    multi_label_models = [
        tree.DecisionTreeClassifier(criterion="entropy", class_weight='balanced'),
        RandomForestClassifier(),
        MLPClassifier(early_stopping=True)]
    Y = np.vstack((ys[0], ys[1], ys[2], ys[3])).T
    print(Y.shape)
    x_train, x_test, y_train, y_test = train_test_split(features, Y, train_size=0.7, random_state=0)
    for j in range(len(multi_label_models)):
        model = MultiOutputClassifier(multi_label_models[j])
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print("#############", multi_label_names[j], "#############")
        for i in range(len(ys)):
            print("-------------", y_names[i], "-------------")
            __metrics(y_test[:, i], y_pred[:, i])
