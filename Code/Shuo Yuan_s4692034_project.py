# 开发时间：2022/10/20 15:50
# 作者：Shawn_Bravo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


def load_data(filename):
    data = pd.read_csv(filename, index_col=None)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y


def all_value_imputation(dataframe):
    df_copy = dataframe.copy()
    mean_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    df_copy.iloc[:, :103] = mean_imp.fit_transform(df_copy.iloc[:, :103])
    mode_imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    df_copy.iloc[:, 103:106] = mode_imp.fit_transform(df_copy.iloc[:, 103:106])

    return df_copy


def class_specific_imputation(dataframe):
    df_copy = dataframe.copy()
    y = df_copy.iloc[:, -1]
    positive_row = []
    negative_row = []

    for index, label in enumerate(y):
        if label == 0:
            negative_row.append(index)
        else:
            positive_row.append(index)

    df_pos = df_copy.iloc[positive_row, :]
    df_pos = all_value_imputation(df_pos)
    df_neg = df_copy.iloc[negative_row, :]
    df_neg = all_value_imputation(df_neg)

    df_copy = pd.merge(df_pos, df_neg, how='outer')
    return df_copy.sample(frac=1, random_state=17)


def knn_imputation(dataframe):
    df_copy = dataframe.copy()

    # best hyperparameter after cross validation
    knn_imp = KNNImputer(missing_values=np.nan, n_neighbors=6)
    df_copy.iloc[:, :] = knn_imp.fit_transform(df_copy.iloc[:, :])
    df_copy.iloc[:, 103:] = df_copy.iloc[:, 103:].round()
    return df_copy


def cross_validation(X, y, model):
    kf = KFold(n_splits=10)
    accuracies = []
    f1_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)

        accuracies.append(accuracy)
        f1_scores.append(f1)
        mean_acc = round(np.mean(accuracies), 3)
        mean_f1 = round(np.mean(f1_scores), 3)
    return mean_acc, mean_f1


def model_based_outlier_detection(dataframe):
    df_copy = dataframe.copy()
    for column in range(103):
        q1, q3 = np.percentile(df_copy.iloc[:, column], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        higher_bound = q3 + (1.5 * iqr)

        for row in range(df_copy.shape[0]):
            if (df_copy.iloc[row, column] < lower_bound) | (df_copy.iloc[row, column] > higher_bound):
                df_copy.iloc[row, column] = np.nan
    return df_copy


def density_based_outlier_detection(dataframe):
    df_copy = dataframe.copy()
    lof = LocalOutlierFactor(contamination=0.03)
    for column in range(103):
        outlier_label = lof.fit_predict(np.array(df_copy.iloc[:, column]).reshape(-1, 1))
        for row in range(df_copy.shape[0]):
            if outlier_label[row] == -1:
                if column < 103:
                    df_copy.iloc[row, column] = np.nan
    return df_copy


def isolation_based_outlier_detection(dataframe):
    df_copy = dataframe.copy()
    iso_forest = IsolationForest(contamination=0.03, random_state=617)
    for column in range(103):
        outlier_label = iso_forest.fit_predict(np.array(df_copy.iloc[:, column]).reshape(-1, 1))
        for row in range(df_copy.shape[0]):
            if outlier_label[row] == -1:
                if column < 103:
                    df_copy.iloc[row, column] = np.nan
    return df_copy


def max_min_normalization(dataframe):
    df_copy = dataframe.copy()
    df_copy.iloc[:, :103] = MinMaxScaler().fit_transform(df_copy.iloc[:, :103])
    return df_copy


def standardization(dataframe):
    df_copy = dataframe.copy()
    df_copy.iloc[:, :103] = StandardScaler().fit_transform(df_copy.iloc[:, :103])
    return df_copy


def generate_combination_csv(train_data):
    df_knn = knn_imputation(train_data)
    df_knn_1 = df_knn.copy()
    df_knn_2 = df_knn.copy()
    df_knn.to_csv("dataset/knn_imp_Ecoli.csv", index=None)

    df_all = all_value_imputation(train_data)
    df_all_1 = df_all.copy()
    df_all_2 = df_all.copy()
    df_all.to_csv("dataset/all_value_imp_Ecoli.csv", index=None)

    df_class = class_specific_imputation(train_data)
    df_class_1 = df_class.copy()
    df_class_2 = df_class.copy()
    df_class.to_csv("dataset/class_specific_imp_Ecoli.csv", index=None)

    df_knn_model = model_based_outlier_detection(df_knn)
    df_knn_model = knn_imputation(df_knn_model)
    df_knn_model.to_csv("dataset/knn_imp_Ecoli_model.csv", index=None)

    df_all_model = model_based_outlier_detection(df_all)
    df_all_model = all_value_imputation(df_all_model)
    df_all_model.to_csv("dataset/all_value_imp_Ecoli_model.csv", index=None)

    df_class_model = model_based_outlier_detection(df_class)
    df_class_model = class_specific_imputation(df_class_model)
    df_class_model.to_csv("dataset/class_specific_imp_Ecoli_model.csv", index=None)

    df_knn_density = density_based_outlier_detection(df_knn_1)
    df_knn_density = knn_imputation(df_knn_density)
    df_knn_density.to_csv("dataset/knn_imp_Ecoli_density.csv", index=None)

    df_all_density = density_based_outlier_detection(df_all_1)
    df_all_density = all_value_imputation(df_all_density)
    df_all_density.to_csv("dataset/all_value_imp_Ecoli_density.csv", index=None)

    df_class_density = density_based_outlier_detection(df_class_1)
    df_class_density = class_specific_imputation(df_class_density)
    df_class_density.to_csv("dataset/class_specific_imp_Ecoli_density.csv", index=None)

    df_knn_iso = isolation_based_outlier_detection(df_knn_2)
    df_knn_iso = knn_imputation(df_knn_iso)
    df_knn_iso.to_csv("dataset/knn_imp_Ecoli_iso.csv", index=None)

    df_all_iso = isolation_based_outlier_detection(df_all_2)
    df_all_iso = all_value_imputation(df_all_iso)
    df_all_iso.to_csv("dataset/all_value_imp_Ecoli_iso.csv", index=None)

    df_class_iso = isolation_based_outlier_detection(df_class_2)
    df_class_iso = class_specific_imputation(df_class_iso)
    df_class_iso.to_csv("dataset/class_specific_imp_Ecoli_iso.csv", index=None)


def preprocessing_benchmark(X, y, file_name):
    dt = DecisionTreeClassifier(random_state=17)
    accuracy, f1 = cross_validation(X, y, dt)
    print(file_name + ' Decision tree accuracy:', accuracy)
    print(file_name + ' Decision tree f1_score:', f1)
    return accuracy, f1


def choose_best_preprocessing_combination(file_names):
    scores = []
    for index in range(len(file_names)):
        X, y = load_data("dataset/" + file_names[index])
        accuracy, score = preprocessing_benchmark(X, y, file_names[index])
        scores.append(score)
    print(scores)
    return file_names[np.argmax(scores)]


def find_best_model(filename):
    X, y = load_data(filename)
    # Decision tree
    param_grid = {'max_depth': [int(x) for x in np.linspace(start=1, stop=300, num=30)],
                  'criterion': ['gini', 'entropy'],
                  'splitter': ['best', 'random'],
                  'min_samples_leaf': [1, 2, 3, 4, 5],
                  'max_features': ['sqrt', 'log2']}

    dt = RandomForestClassifier(random_state=17)
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    print(grid_search.best_params_)

    best_DT = DecisionTreeClassifier(random_state=17, max_depth=5, splitter='best')

    # RandomForest Classifier
    param_grid2 = {'max_features': [int(x) for x in np.linspace(start=10, stop=100, num=30)],
                   'criterion': ['gini', 'log_loss', 'entropy'],
                   'n_estimators': [int(x) for x in np.linspace(start=20, stop=200, num=50)]}

    rf = RandomForestClassifier(random_state=17)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid2, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    print(grid_search.best_params_)

    # 83 40 0.818, 39 75 0.813 39 71 gini 0.811
    best_RF = RandomForestClassifier(random_state=17, n_estimators=85, max_features=40)

    # k nearest neighbor
    scores = []
    for i in np.arange(1, 11, 1):
        print(i)
        knn_model = KNeighborsClassifier(n_neighbors=i)
        accuracy, score = cross_validation(X, y, knn_model)
        scores.append(score)
    print(max(scores), ([*np.arange(1, 11, 1)][scores.index(max(scores))]))
    plt.figure(figsize=[20, 5])
    plt.plot(np.arange(1, 11, 1), scores)
    plt.show()

    # naive bayes
    nb1 = GaussianNB()
    nb2 = MultinomialNB()
    nb3 = BernoulliNB()

    acc1, sc1 = cross_validation(X, y, nb1)
    acc2, sc2 = cross_validation(X, y, nb2)
    acc3, sc3 = cross_validation(X, y, nb3)

    best_NB = BernoulliNB()

def main():
    """
    # find best combination of preprocessing techniques
    train_data = pd.read_csv('Ecoli.csv')
    generate_combination_csv(train_data)
    file_names = ["knn_imp_Ecoli.csv", "all_value_imp_Ecoli.csv",
                  "class_specific_imp_Ecoli.csv", "knn_imp_Ecoli_model.csv",
                  "all_value_imp_Ecoli_model.csv", "class_specific_imp_Ecoli_model.csv",
                  "knn_imp_Ecoli_density.csv", "all_value_imp_Ecoli_density.csv",
                  "class_specific_imp_Ecoli_density.csv", "knn_imp_Ecoli_iso.csv",
                  "all_value_imp_Ecoli_iso.csv", "class_specific_imp_Ecoli_iso.csv"]

    best_combination = choose_best_preprocessing_combination(file_names)
    df_best = pd.read_csv("dataset/" + best_combination)
    df_best_max_min = max_min_normalization(df_best)
    df_best_max_min.to_csv("dataset/max_min_" + best_combination)
    df_best_stand = standardization(df_best)
    df_best_stand.to_csv("dataset/standardization_" + best_combination)

    normalization_files = [best_combination, "max_min_" + best_combination, "standardization_" + best_combination]
    best_combination = choose_best_preprocessing_combination(normalization_files)
    print(best_combination)

    # test the performance of best model on training set
    find_best_model("dataset/class_specific_imp_Ecoli_iso.csv")

    X, y = load_data("dataset/max_min_class_specific_imp_Ecoli_iso.csv.csv")
    best_model = RandomForestClassifier(random_state=17, max_features=40, n_estimators=85)
    acc, sc = cross_validation(X, y, best_model)
    print(acc)
    print(sc)
    """

    train_data = pd.read_csv("Ecoli.csv")
    train_data = class_specific_imputation(train_data)
    train_data = model_based_outlier_detection(train_data)
    train_data = class_specific_imputation(train_data)
    train_data = max_min_normalization(train_data)

    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    best_model = RandomForestClassifier(random_state=17, max_features=40, n_estimators=85)
    best_model.fit(X_train, y_train)

    X_test = pd.read_csv('Ecoli_test.csv', index_col=None)
    X_test = max_min_normalization(X_test)
    y_test = best_model.predict(X_test)

    train_accuracy, train_f1 = cross_validation(X_train, y_train, best_model)

    a = {'column1': y_test, 'column2': None}
    b = pd.DataFrame(a)
    b.to_csv('s4692034.csv', header=0, index=0, sep=',', na_rep='')
    pd.DataFrame(np.array((train_accuracy, train_f1)).reshape(1, 2)).to_csv("s4692034.csv", mode='a',
                                                                            index=False, header=False)


if __name__ == '__main__':
    main()
