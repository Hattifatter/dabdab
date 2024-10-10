import pandas as pd
import numpy as np
import joblib

#from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

data_x = pd.read_csv(r'D:\Article\data_x.csv')
data_y = pd.read_csv(r'D:\Article\data_y.csv')
data_x.columns = ['s', 'e', 'c', 'L']
data_y.columns = ['Label']

for q in range(0,13):
    L= 2**q
    x_train = data_x[data_x['L'] == L]
    y_train = data_y.loc[x_train.index]
    # print(filtered_data_x)
    #print(filtered_data_y)
    # print(data_x)
    # print(data_y)
    # x_train, x_test, y_train, y_test = train_test_split(
    #     filtered_data_x[['s', 'e', 'c']],
    #     filtered_data_y,
    #     test_size=0.2,
    #     random_state=42
    # )
    #y_test = y_test.values.ravel()
    y_train = y_train.values.ravel()

    parameters = {'clf__kernel': ['rbf'], 'clf__C': [1, 10], 'clf__epsilon': [0.1, 0.2], 'clf__max_iter':[500,1000]}
    svr = SVR()
    #pipe = make_pipeline(("scaler",StandardScaler()), ("clf",svr))
    pipe = Pipeline(steps = [("scaler",StandardScaler()), ("clf",svr)])
    clf = GridSearchCV(pipe, parameters, verbose=1, n_jobs=-1)
    clf.fit(x_train, y_train)
    print("best_clf:", clf.best_params_)
    print(f'best_clf: {clf.best_score_}')
    # classifier = SVC(kernel='rbf')
    # classifier.fit(x_train, y_train)

    #y_pred = clf.predict(x_test)
    #print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    #print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    #print("R^2 Score:", r2_score(y_test, y_pred))
    joblib.dump(clf, f'./svr_model_L{L}.pkl')
    #print("тип выходные данные", accuracy_score(y_test, y_pred))
    #print(classification_report(y_test, y_pred))
