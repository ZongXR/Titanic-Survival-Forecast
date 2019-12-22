# %% 导入包
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import minmax_scale


# %% 定义一个通用的处理函数，新增属性，丢弃两个Embarked缺失的样本
def expand_column(df):
    result = df.copy()
    # 将性别进行0-1映射
    result["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    # 补充1列是否有Age
    result["no_age"] = df["Age"].isnull()
    # 补充1列是否有Fare
    result["no_fare"] = df["Fare"].isnull()
    # 补充1列是否有Cabin
    result["no_cabin"] = df["Cabin"].isnull()
    # 对Ticket属性进行OrdinalEncoder
    result["Ticket"] = OrdinalEncoder().fit_transform(df[["Ticket"]])
    # 新增一列，Ticket重复数
    result = result.merge(result["Ticket"].value_counts(), left_on="Ticket", how="left", right_index=True)
    # 新增1列，Fare平均值
    result["Fare_mean"] = result["Fare"] / result["Ticket_y"]
    # 对Fare_mean列缺失值补充-1
    result["Fare_mean"].fillna(-1, inplace=True)
    # 丢弃两个Embarked缺失的样本
    result = result[result["Embarked"].notnull()]
    return result


# %% 定义一个通用的处理函数，对列进行处理
def column_operator(df):
    # 对Embarked列进行OneHot
    result = pd.get_dummies(df, columns=["Embarked"])
    # 对Cabin列进行OrdinalEncoder
    result["Cabin"] = OrdinalEncoder().fit_transform(df[["Cabin"]])
    return result


# %% 定义一个通用的处理函数，填充缺失值
def fill_na(df):
    result = df.copy()
    # Age列缺失值填充-1
    result["Age"] = df["Age"].fillna(-1)
    # Fare列缺失值填充-1
    result["Fare"] = df["Fare"].fillna(-1)
    # Cabin, Embarked列缺失值填充""
    result[["Cabin", "Embarked"]] = df[["Cabin", "Embarked"]].fillna("")
    return result


# %% 进行特征选择
def feature_select(df):
    result = df.drop(["PassengerId", "Name", "Ticket_x", "Cabin"], axis=1)
    return result


if __name__ == "__main__":
    # %% 读入数据
    data_train = pd.read_csv("data/train.csv")
    data_test = pd.read_csv("data/test.csv")
    data = data_train.merge(data_test, how="outer")
    print(data_train.info())
    print(data_test.info())

    # %% 数据预处理
    pipeline = Pipeline([
        ("expand_column", FunctionTransformer(expand_column, validate=False)),
        ("fill_na", FunctionTransformer(fill_na, validate=False)),
        ("column_operator", FunctionTransformer(column_operator, validate=False)),
        ("feature_select", FunctionTransformer(feature_select, validate=False))
    ])
    data = pipeline.fit_transform(data)
    data_train = pipeline.transform(data_train)
    X_test = minmax_scale(pipeline.transform(data_test))

    # %% 分割标签数据
    X_train = minmax_scale(data_train.drop("Survived", axis=1))
    y_train = data_train["Survived"].copy()

    # %% 训练SVM模型
    params_svm = {
        "gamma": [0.00000001, 0.00000003, 0.0000001, 0.0000003],
        "C": [100000, 300000, 1000000, 3000000]
    }
    grid_svm = GridSearchCV(
        estimator=SVC(verbose=1, kernel="rbf"), param_grid=params_svm, cv=5,
        n_jobs=-1, iid=False, scoring="accuracy", verbose=1
    )
    try:
        model_svm = load("model/SVC.pkl")
    except FileNotFoundError:
        grid_svm.fit(X_train, y_train)
        model_svm = grid_svm.best_estimator_
        dump(model_svm, "model/SVC.pkl")
        print(grid_svm.best_params_)

    # %% 训练随机森林模型
    params_forest = {
        "n_estimators": range(3, 40),
        "max_depth": range(3, 6)
    }
    grid_forest = GridSearchCV(
        estimator=RandomForestClassifier(n_jobs=-1, verbose=1, random_state=1),
        cv=5, param_grid=params_forest,
        n_jobs=-1, iid=False, scoring="accuracy", verbose=1
    )
    try:
        model_forest = load("model/RandomForestClassifier.pkl")
    except FileNotFoundError:
        grid_forest.fit(X_train, y_train)
        model_forest = grid_forest.best_estimator_
        dump(model_forest, "model/RandomForestClassifier.pkl")
        print(grid_forest.best_params_)

    # %% 预测
    print(accuracy_score(model_svm.predict(X_train), y_train))
    print(accuracy_score(model_forest.predict(X_train), y_train))
    svm_predict = data_test.copy()
    forest_predict = data_test.copy()
    svm_predict["Survived"] = model_svm.predict(X_test)
    svm_predict.to_csv("data/svm_predict.csv")
    forest_predict["Survived"] = model_forest.predict(X_test)
    forest_predict.to_csv("data/forest_predict.csv")



