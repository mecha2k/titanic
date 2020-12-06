import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn import model_selection


def piecount3(data, column):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.countplot(x=data[column], ax=ax[0])
    ax[0].set_title(column)
    data[column].value_counts().plot.pie(explode=[0, 0.0, 0], autopct="%0.4f%%", ax=ax[1], shadow=True)
    ax[1].set_title(column)
    ax[1].set_ylabel("")
    plt.show()


def main():
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")

    train.info()
    test.info()
    print(train.describe())
    num_train = len(train)

    train = pd.concat((train, test))

    # missingno.matrix(train, figsize=(15, 8))
    # plt.show()

    print(train.isnull().sum().sort_values(ascending=False))

    sns.set(font_scale=1.2)
    # plt.figure(figsize=(15, 8))
    # plt.title("Overall Correlation of Titanic Features", fontsize=18)
    # sns.heatmap(train.corr(), annot=True, cmap="RdYlGn", linewidths=0.2, annot_kws={"size": 20})
    # plt.show()
    #
    # sns.countplot(x=train["Survived"])
    # print(train["Survived"].value_counts())
    # plt.show()
    #
    # piecount3(train, "Pclass")
    #
    # print(pd.crosstab(train["Pclass"], train["Survived"], margins=True))
    #
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    train[["Pclass", "Survived"]].groupby(["Pclass"]).mean().plot.bar(ax=ax[0])
    ax[0].set_title("Survived per Pcalss")
    sns.countplot(x=train["Pclass"], hue=train["Survived"], ax=ax[1])
    ax[1].set_title("Pcalss Survived vs Not Survived")
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    train[["Sex", "Survived"]].groupby(["Sex"]).mean().plot.bar(ax=ax[0])
    ax[0].set_title("Survived per Sex")
    sns.countplot(x=train["Sex"], hue=train["Survived"], ax=ax[1])
    ax[1].set_title("Sex Survived vs Not Survived")
    plt.show()
    #
    # print("Oldest Passenger was ", train["Age"].max(), "Years")
    # print("Youngest Passenger was ", train["Age"].min(), "Years")
    # print("Average Age on the ship was ", int(train["Age"].mean()), "Years")
    #
    # sns.swarmplot(x=train["Survived"], y=train["Age"], size=2)
    # plt.xlabel("Survived")
    # plt.ylabel("Age")
    # plt.show()

    train["Age"].fillna(train["Age"].median(), inplace=True)
    train["Fare"].fillna(train["Fare"].median(), inplace=True)
    train["Embarked"].fillna(train["Embarked"].value_counts(ascending=False).index[0], inplace=True)
    train.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

    train.loc[(train["Sex"] == "male"), "Sex"] = 1
    train.loc[(train["Sex"] == "female"), "Sex"] = 2
    train.loc[(train["Age"] < 1), "Sex"] = 3

    train["Embarked"] = train["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    train = train.astype({"Embarked": np.int, "Sex": np.int})

    # print(train.head())
    # print(train["Sex"].value_counts() / len(train))

    print(train.isnull().sum().sort_values(ascending=False))
    print(train.info())
    print(train.head())

    # train["agerange"] = pd.qcut(train["Age"], 6)
    # train["initial"] = train["Name"].str.extract("([A-Za-z]+)\.")
    # train["lastname"] = train["Name"].str.extract("([A-Za-z]+)")
    #
    # freq = train["initial"].value_counts(normalize=True)
    # small_freq = freq[freq < 0.01].index
    #
    # train["initial"] = train["initial"].replace(small_freq, "other")
    # print(train["initial"].value_counts(normalize=True))
    # print(train.isnull().mean())
    #
    # sns.heatmap(train, cmap="YlGnBu", fmt="d", annot=True, cbar=False)
    # plt.show()
    # sns.countplot(x="Pclass", hue="Survived", data=train)
    # plt.show()
    # sns.boxplot(x="Pclass", y="Fare", data=train)
    # plt.show()

    x_train = train[:num_train]
    y_train = x_train["Survived"]
    x_test = train[num_train:]
    x_train = x_train.drop("Survived", axis=1)
    x_test = x_test.drop("Survived", axis=1)

    print(x_train.info())
    print(x_test.info())

    log_reg = LogisticRegression(solver="lbfgs", max_iter=500, random_state=0)
    log_reg.fit(x_train, y_train)
    preds = log_reg.predict(x_test)
    accuracy = log_reg.score(x_train, y_train)
    print("Logistic Regression :", accuracy)

    model_df = pd.DataFrame(
        {
            "name": [
                "Gradient Boosting",
                "XGBoost",
                "Random Forest",
                "Logistic Regression",
            ],
            "model": [
                GradientBoostingClassifier(),
                XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05),
                RandomForestClassifier(n_estimators=100, random_state=0),
                LogisticRegression(solver="lbfgs", max_iter=500, random_state=0),
            ],
        }
    )

    scores = []
    for i in range(len(model_df)):
        model = model_df["model"][i]
        model.fit(x_train, y_train)
        acc = cross_val_score(model, x_train, y_train, scoring="accuracy", cv=10)
        scores.append(acc.mean())
    model_df["score"] = scores

    for i in range(len(model_df)):
        print(model_df["name"][i], "(acc) : ", model_df["score"][i])

    results = model_df.sort_values(by="score", ascending=False).reset_index(drop=True)
    results.head(11)

    sns.barplot(x="score", y="name", data=model_df, color="c")
    plt.title("Machine Learning Algorithm Accuracy Score \n")
    plt.xlabel("Accuracy Score (%)")
    plt.ylabel("Algorithm")
    plt.xlim(0.8, 0.9)
    plt.show()

    # ran = RandomForestClassifier(n_estimators=100)
    # knn = KNeighborsClassifier()
    # log = LogisticRegression()
    # xgb = XGBClassifier()
    # gbc = GradientBoostingClassifier()
    # svc = SVC(probability=True)
    # ext = ExtraTreesClassifier()
    # ada = AdaBoostClassifier()
    # gnb = GaussianNB()
    # gpc = GaussianProcessClassifier()
    # bag = BaggingClassifier()
    #
    # models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]
    #
    # for mod in models:
    #     mod.fit(x_train, y_train)
    #     acc = cross_val_score(mod, x_train, y_train, scoring="accuracy", cv=10)
    #     scores.append(acc.mean())
    #
    # results = pd.DataFrame(
    #     {
    #         "Model": [
    #             "Random Forest",
    #             "K Nearest Neighbour",
    #             "Logistic Regression",
    #             "XGBoost",
    #             "Gradient Boosting",
    #             "SVC",
    #             "Extra Trees",
    #             "AdaBoost",
    #             "Gaussian Naive Bayes",
    #             "Gaussian Process",
    #             "Bagging Classifier",
    #         ],
    #         "Score": scores,
    #     }
    # )
    #
    # result_df = results.sort_values(by="Score", ascending=False).reset_index(drop=True)
    # result_df.head(11)
    #
    # sns.barplot(x="Score", y="Model", data=result_df, color="c")
    # plt.title("Machine Learning Algorithm Accuracy Score \n")
    # plt.xlabel("Accuracy Score (%)")
    # plt.ylabel("Algorithm")
    # plt.xlim(0.70, 0.84)
    # plt.show()


if __name__ == "__main__":
    main()
