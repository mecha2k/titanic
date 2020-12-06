import pandas as pd

train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")

print(train_data.info())
print("train:", train_data.shape, "test:", test_data.shape)

total = train_data.isnull().sum().sort_values(ascending=False)
percent = (train_data.isnull().sum() / train_data.isnull().count() * 100).sort_values(ascending=False)
print(total)

print(train_data[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean())
print(train_data[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean())
print(train_data[["SibSp", "Survived"]].groupby(["SibSp"], as_index=False).mean())
print(train_data[["Parch", "Survived"]].groupby(["Parch"], as_index=False).mean())

train_data["Age"].fillna(train_data["Age"].median(), inplace=True)
train_data["Embarked"].fillna(train_data["Embarked"].mode()[0], inplace=True)
train_data.drop(["Cabin", "Ticket"], axis=1, inplace=True)

test_data["Age"].fillna(test_data["Age"].median(), inplace=True)
test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)
test_data.drop(["Cabin", "Ticket"], axis=1, inplace=True)
print(train_data.info())


all_data = [train_data, test_data]
print(all_data[0].shape, all_data[1].shape)

for data in all_data:
    data["Title"] = data["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
print(pd.crosstab(train_data["Title"], train_data["Sex"]))


for data in all_data:
    data["Title"] = data["Title"].replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare"
    )
    data["Title"] = data["Title"].replace("Mlle", "Miss")
    data["Title"] = data["Title"].replace("Ms", "Miss")
    data["Title"] = data["Title"].replace("Mme", "Mrs")

print(pd.crosstab(train_data["Title"], train_data["Survived"]))
print(train_data[["Title", "Survived"]].groupby(["Title"], as_index=False).mean())

title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for data in all_data:
    data["Title"] = data["Title"].map(title_map)
print(data["Title"].isnull().sum())

train_data.drop(["Name", "PassengerId"], axis=1, inplace=True)
test_data.drop(["Name"], axis=1, inplace=True)

print("train:", train_data.shape, "test:", test_data.shape)

all_data = [train_data, test_data]

for data in all_data:
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1}).astype(int)
    data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)


print(train_data.head())
print(test_data.head())
print(train_data.isnull().sum().sum())
print(test_data.isnull().sum().sum())


from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

X = train_data.drop("Survived", axis=1)
X_test = test_data.drop("PassengerId", axis=1)
y = train_data["Survived"]

print(X.shape, y.shape, X_test.shape)

log_reg = LogisticRegression(solver="lbfgs", max_iter=500, random_state=0)
log_reg.fit(X, y)
preds = log_reg.predict(X_test)
accuracy = log_reg.score(X, y)
print("Logistic Regression :", accuracy)

svm = SVC(gamma="auto", probability=True)
svm.fit(X, y)
preds = svm.predict(X_test)
accuracy = svm.score(X, y)
print("Support Vector Machine :", accuracy)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X, y)
preds = random_forest.predict(X_test)
accuracy = random_forest.score(X, y)
print("Random Forest Classifier :", accuracy)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X, y)
preds = decision_tree.predict(X_test)
accuracy = decision_tree.score(X, y)
print("Decision Tree Classifier :", accuracy)

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

voting = VotingClassifier(
    estimators=[("log_reg", log_reg), ("random_forest", random_forest), ("decision_tree", decision_tree), ("svm", svm)],
    voting="hard",
)

models = [log_reg, svm, random_forest, decision_tree, voting]

for mod in models:
    mod.fit(X, y)
    preds = mod.predict(X_test)
    accuracy = mod.score(X, y)
    print(mod.__class__.__name__, accuracy)

import xgboost as xgb

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
gbm.fit(X, y)
preds = gbm.predict(X_test)
accuracy = gbm.score(X, y)
print("XGBoost :", accuracy)

submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": preds})
submission.to_csv("submission.csv", index=False)


"""
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)


print(train_data.isnull().sum())
print(test_data.isnull().sum())

all_data = [train_data, test_data]

for data in all_data:
	data['Family'] = data['SibSp'] + data['Parch'] + 1



import re

def get_title(name):
	ti_search = re.search(' ([A-Za-z]+)/.', name)
	
	if ti_search: 
		return ti_search.group(1)

	return ""

for data in all_data:
	data['Title'] = data['Name'].apply(get_title)

print(data['Title'])
print(data['Family'])

for data in all_data:
	data['Title'] = data['Title'].replace([
		'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 
		'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
	data['Title'] = data['Title'].replace('Mlle', 'Miss')
	data['Title'] = data['Title'].replace('Ms', 'Miss')
	data['Title'] = data['Title'].replace('Mme', 'Mrs')

print(data['Title'])

print(data['Fare'].value_counts())

for data in all_data:
	data['Age_bin'] = pd.cut(data['Age'], bins=[0, 12, 20, 40, 120], 
		labels=['Children', 'Teenage', 'Adult', 'Elder'])
	data['Fare_bin'] = pd.cut(data['Fare'], bins=[7.91, 14,45, 31, 120], 
		labels=['Low', 'Median', 'Average', 'High'])
"""
