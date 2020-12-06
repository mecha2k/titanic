import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import os

path = os.getcwd()

plt.rcParams["font.size"] = 14
plt.rcParams["axes.grid"] = True
plt.rcParams["lines.linewidth"] = 1
plt.rcParams["lines.markersize"] = 6

train = pd.read_csv("Kaggle/Titanic/train.csv")
test = pd.read_csv("Kaggle/Titanic/test.csv")

print("train:", train.shape, "test:", test.shape)
print(train.isnull().sum())

train.dropna(subset=["Survived"], axis=0, inplace=True)
y = train["Survived"]
train.drop(["Survived"], axis=1, inplace=True)
print(train.info())

X_train_f, X_valid_f, y_train, y_valid = train_test_split(train, y, test_size=0.2, random_state=0)

num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
cat_cols = ["Sex", "Cabin", "Embarked"]

tot_cols = num_cols + cat_cols
X_train = X_train_f[tot_cols].copy()
X_valid = X_valid_f[tot_cols].copy()
X_test = test[tot_cols].copy()

print(X_train.info())


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Preprocessing for numerical data
num_trans = SimpleImputer(strategy="median")

# Preprocessing for categorical data
cat_trans = Pipeline(
    steps=[("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# Bundle preprocessing for numerical and categorical data
preprocess = ColumnTransformer(transformers=[("num", num_trans, num_cols), ("cat", cat_trans, cat_cols)])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

# Preprocessing of training data, fit model
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)
print("MAE:", mean_absolute_error(y_valid, preds))

preds = clf.predict(X_test).round()

# Save test predictions to file
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds})
submission.to_csv("submission.csv", index=False)


"""
train.hist(bins=10, figsize=(12, 8))
plt.figure()
plt.show()

train.head().to_csv('titanic_data.csv', header=True)
with open('titanic_data.csv', 'a') as file:
	test.head().to_csv(file, header=True)
	train.isnull().sum().to_csv(file, header=True)
	test.isnull().sum().to_csv(file, header=True)
	train.describe().to_csv(file, header=True)
	test.describe().to_csv(file, header=True)
	train.describe(include=['O']).to_csv(file, header=True)
	


train['Survived'] = train['Survived'].astype(object)
train['Pclass'] = train['Pclass'].astype(object)

missing = train.isnull().sum().reset_index()
missing.columns = ['column', 'count']
missing['ratio'] = missing['count'] / train.shape[0]
print(missing.loc[missing['ratio'] != 0])

print(train['Survived'].value_counts())

feature = [x for x in train.columns if train[x].dtypes == "object"]
print(type(feature))

table = pd.crosstab([train['Sex'], train['Survived']], train.Pclass, margins=True)
print(table)

import seaborn as sb

sb.set(font_scale=2)
#sb.heatmap(table, cmap="YlGnBu", fmt='d', annot=True, cbar=False)
#plt.show()

ax = plt.subplots(4, 4, figsize=(12, 8))
#for n in range(len(feature)):

n = 0
sb.countplot(x=feature[n], data=train[feature[n]], ax=ax[n])

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()


#sex = train.groupby(['Sex', 'Survived'])['Survived'].count().unstack('Survived')
#print(sex)

#no_feature = list(set(train.columns) - set(feature) - set(['PassengerId','Survived']))
#no_feature = np.sort(no_feature)
"""


"""
import seaborn
seaborn.set()
seaborn.set_style('whitegrid')
seaborn.set_color_codes()

#for x in no_feature:
#    seaborn.distplot(train.loc[train[x].notnull(), x])
#    plt.title(x)
#    plt.show()

seaborn.pairplot(train[list(no_feature) + ['Survived']], 
	hue='Survived', x_vars=no_feature, y_vars=no_feature)
plt.show()
"""


"""
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

hist = train.hist(bins=50, figsize=(20, 15))
plt.show()


from pandas.plotting import scatter_matrix

attrib = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
scatter = scatter_matrix(train[attrib], figsize=(12, 8))
plt.savefig('./Titanic/scatter.png', format='png', dpi=300)
plt.show()
"""


"""
x_train = train.drop('Survived', axis=1)
y_train = train['Survived']

part_x_train = x_train[:700]
part_y_train = y_train[:700]

x_val = x_train[700:]
y_val = y_train[700:]

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(11, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(part_x_train, part_y_train, epochs=2, batch_size=256, 
					validation_data=(x_val, y_val))

history_dict = history.history
print(history_dict.keys())
"""
