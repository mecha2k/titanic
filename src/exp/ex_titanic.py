import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = os.getcwd()

plt.rcParams["font.size"] = 14
plt.rcParams["axes.grid"] = True
plt.rcParams["lines.linewidth"] = 1
plt.rcParams["lines.markersize"] = 6

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")


print(train.info())
print("train:", train.shape, "test:", test.shape)

train.head().to_csv("../data/titanic_data.csv", header=True)
with open("../data/titanic_data.csv", "a") as file:
    test.head().to_csv(file, header=True)
    train.isnull().sum().to_csv(file, header=True)
    test.isnull().sum().to_csv(file, header=True)
    train.describe().to_csv(file, header=True)
    test.describe().to_csv(file, header=True)
    train.describe(include=["O"]).to_csv(file, header=True)

train["Survived"] = train["Survived"].astype(object)
train["Pclass"] = train["Pclass"].astype(object)

missing = train.isnull().sum().reset_index()
missing.columns = ["column", "count"]
missing["ratio"] = missing["count"] / train.shape[0]
print(missing.loc[missing["ratio"] != 0])

print(train["Survived"].value_counts())

feature = [x for x in train.columns if train[x].dtypes == "object"]
print(type(feature))

table = pd.crosstab([train["Sex"], train["Survived"]], train.Pclass, margins=True)
print(table)

import seaborn as sb

sb.set(font_scale=1)

titanic = sb.load_dataset("titanic")

sb.heatmap(table, cmap="YlGnBu", fmt="d", annot=True, cbar=False)
plt.show()

ax = plt.subplots(4, 4, figsize=(12, 8))
for n in range(len(feature)):
    sb.countplot(x=feature[n], data=train, ax=ax[n])

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()


sex = train.groupby(["Sex", "Survived"])["Survived"].count().unstack("Survived")
print(sex)

no_feature = list(set(train.columns) - set(feature) - {"PassengerId", "Survived"})
no_feature = np.sort(no_feature)


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
