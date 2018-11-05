import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import tree
import seaborn as sns

sns.set(color_codes=True)

df = pd.read_csv("pulsar_stars.csv")

# Convert strings into frequency numbers
labelencoder=LabelEncoder()
for col in df.columns:
    df[col] = labelencoder.fit_transform(df[col])

# Split into train and test
train, test = train_test_split(df, test_size = 0.30)

label = 'target_class'
# Train set
train_y = train[label]
train_x = train[[x for x in train.columns if label not in x]]
# Test/Validation set
test_y = test[label]
test_x = test[[x for x in test.columns if label not in x]]

training_accuracy = []
validation_accuracy = []
test_accuracy = []
training_size = []
for k in range(1, 100):
    training_size.append(k)

# For knn
for s in training_size:
    # Define the classifier
    clf = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 1, max_depth=3, min_samples_leaf=3)
    temp_train, _ = train_test_split(train, test_size= 1 - s / 100.0)

    # Train set
    percent_train_y = temp_train[label]
    percent_train_x = temp_train[[x for x in train.columns if label not in x]]

    clf.fit(percent_train_x, percent_train_y)

    print 'Size: ', s, '%'

    training_accuracy.append(accuracy_score(percent_train_y, clf.predict(percent_train_x)))
    cv = cross_val_score(clf, percent_train_x, percent_train_y, cv=7).mean()
    validation_accuracy.append(cv)
    test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))

clf = tree.DecisionTreeClassifier(max_features=8, random_state=1)
clf.fit(train_x, train_y)

training_accuracy.append(accuracy_score(train_y, clf.predict(train_x)))
cv = cross_val_score(clf, train_x, train_y, cv=7).mean()
validation_accuracy.append(cv)
test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))
training_size.append(100)

fig = plt.figure()
line3, = plt.plot(training_size, validation_accuracy, 'y', label="Cross Validation")
line2, = plt.plot(training_size, training_accuracy, 'r', label="Training Set")
line1, = plt.plot(training_size, test_accuracy, 'b', label="Testing Set")

plt.xlabel('Size of Training Set (%)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Training size versus for Pulsar Stars')
plt.legend(loc='best')
fig.savefig('figures/Pulsar_stars_decision_trainingSize.png')
plt.close(fig)


# For Max depth
training_accuracy2 = []
validation_accuracy2 = []
test_accuracy2 = []
depthList = []
for k in range(1,100):
    depthList.append(k)
    
for s in depthList:
    # Define the classifier
    clf = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 1, max_depth= s, min_samples_leaf=3)
    temp_train, _ = train_test_split(train, test_size= 0.3)

    # Train set
    percent_train_y = temp_train[label]
    percent_train_x = temp_train[[x for x in train.columns if label not in x]]

    clf.fit(percent_train_x, percent_train_y)

    print 'Depth: ', s, ' '

    training_accuracy2.append(accuracy_score(percent_train_y, clf.predict(percent_train_x)))
    cv = cross_val_score(clf, percent_train_x, percent_train_y, cv=7).mean()
    validation_accuracy2.append(cv)
    test_accuracy2.append(accuracy_score(test_y, clf.predict(test_x)))

clf = tree.DecisionTreeClassifier(max_features=8, random_state=1)
clf.fit(train_x, train_y)

training_accuracy2.append(accuracy_score(train_y, clf.predict(train_x)))
cv = cross_val_score(clf, train_x, train_y, cv=7).mean()
validation_accuracy2.append(cv)
test_accuracy2.append(accuracy_score(test_y, clf.predict(test_x)))
depthList.append(100)

fig = plt.figure()
line3, = plt.plot(depthList, validation_accuracy, 'y', label="Cross Validation")
line2, = plt.plot(depthList, training_accuracy2, 'r', label="Training Set")
line1, = plt.plot(depthList, test_accuracy2, 'b', label="Testing Set")

plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Max Depth for Pulsar Stars')
plt.legend(loc='best')
fig.savefig('figures/Pulsar_stars_decision_maxdepth.png')
plt.close(fig)
#modified code from Can Kabuloglu