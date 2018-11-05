import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import seaborn as sns

sns.set(color_codes=True)

df = pd.read_csv("pulsar_stars.csv")

label = 'target_class'

# Convert strings into frequency numbers
labelencoder=LabelEncoder()
for col in df.columns:
    df[col] = labelencoder.fit_transform(df[col])

# Split into train and test
train, test = train_test_split(df, test_size = 0.30, random_state=1)


# Train set
train_y = train[label]
train_x = train[[x for x in train.columns if label not in x]]
# Test/Validation set
test_y = test[label]
test_x = test[[x for x in test.columns if label not in x]]

training_accuracy = []
validation_accuracy = []
test_accuracy = []
layer_values = range(13)  # Up to 8 hidden layers

# For the neural network, experiment on different number of hidden layers
for layer in layer_values:

    # Define the classifier
    hiddens = tuple(layer * [32])
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hiddens, random_state=1)
    clf.fit(train_x, train_y)

    print 'layer:', layer

    training_accuracy.append(accuracy_score(train_y, clf.predict(train_x)))
    test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))


fig = plt.figure()
plt.plot(layer_values, training_accuracy, 'r', label="Training Set")
plt.plot(layer_values, test_accuracy, 'b', label="Testing Set")
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Hidden Layers for Pulsar Stars')
plt.legend(loc='best')
fig.savefig('figures/Pulsar_stars_neural_hidden.png')
plt.close(fig)

# For the neural network, experiment on different number of neurons
training_accuracy = []
validation_accuracy = []
test_accuracy = []
neurons = range(1,65)

for neuron in neurons:
    # Define the classifier
    hiddens = tuple(2 * [neuron])
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hiddens, random_state=1)
    clf.fit(train_x, train_y)

    print 'neuron:', neuron

    training_accuracy.append(accuracy_score(train_y, clf.predict(train_x)))
    test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))


fig = plt.figure()
plt.plot(neurons, training_accuracy, 'r', label="Training Set")
plt.plot(neurons, test_accuracy, 'b', label="Testing Set")
plt.xlabel('Number of Neurons')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Neurons for Pulsar Stars')
plt.legend(loc='best')
fig.savefig('figures/Pulsar stars_neural_neuron.png')
plt.close(fig)

# After finding the right hidden layer value, experiment on training set size
training_accuracy = []
test_accuracy = []
training_size = []
for k in range(1, 100):
    training_size.append(k)

# For knn
print "--- KNN ---"
for s in training_size:
    # Define the classifier
    hiddens = tuple(2 * [32])
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hiddens, random_state=1)

    temp_train, _ = train_test_split(train, test_size= 1 - s / 100.0, random_state=1)

    # Train set
    percent_train_y = temp_train[label]
    percent_train_x = temp_train[[x for x in train.columns if label not in x]]

    print percent_train_x.shape

    clf.fit(percent_train_x, percent_train_y)

    print 'Size: ', s, '%'
    print accuracy_score(test_y, clf.predict(test_x))

    training_accuracy.append(accuracy_score(percent_train_y, clf.predict(percent_train_x)))
    test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hiddens, random_state=1)
clf.fit(train_x, train_y)

training_accuracy.append(accuracy_score(train_y, clf.predict(train_x)))
test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))
training_size.append(100)

fig = plt.figure()
plt.plot(training_size, training_accuracy, 'r', label="Training Set")
plt.plot(training_size, test_accuracy, 'b', label="Testing Set")
plt.xlabel('Size of Training Set (%)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Training size for Pulsar Stars')
plt.legend(loc='best')
fig.savefig('figures/Pulsar_stars_neural_trainingSize.png')
plt.close(fig)
#modified code from Can Kabuloglu