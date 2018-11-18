# source: https://github.com/joshuamorton/Machine-Learning

import numpy as np
from StringIO import StringIO
from pprint import pprint
import argparse
from matplotlib import pyplot as plt
from collections import Counter


from sklearn.decomposition.pca import PCA as PCA
from sklearn.decomposition import FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection as RandomProjection
from sklearn.cluster import KMeans as KM
from sklearn.mixture import GMM as EM
from sklearn.feature_selection import SelectKBest as best
from sklearn.feature_selection import chi2


from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

# first map things to things
def create_mapper(l):
    return {l[n] : n for n in xrange(len(l))}

apartment_map = create_mapper(["'New York'", "'San Francisco'"])

cap_shape = create_mapper(['b', 'c', 'f', 'x', 'k', 's'])
cap_surface = create_mapper(['f', 'g', 'y', 's'])
cap_color = create_mapper(['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'])
bruises = create_mapper(['t', 'f'])
odor = create_mapper(['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'])
gill_attach = create_mapper(['a', 'd', 'f', 'n'])
gill_space = create_mapper(['c', 'w', 'd'])
gill_size = create_mapper(['b', 'n'])
gill_color = create_mapper(['k','n','b','h','g','r','o','p','u','e','w','y'])
stalk_shape = create_mapper(['e','t'])
stalk_root = create_mapper(['b','c','u','e','z','r','?'])
stalk_surface_above_ring = create_mapper(['f','y','k','s'])
stalk_surface_below_ring = create_mapper(['f','y','k','s'])
stalk_color_above_ring = create_mapper(['n','b','c','g','o','p','e','w','y'])
stalk_color_below_ring = create_mapper(['n','b','c','g','o','p','e','w','y'])
veil_type = create_mapper(['p','u'])
veil_color = create_mapper(['n','o','w','y'])
ring_number = create_mapper(['n','o','t'])
ring_type = create_mapper(['c','e','f','l','n','p','s','z'])
spore_print_color = create_mapper(['k','n','b','h','r','o','u','w','y'])
population = create_mapper(['a','c','n','s','v','y'])
habitat = create_mapper(['g','l','m','p','u','w','d'])
classification = create_mapper(['p','e'])

SentimentDataSetConverters = {}
ApartmentDataSetConverters = {8: lambda x: apartment_map[x]}
MushroomsDatasetConverters = {
    0: lambda x: cap_shape[x],
    1: lambda x: cap_surface[x],
    2: lambda x: cap_color[x],
    3: lambda x: bruises[x],
    4: lambda x: odor[x],
    5: lambda x: gill_attach[x],
    6: lambda x: gill_space[x],
    7: lambda x: gill_size[x],
    8: lambda x: gill_color[x],
    9: lambda x: stalk_shape[x],
    10: lambda x: stalk_root[x],
    11: lambda x: stalk_surface_above_ring[x],
    12: lambda x: stalk_surface_below_ring[x],
    13: lambda x: stalk_color_above_ring[x],
    14: lambda x: stalk_color_below_ring[x],
    15: lambda x: veil_type[x],
    16: lambda x: veil_color[x],
    17: lambda x: ring_number[x],
    18: lambda x: ring_type[x],
    19: lambda x: spore_print_color[x],
    20: lambda x: population[x],
    21: lambda x: habitat[x],
    22: lambda c: classification[c]
    }

converters = {"sentiment": SentimentDataSetConverters,
              "apartments": ApartmentDataSetConverters,
              "mushrooms": MushroomsDatasetConverters}


def load(filename, converter):
    with open(filename) as data:
        instances = [line for line in data if "?" not in line]  # remove lines with unknown data

    return np.loadtxt(instances,
                      delimiter=',',
                      converters=converter,
                      dtype='u4')

def create_dataset(name, test, train):
    training_set = load(train, converters[name])
    testing_set = load(test, converters[name])
    train_x, train_y = np.hsplit(training_set, [training_set[0].size-1])
    test_x, test_y = np.hsplit(testing_set, [testing_set[0].size-1])
    # this splits the dataset on the last instance, so your label must
    # be the last instance in the dataset
    return train_x, train_y, test_x, test_y


def plot(axes, values, x_label, y_label, title, name):
    plt.clf()
    plt.plot(*values)
    plt.axis(axes)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(name+".png", dpi=500)
    # plt.show()
    plt.clf()


def pca(tx, ty, rx, ry):
    compressor = PCA(n_components = tx[1].size/2)
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    em(newtx, ty, newrx, ry, add="wPCAtr", times=10)
    km(newtx, ty, newrx, ry, add="wPCAtr", times=10)
    nn(newtx, ty, newrx, ry, add="wPCAtr")


def ica(tx, ty, rx, ry):
    compressor = ICA(whiten=False)  # for some people, whiten needs to be off
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    em(newtx, ty, newrx, ry, add="wICAtr", times=10)
    km(newtx, ty, newrx, ry, add="wICAtr", times=10)
    nn(newtx, ty, newrx, ry, add="wICAtr")


def randproj(tx, ty, rx, ry):
    compressor = RandomProjection(tx[1].size)
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    # compressor = RandomProjection(tx[1].size)
    newrx = compressor.transform(rx)
    em(newtx, ty, newrx, ry, add="wRPtr", times=10)
    km(newtx, ty, newrx, ry, add="wRPtr", times=10)
    nn(newtx, ty, newrx, ry, add="wRPtr")


def kbest(tx, ty, rx, ry):
    compressor = best(chi2)
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    em(newtx, ty, newrx, ry, add="wKBtr", times=10)
    km(newtx, ty, newrx, ry, add="wKBtr", times=10)
    nn(newtx, ty, newrx, ry, add="wKBtr")


def em(tx, ty, rx, ry, add="", times=5):
    errs = []

    # this is what we will compare to
    checker = EM(n_components=2)
    checker.fit(ry)
    truth = checker.predict(ry)

    # so we do this a bunch of times
    for i in range(2,times):
        clusters = {x:[] for x in range(i)}

        # create a clusterer
        clf = EM(n_components=i)
        clf.fit(tx)  #fit it to our data
        test = clf.predict(tx)
        result = clf.predict(rx)  # and test it on the testing set

        # here we make the arguably awful assumption that for a given cluster,
        # all values in tha cluster "should" in a perfect world, belong in one
        # class or the other, meaning that say, cluster "3" should really be
        # all 0s in our truth, or all 1s there
        #
        # So clusters is a dict of lists, where each list contains all items
        # in a single cluster
        for index, val in enumerate(result):
            clusters[val].append(index)

        # then we take each cluster, find the sum of that clusters counterparts
        # in our "truth" and round that to find out if that cluster should be
        # a 1 or a 0
        mapper = {x: round(sum(truth[v] for v in clusters[x])/float(len(clusters[x]))) if clusters[x] else 0 for x in range(i)}

        # the processed list holds the results of this, so if cluster 3 was
        # found to be of value 1,
        # for each value in clusters[3], processed[value] == 1 would hold
        processed = [mapper[val] for val in result]
        errs.append(sum((processed-truth)**2) / float(len(ry)))
    plot([0, times, min(errs)-.1, max(errs)+.1],[range(2, times), errs, "ro"], "Number of Clusters", "Error Rate", "Expectation Maximization Error", "EM"+add)

    # dank magic, wrap an array cuz reasons
    td = np.reshape(test, (test.size, 1))
    rd = np.reshape(result, (result.size, 1))
    newtx = np.append(tx, td, 1)
    newrx = np.append(rx, rd, 1)
    nn(newtx, ty, newrx, ry, add="onEM"+add)



def km(tx, ty, rx, ry, add="", times=5):
    #this does the exact same thing as the above
    errs = []

    checker = KM(n_clusters=2)
    checker.fit(ry)
    truth = checker.predict(ry)

    # so we do this a bunch of times
    for i in range(2,times):
        clusters = {x:[] for x in range(i)}
        clf = KM(n_clusters=i)
        clf.fit(tx)  #fit it to our data
        test = clf.predict(tx)
        result = clf.predict(rx)  # and test it on the testing set
        for index, val in enumerate(result):
            clusters[val].append(index)
        mapper = {x: round(sum(truth[v] for v in clusters[x])/float(len(clusters[x]))) if clusters[x] else 0 for x in range(i)}
        processed = [mapper[val] for val in result]
        errs.append(sum((processed-truth)**2) / float(len(ry)))
    plot([0, times, min(errs)-.1, max(errs)+.1],[range(2, times), errs, "ro"], "Number of Clusters", "Error Rate", "KMeans clustering error", "KM"+add)

    td = np.reshape(test, (test.size, 1))
    rd = np.reshape(result, (result.size, 1))
    newtx = np.append(tx, td, 1)
    newrx = np.append(rx, rd, 1)
    nn(newtx, ty, newrx, ry, add="onKM"+add)


def nn(tx, ty, rx, ry, add="", iterations=250):
    """
    trains and plots a neural network on the data we have
    """
    resultst = []
    resultsr = []
    positions = range(iterations)
    network = buildNetwork(tx[1].size, 5, 1, bias=True)
    ds = ClassificationDataSet(tx[1].size, 1)
    for i in xrange(len(tx)):
        ds.addSample(tx[i], [ty[i]])
    trainer = BackpropTrainer(network, ds, learningrate=0.01)
    train = zip(tx, ty)
    test = zip(rx, ry)
    for i in positions:
        trainer.train()
        resultst.append(sum(np.array([(round(network.activate(t_x)) - t_y)**2 for t_x, t_y in train])/float(len(train))))
        resultsr.append(sum(np.array([(round(network.activate(t_x)) - t_y)**2 for t_x, t_y in test])/float(len(test))))
        # resultsr.append(sum((np.array([round(network.activate(test)) for test in rx]) - ry)**2)/float(len(ry)))
        print i, resultst[-1], resultsr[-1]
    plot([0, iterations, 0, 1], (positions, resultst, "ro", positions, resultsr, "bo"), "Network Epoch", "Percent Error", "Neural Network Error", "NN"+add)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run clustering algorithms on stuff')
    parser.add_argument("name")
    args = parser.parse_args()
    name = args.name
    train = name+".data"
    test = name+".test"
    train_x, train_y, test_x, test_y = create_dataset(name, test, train)
    # nn(train_x, train_y, test_x, test_y); print 'nn done'
    # em(train_x, train_y, test_x, test_y, times = 10); print 'em done'
    # km(train_x, train_y, test_x, test_y, times = 10); print 'km done'
    # pca(train_x, train_y, test_x, test_y); print 'pca done'
    ica(train_x, train_y, test_x, test_y); print 'ica done'
    # randproj(train_x, train_y, test_x, test_y); print 'randproj done'
    # kbest(train_x, train_y, test_x, test_y); print 'kbest done'