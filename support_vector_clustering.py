#Implementation of Support Vector Clustering technique,
#based on the 2001 paper by Vapnik et. al

import numpy
from sklearn import svm
import random

def one_class_classifier(data, kernel="poly", degree=2):
    """
    Returns a SVDD (Support Vector Data Description) model
    based on input dataset
    """
    nu = 1/ float(len(data))
    classifier = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=0.0, degree=degree)
    classifier.fit(data)
    return classifier


def check_same_cluster(classifier, vector1, vector2, n=40):
    """
    Check if two points in input space belong to the same cluster

    This is done by choosing n random points on the line joining
    the two points in input space, and checking whether all of them
    get classified as non-outliers in feature space

    If yes, then the two points belong to the same cluster
    """
    points = []
    for i in range(n):
        a = random.random()
        points.append(a*vector1 + (1-a)*vector2)
    if -1 in classifier.predict(points):
        return 0
    else:
        return 1


def generate_clusters(data, kernel='poly', degree=2):
    """
    Generate clusters based on given data and method of
    Support Vector Clustering
    """
    classifier = one_class_classifier(data, kernel, degree)
    lookup = {}
    points = {}
    clusters = {}
    current_count = 0
    for i in range(len(data) - 1):
        for j in range(i+1, len(data)):
            lookup[(i, j)] = check_same_cluster(classifier, data[i],
                                                 data[j])
    for i in range(len(data)):
        if i not in points:
            current_count += 1
            points[i] = current_count
            pts = [i]
            for j in range(i+1, len(data)):
                if j not in points:
                    if lookup[tuple([i, j])] == 1:
                        points[j] = current_count
                        pts.append(j)
            clusters[current_count] = pts
    return clusters


if __name__ == '__main__':
    vectors = []
    for i in range(300):
        r = random.random()
        if r < 0.25:
            vectors.append(numpy.array([3 + random.random(), 5 + 0.6*random.random(), 13+2*random.random()]))
        elif r < 0.5:
            vectors.append(numpy.array([7 + random.random(), 10 + 1.6*random.random(), 14+1.2*random.random()]))
        elif r < 0.75:
            vectors.append(numpy.array([17 + 2*random.random(), 4 + 0.3*random.random(), 11*((3 + random.random())/3)]))
        else:
            vectors.append(numpy.array([random.random(), 18 - 1.6*random.random(), 10 - 0.5 * random.random()]))

    clusters = generate_clusters(vectors)

    for cluster in clusters:
        l = len(clusters[cluster])
        total = 0
        for x in clusters[cluster]:
            total += vectors[x]
        print total/l, len(clusters[cluster])
