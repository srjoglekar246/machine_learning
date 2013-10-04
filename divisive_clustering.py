#Automatic clustering using Divisive clustering with K-Means at each step
#Optimum point to stop splitting is decided based on cluster qualities of
#children clusters in comparison to parent's quality

import numpy
import random


def generate_clusters(data, iterations=40):
    """
    Automatic clustering using divisive clustering
    """
    temp = divisive_clustering(data, [i for i in range(len(data))], 2, iterations)
    clusters = {}
    for i, x in enumerate(temp):
        clusters[i] = x
    return clusters


def divisive_clustering(vectors, to_cluster, sensitivity=1, n=2,
                        iterations=40, initial_quality=0):
    """
    Performs divisive clustering using K-Means algorithm.
    Uses cluster qualities as a stopping criterion.

    'vectors' is a list of Numpy arrays

    'to_cluster' is a list of the vector numbers from the 'vectors'
    list that need to be clustered

    'sensitivity' is a value from 0 to 1. The lower this value,
    the greater is the relaxation of the stopping criterion, and
    hence greater is the tendency to keep splitting

    'n' is the number of clusters to be formed at each level
    using flat clustering(K-Means)

    'iterations' is the fixed number of iterations to perform
    for K-Means E-M
    """

    if len(to_cluster) == 1:
        return [to_cluster]
    clusters, centroids = kmeansclustering(vectors, to_cluster, n, iterations)
    to_del = []
    for x in clusters:
        if len(clusters[x]) == 0:
            to_del.append(x)
    for x in to_del:
        del clusters[x]
        del centroids[x]
    qualities = cluster_qualities(vectors, clusters, centroids)
    new_quality = 0
    for x in clusters:
        new_quality += qualities[x] * float(len(clusters[x])) / float(len(to_cluster))
    if new_quality <= (initial_quality * sensitivity):
        return [to_cluster]
    main = [clusters[x] for x in clusters]
    if len(main) == 1:
        return main
    total = []
    for i, cluster in enumerate(main):
        total.extend(divisive_clustering(vectors, cluster, \
                                         sensitivity, n, iterations, qualities[i]))
    return total


def cluster_qualities(vectors, cluster_dict, centroids):
    """
    Returns normalized qualities of clusters formed using K-Means
    algorithm

    Cluster quality is taken to be (Avg inter distance/ Avg intra dist)

    Avg inter distance is the average distance between a certain cluster's
    centroid and the centroid of any other cluster. Should be maximised.

    Avg intra distance is the average distance between a certain cluster's
    centroid and any of the vectors assigned to it. Should be minimised.
    """

    qualities = []
    if len(cluster_dict) == 1:
        return [1000]
    cache = {}
    for i in range(len(centroids)):
        inter_dist = 0
        for j in range(len(centroids)):
            if (i,j) in cache:
                inter_dist += cache[(i, j)]
            elif (j, i) in cache:
                inter_dist += cache[(j, i)]
            else:
                cache[(i, j)] = numpy.linalg.norm(centroids[i]-centroids[j])
                inter_dist += cache[(i, j)]
        inter_dist = inter_dist/float(len(centroids)-1)
        intra_dist = 0
        for k in cluster_dict[i]:
            intra_dist += numpy.linalg.norm(vectors[k] - centroids[i])
        intra_dist = intra_dist/float(len(cluster_dict[i]))
        if intra_dist == 0:
            qualities.append(0)
        else:
            qualities.append(inter_dist/intra_dist)
    return qualities


def kmeansclustering(vectors, to_cluster, no_of_clusters, iterations=40):
    """
    Clustering according to the usual K-Means clustering algo

    Works with n number of iterations, not objective function

    to_cluster is the list of vector indices wrt the list 'vectors'
    that need to be clustered

    Returns a tuple of d, centroids where d is a dict mapping every
    cluster number to a list of the vector numbers that belong to it
    """
    
    centroids = {}
    temp = initial_centroids([vectors[i] for i in to_cluster], no_of_clusters)
    for i in range(no_of_clusters):
        centroids[i] = temp[i]
    for i in range(iterations):
        allotment = []
        c = {}
        for j in range(len(centroids)):
            c[j] = 0
        #Expectation step
        for j in to_cluster:
            temp = [k for k in range(len(centroids))]
            temp.sort(key=lambda x : numpy.linalg.norm(vectors[j] - \
                                                       centroids[x]))
            allotment.append(temp[0])
            c[temp[0]] += vectors[j]
        #Maximisation step
        for k in range(len(centroids)):
            n = allotment.count(k)
            if n != 0:
                c[k] = c[k]/float(n)
        centroids = c
    c = {}
    for k in range(len(centroids)):
        c[k] = []
    for i in range(len(allotment)):
        c[allotment[i]].append(to_cluster[i])
    return c, centroids


def initial_centroids(vectors_list, no_of_clusters):
    """
    Computes the initial centroids for K-Means algorithm,
    to avoid using a random set

    This method is the same as the one used in K-Means++ algorithm
    """
    noofvectors = len(vectors_list)
    centroids = []
    #Multiplication by 1.0001 to ensure non-zero denominators in actual algo
    centroids.append(vectors_list[int((noofvectors-1) * \
                                      random.random())] * 1.0001)
    weight_list = []
    while len(centroids) < no_of_clusters:
        weight_list = [min((numpy.linalg.norm(centroids[i] - x) ** 2) for \
                           i in range(len(centroids))) for x in vectors_list]
        randomnumber = sum(weight_list) * random.random()
        i = 0
        while i < noofvectors:
            randomnumber = randomnumber - weight_list[i]
            if randomnumber < 0:
                break
            i += 1
        centroids.append(vectors_list[i] * 1.0001)
    return centroids

