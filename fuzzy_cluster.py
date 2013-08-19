import numpy
import random

def cluster_lists(vectors_list, no_of_clusters, fuzzy_cooef, delta):
    """
    vectors_list is a list of numpy arrays.
    This function clusters the list into a number of clusters given by
    no_of_clusters.
    The fuzzy coefficient and delta parameters as specified are used in
    the clustering procedure.

    Returns a dictionary having key as the vector number, and the value
    as a list of membership degrees of the vector to the various clusters AND
    the centroids as list of numpy arrays
    """
    fuzzy_cooef = float(fuzzy_cooef)
    cooef_value = (2.0/ (fuzzy_cooef - 1))
    noofvectors = len(vectors_list)
    temp_delta = 1000
    membership_matrix = numpy.zeros((noofvectors, no_of_clusters))
    #Initialise random membership matrix
    centroids = initial_centroids(vectors_list, no_of_clusters)
    #Do iterative process till temp_delta goes below delta or number of iteratione go too high
    count = 1
    objective = 0
    while temp_delta > delta:
        if count > 50:
            break
        #Redefine centroids
        if (count > 1):
            for i in range(no_of_clusters):
                coordinates = 0
                sumofdegrees = 0
                for j in range(noofvectors):
                    temp_value = membership_matrix[j][i] ** fuzzy_cooef
                    coordinates += temp_value * vectors_list[j]
                    sumofdegrees += temp_value
                centroids[i] = coordinates / float(sumofdegrees)
        #Update membership matrix
        temp_value = [0 for i in range(noofvectors)]
        for i in range(noofvectors):
            for j in range(no_of_clusters):
                temp_value[i] += 1.0 / (numpy.linalg.norm(vectors_list[i] - centroids[j]) ** cooef_value)
        for i in range(noofvectors):
            for j in range(no_of_clusters):
                membership_matrix[i][j] = 1.0 / (numpy.linalg.norm(vectors_list[i] - centroids[j]) ** cooef_value * temp_value[i])
        #Calculate new objective and temp_delta
        new_objective = _objective_function(centroids, vectors_list, membership_matrix, fuzzy_cooef)
        temp_delta = abs(objective - new_objective)
        if count == 1:
            temp_delta = delta + 1
        #Redefine objective
        objective = new_objective
        count += 1
    #Process results and return
    result_dict = {}
    for i in range(noofvectors):
        result_dict[i] = list(membership_matrix[i])
    return result_dict, centroids

def initial_centroids(vectors_list, no_of_clusters):
    """
    Computes the initial centroids for Fuzzy-C Means algorithm,
    to avoid using a random set

    This method is the same as the one used in K-Means++ algo
    """
    noofvectors = len(vectors_list)
    centroids = []
    centroids.append(vectors_list[int((noofvectors-1) * random.random())] * 1.0023)
    weight_list = []
    while len(centroids) < no_of_clusters:
        weight_list = [min((numpy.linalg.norm(centroids[i] - x) ** 2) for i in range(len(centroids))) for x in vectors_list]
        randomnumber = sum(weight_list) * random.random()
        i = 0
        while i < noofvectors:
            randomnumber = randomnumber - weight_list[i]
            if randomnumber < 0:
                break
            i += 1
        centroids.append(vectors_list[i] * 1.0023)
    return centroids
    

def _objective_function(centroids, vectors, membership_matrix, fuzzy_cooef):
    """
    Calculates the value of the objective function that has to be
    minimized
    """
    objective = 0
    for i in range(len(vectors)):
        for j in range(len(centroids)):
            objective = objective + \
                        (membership_matrix[i][j] ** fuzzy_cooef) * (numpy.linalg.norm(vectors[i] - centroids[j]))**2
    return objective
                
