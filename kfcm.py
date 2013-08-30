import numpy
import random

def kernel_fcm(vectors_list, no_of_clusters, fuzzy_cooef, delta, max_iterations=50):
    """
    Kernelized FCM for clustering of data

    Based on paper by Zhang and Chen, with some modifications based on the K-Means++
    algorithm, for better initialization of pseudo-random cluster centroids

    Uses the rbf kernel, this method can be re-factored to use any kernel that
    satisfies K(x, x) == 1
    
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
    cooef_value = (1.0/ (fuzzy_cooef - 1))
    noofvectors = len(vectors_list)
    temp_delta = 1000
    membership_matrix = numpy.zeros((noofvectors, no_of_clusters))
    kernel_cache = numpy.zeros((noofvectors, no_of_clusters))
    #Initialise random membership matrix
    centroids = initial_centroids(vectors_list, no_of_clusters)
    #Do iterative process till temp_delta goes below delta or number of iteratione go too high
    count = 1
    objective = 0
    while temp_delta > delta:
        if count > max_iterations:
            break
        #Update membership matrix
        temp_cache = numpy.zeros((noofvectors, no_of_clusters))
        total_values = []
        for i in range(noofvectors):
            s = 0
            for j in range(no_of_clusters):
                kernel_cache[i][j] = rbf_kernel(vectors_list[i], centroids[j])
                try:
                    temp_value = 1.0/(1 - kernel_cache[i][j])**cooef_value
                except:
                    break
                temp_cache[i][j] = temp_value
                s += temp_value
            total_values.append(s)
        for i in range(noofvectors):
            for j in range(no_of_clusters):
                membership_matrix[i][j] = temp_cache[i][j] / total_values[i]
        #Redefine centroids
        for j in range(no_of_clusters):
            numerator = 0
            denominator = 0
            for i in range(noofvectors):
                temp = membership_matrix[i][j]**fuzzy_cooef * \
                       kernel_cache[i][j]
                numerator += temp * vectors_list[i]
                denominator += temp
            centroids[j] = numerator/denominator
        #Calculate new objective and temp_delta
        new_objective = _objective_function(centroids, vectors_list,
                                            membership_matrix, fuzzy_cooef)
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

    This method is the same as the one used in K-Means++ algorithm
    """
    noofvectors = len(vectors_list)
    centroids = []
    centroids.append(vectors_list[int((noofvectors-1) * random.random())] * 1.0001)
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
    

def _objective_function(centroids, vectors, membership_matrix, fuzzy_cooef):
    """
    Calculates the value of the objective function that has to be
    minimized
    """
    objective = 0
    for i in range(len(vectors)):
        for j in range(len(centroids)):
            objective = objective + \
                        (membership_matrix[i][j] ** fuzzy_cooef) * \
                        (1 - rbf_kernel(vectors[i], centroids[j]))
    return 2 * objective

                
def rbf_kernel(vector1, vector2, a=1, b=2, sigma=2):
    """
    Represents the rbf kernel
    If parameters a, b are not specified, it defaults to the Gaussian kernel
    """
    exp_numerator = 0
    for i in range(len(vector1)):
        exp_numerator -= (float(abs(vector1[i]**a - vector2[i]**a)))**b
    return numpy.exp(exp_numerator/sigma**2)
