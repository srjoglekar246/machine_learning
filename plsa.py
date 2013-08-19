from numpy import log, array
from random import random
from gensim.parsing import PorterStemmer

#Initialize stemmer
stemmer = PorterStemmer()

def total_1_list(n):
    out = [random() for i in range(n)]
    outsum = sum(out)
    return [float(x)/outsum for x in out]

def plsi_initialize(documents, topics):
    """
    Initialization of the parameters as needed by plsi function
    Returns Doc_list, Topic_list and list of all words

    Parameters
    ==========

    documents : list
        List of strings

    topics : int
        The number of topics

    """
    
    Doc_list = [{} for i in range(len(documents))]
    word_lookup = {}
    count = 0
    for i, doc in enumerate(documents):
        words = [stemmer.stem(word) for word \
                 in doc.strip().lower().translate(
                     None, '?,:;()[]"`|><^\'"*1234567890').split()]
        for word in words:
            if word not in word_lookup:
                count += 1
                word_lookup[word] = count
            Doc_list[i][word_lookup[word]] = \
                                           Doc_list[i].get(word_lookup[word], 0) + 1
    Topic_list = {}
    topic_probabilities = [1.0/topics for t in range(topics)]
    for k in range(topics):
        d_list = total_1_list(len(documents))
        w_list = total_1_list(len(word_lookup))
        Topic_list[k] = [topic_probabilities[k], d_list, w_list]

    words = word_lookup.keys()
    words.sort(key = lambda x: word_lookup[x])
    return Doc_list, Topic_list, words


def plsi(Doc_list, Topic_list, max_iterations, threshold):
    """
    The plsi expectation-maximization algorithm.
    Returns lists of topic probabilities, document vectors and word
    vectors. Vectors are in the form of numpy arrays.

    Parameters
    ==========

    Doc_list : list of dictionaries
        Each dictionary in list corresponds to a document.
        Do Doc_list[i].get(j, 0) to get n(di, wj)

    Topic_list : dictionary
        Dictionary of topics.
        Key is topic number and value is a list having -
        [P(zi), [P(d1|zi, P(d2|zi),...], [P(w1|zi), ...]]

    max_iterations, threshold : int, float
        Parameters for the E-M process

    """

    count = 1
    #List of log_likelihood values of previous and current iterations
    log_likelihood = [1000, 10000]
    #Get the basic values
    noofdocs = len(Doc_list)
    nooftopics = len(Topic_list)
    noofwords = len(Topic_list[0][2])
    doc_word_sum = 0.0
    for i in range(noofdocs):
        for j in range(noofwords):
            doc_word_sum += Doc_list[i].get(j, 0)
    #Initialize latent variable values P(z|d, w)
    Topic_probability = {}
    for i in range(noofdocs):
        for j in range(noofwords):
            Topic_probability[(i, j)] = \
                                  [0 for t in range(nooftopics)]

    print "Starting E-M iterations..."

    while 1:
        if count > max_iterations:
            break

        #Expectation step
        #Re-calculate P(z|d, w) for every d,w,z
        
        for i in range(noofdocs):
            for j in range(noofwords):
                topic_list = [Topic_list[t][0] * \
                              Topic_list[t][1][i] * \
                              Topic_list[t][2][j] for t \
                              in range(nooftopics)]
                sum_prob = sum(topic_list)
                for k in range(nooftopics):
                    if sum_prob == 0:
                        Topic_probability[(i, j)][k] = random()
                    else:
                        Topic_probability[(i, j)][k] = topic_list[k] / float(sum_prob)

        #Maximisation step
        #Re-calculate P(z), P(w|z), P(d|z)

        temp_dict = {}

        for i in range(noofdocs):
            for j in range(noofwords):
                for k in range(nooftopics):
                    temp_dict[(i, j, k)] = Doc_list[i].get(j, 0) * \
                                           Topic_probability[(i, j)][k]

        for k in range(nooftopics):
            topic_sum = 0
            doc_list = []
            for i in range(noofdocs):
                doc_sum = 0
                for j in range(noofwords):
                    topic_sum += temp_dict[(i, j, k)]
                    doc_sum += temp_dict[(i, j, k)]
                doc_list.append(doc_sum)
            if topic_sum == 0:
                doc_list = [1.0/noofdocs for g in range(noofdocs)]
            else:
                doc_list = [float(x) / topic_sum for x in doc_list]
            Topic_list[k][1] = doc_list

            #Store P(z)
            Topic_list[k][0] = float(topic_sum) / doc_word_sum

            word_list = []
            for j in range(noofwords):
                word_sum = 0
                for i in range(noofdocs):
                    word_sum += temp_dict[(i, j, k)]
                word_list.append(word_sum)
            word_list = [float(x) / topic_sum for x in word_list]
            Topic_list[k][2] = word_list

        #Calculate log likelihood function value
        value = 0
        for i in range(noofdocs):
            for j in range(noofwords):
                prob = 0
                for k in range(nooftopics):
                    prob += Topic_list[k][1][i] * Topic_list[k][2][j] * Topic_list[k][0]
                if prob != 0:
                    value += log(prob) * Doc_list[i].get(j, 0)
        log_likelihood[0] = log_likelihood[1]
        log_likelihood[1] = value

        count += 1

    print "Finished E-M..."
    print "Number of iterations completed - " + `count-1`

    #List of probabilities of all topics
    topics = [Topic_list[x][0] for x in Topic_list]
    #Construct document vectors
    docs = []
    for i in range(noofdocs):
        doc_temp = [Topic_list[x][0] * Topic_list[x][1][i] for x in Topic_list]
        doc_temp = [float(x)/sum(doc_temp) for x in doc_temp]
        docs.append(array(doc_temp))
    #Construct word vectors
    words = []
    for j in range(noofwords):
        word_temp = [Topic_list[x][0] * Topic_list[x][2][j] for x in Topic_list]
        if sum(word_temp) == 0:
            words.append(array([0 for i in range(nooftopics)]))
        else:
            word_temp = [float(x)/sum(word_temp) for x in word_temp]
            words.append(array(word_temp))

    return topics, docs, words


if __name__ == '__main__':
    f = open('/home/sachin/doc_clustering/corpus', 'r')
    documents = f.readlines()
    doc_list, topic_list, words = plsi_initialize(documents, 2)
    topics, docs, words = plsi(doc_list, topic_list, 300, 0.0000001)
