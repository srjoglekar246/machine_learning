from gensim import corpora, models, similarities
from gensim.parsing import PorterStemmer
from numpy import array, sqrt

#Code for document preprocessing

#Build set of stopwords
f=open('stopwords')
stoplist = []
for line in f :
    stoplist.append(line[0:-1])
f.close()
stoplist = set(stoplist)

#Initialize stemmer
stemmer = PorterStemmer()


def to_vector_model(corpusfile, tfidf=False):
    """
    Convert a corpus to vector space model for processing

    If tfidf = True, returns a tf-idf based model

    Returns the model and dictionary of words.
    """
    
    f = open(corpusfile, "r")
    documents = f.readlines()
    f.close()
    #Remove stopwords and stem
    documents = [list(set([stemmer.stem(word) for word in \
                           document.lower().translate(None, '?,.:;()[]""').split() \
                  if word not in stoplist])) for document in documents]
    #Build a gensim dictionary
    dictionary = corpora.Dictionary(documents)
    #Build a vector space model from corpus
    corpus = [dictionary.doc2bow(text) for text in documents]
    if not tfidf:
        return corpus, dictionary
    #If tfidf is true, build tf-idf based model
    tfidf = models.TfidfModel(corpus)
    corpus = tfidf[corpus]

    return corpus, dictionary, tfidf

def to_lsi(vector_model, dictionary, concepts=None):
    """
    Converts a vector model to lower dimension space using Latent Semantic Indexing

    Return the LSI model and the transformed corpus
    """

    if not concepts:
        concepts = int(sqrt(len(dictionary)))
    lsi = models.LsiModel(vector_model, id2word=dictionary, num_topics=concepts)
    corpus_lsi = lsi[vector_model]

    lsi_vectors = []
    for x in corpus_lsi:
        temp = [0 for i in range(concepts)]
        for y in x:
            temp[y[0]] = y[1]
        lsi_vectors.append(array(temp))

    return lsi, lsi_vectors

    


