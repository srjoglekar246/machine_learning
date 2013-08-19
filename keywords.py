#Use top_keywords function to get top n most relevant keywords from a
#single document
#Based on a paper by Yutaka Matsuo, Mitsuru Ishizuka

from gensim.parsing import PorterStemmer


#Generate set of stopwords
#Build set of stopwords
f=open('stopwords')
stoplist = []
for line in f :
    stoplist.append(line[0:-1])
f.close()
stoplist = set(stoplist)

#Initialize stemmer
stemmer = PorterStemmer()

#Build stem lookup
#This lookup prevents words with same stemmed form
#from being treated as different words in the corpus
class stemmingHelper(object):
    def __init__(self):
        self.word_lookup = {}

    def __call__(self, word):
        stemmed = stemmer.stem(word)
        if stemmed not in self.word_lookup:
            self.word_lookup[stemmed] = word
            return word
        else:
            return self.word_lookup[stemmed]

primary_form = stemmingHelper()

def get_word_sets(filename):
    """ Returns dict of total vocabulary and dicts of sentence vocabularies """
    f = open(filename, 'r')
    sentences = []
    for line in f.readlines():
        temp = line.strip().lower().translate(None,
                                              '?,:;()[]"`-|><^%"*1234567890').split('.')
        for i, x in enumerate(temp):
            if len(x) > 2:
                sentences.append(x)
    sentence_words = []
    vocabulary = {}

    for x in sentences:
        sentence_vocab = {}
        temp = x.split()
        for i in range(len(temp)-1):
            if temp[i] not in stoplist and temp[i+1] not in stoplist:
                bigram = ' '.join([temp[i], temp[i+1]])
                sentence_vocab[bigram] = sentence_vocab.get(bigram, 0) + 1
                vocabulary[bigram] = vocabulary.get(bigram, 0) + 1
            if temp[i] not in stoplist and len(temp[i]) > 2:
                word = primary_form(temp[i])
                sentence_vocab[word] = sentence_vocab.get(word, 0) + 1
                vocabulary[word] = vocabulary.get(word, 0) + 1
        try:
            if temp[-1] not in stoplist and len(temp[-1]) > 2:
                word = primary_form(temp[-1])
                sentence_vocab[word] = sentence_vocab.get(word, 0) + 1
                vocabulary[word] = vocabulary.get(word, 0) + 1
        except:
            pass
        sentence_words.append(sentence_vocab)
            
    return vocabulary, sentence_words


def get_param_matrices(vocabulary, sentence_words, n=None):
    """
    Returns -
    1. Top n most frequent words
    2. co-occurence matrix wrt top-n words(dict)
    3. vector of Pg of most-frequent n words(dict)
    4. nw of each word(dict)
    """

    if n is None or n > len(vocabulary):
        n = int(0.33 * len(vocabulary))
    topwords = vocabulary.keys()
    topwords.sort(key = lambda x: vocabulary[x], reverse = True)
    topwords = topwords[:n]
    nw = {}
    co_occur = {}
    for x in vocabulary:
        co_occur[x] = [0 for i in range(len(topwords))]
    
    for sentence in sentence_words:
        total_words = sum(sentence.values())
        top_indices = []
        for word in sentence:
            if word in topwords:
                top_indices.append(topwords.index(word))
        for word in sentence:
            nw[word] = nw.get(word, 0) + total_words
            for index in top_indices:
                co_occur[word][index] += sentence[word] * \
                                         sentence[topwords[index]]

    Pg = {}
    N = sum(vocabulary.values())
    for x in topwords:
        Pg[x] = float(nw[x])/N
    return topwords, co_occur, Pg, nw


def get_main_values(vocabulary, topwords, co_occur, Pg, nw):
    """ Calculates the main comparison values to find top keywords """
    result = {}
    N = sum(vocabulary.values())
    for word in vocabulary:
        result[word] = 0
        for x in Pg:
            expected_cooccur = nw[word] * Pg[x]
            result[word] += (co_occur[word][topwords.index(x)] - \
                             expected_cooccur)**2/ float(expected_cooccur)
        result[word] *= vocabulary[word]/float(N)
    return result


def top_keywords(filename, n=5):
    vocabulary, sentence_words = get_word_sets(filename)
    topwords, co_occur, Pg, nw = get_param_matrices(vocabulary, sentence_words)
    result = get_main_values(vocabulary, topwords, co_occur, Pg, nw)
    if n> len(vocabulary):
        raise ValueError("Value of n cannot exceed vocabulary size :" + `len(vocabulary)`)
    words = result.keys()
    words.sort(key = lambda x: result[x], reverse=True)
    count = 0
    i = 0
    wordset = set([])
    topn = []
    while count < n:
        if ' ' in words[i]:
            for x in words[i].split():
                if x in wordset:
                    try:
                        topn.remove(x)
                        count -= 1
                    except:
                        pass
                else:
                    wordset.add(x)
            topn.append(words[i])
            count += 1
        else:
            if words[i] not in wordset:
                topn.append(words[i])
                wordset.add(words[i])
                count += 1
        i += 1
    return topn
        
