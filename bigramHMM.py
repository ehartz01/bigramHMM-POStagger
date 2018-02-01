import sys
from collections import defaultdict
from math import log, exp

from nltk.corpus import treebank
from nltk.tag.util import untag  # Untags a tagged sentence. 

unknown_token = "<UNK>"  # unknown word token.
start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.

""" Remove trace tokens and tags from the treebank as these are not necessary.
"""
def TreebankNoTraces():
    return [[x for x in sent if x[1] != "-NONE-"] for sent in treebank.tagged_sents()]

#takes vocabulary and a training set as input
#adds all tokens appearing more than once to the vocab
#returns the new vocab
def BuildVocab(training):
    first_occur = set()
    s_occur = set()
    for sent in training:
        for token in untag(sent):
            if (token not in s_occur and token in first_occur):
                s_occur.add(token)
            if (token not in first_occur):
                first_occur.add(token)
    return s_occur

#takes a corpus which is a list of lists of words and set which is the vocabulary
#adds sentence begin and end tokens, replaces unknown words with the UNK token
def PreprocessText(corpus, vocabulary):
    new_corp = []
    for sent in corpus: 
        new_sent = []
        new_sent.append((start_token , start_token))
        for token in sent:
            if (token[0] in vocabulary):
                new_sent.append(token)
            if (token[0] not in vocabulary):
                new_sent.append((unknown_token,token[1]))
        new_sent.append((end_token , end_token))
        new_corp.append(new_sent)
    return new_corp

class BigramHMM:
    def __init__(self):
        """ Implement:
        self.transition, the A matrix of the HMM: a_{ij} = P(t_j | t_i)
        self.emissions, the B matrix of the HMM: b_{ii} = P(w_i | t_i)
        self.dictionary, a dictionary that maps a word to the set of possible tags
        """
        self.ttcounts = defaultdict(float)
        self.tagcounts = defaultdict(float)
        self.transition = defaultdict(lambda: log(1.0/45))

        self.wordcounts = defaultdict(float)
        self.wtcounts = defaultdict(float)
        self.emissions = defaultdict(float)

        self.dictionary = defaultdict(set)

        self.capcounts = defaultdict(float)
        self.capprobs = defaultdict(float)
        
    def Train(self, training_set):
        """ 
        1. Estimate the A matrix a_{ij} = P(t_j | t_i)
        2. Estimate the B matrix b_{ii} = P(w_i | t_i)
        3. Compute the tag dictionary 
        """
        last_tok = ("","")
        #take wt tt w and t counts
        for sent in training_set:
            last_tok = ("","")
            for tok in sent:
                self.wtcounts[tok] += 1.0
                self.tagcounts[tok[1]] += 1.0
                if tok[0][0].isupper():
                    self.capcounts[tok[1]] += 1.0
                if tok[1] not in self.dictionary[tok[0]]:
                    self.dictionary[tok[0]].add(tok[1])
                if (last_tok != ("","")):
                    self.ttcounts[(last_tok[1],tok[1])] += 1.0
                last_tok = tok
        #estimate A matrix (transition from tag to tag, note we stored it as (preceding tag, tag))
        for k,v in self.capcounts.iteritems():
            self.capprobs[k] = log( v / self.tagcounts[k] )
        for k, v in self.ttcounts.iteritems():
            self.transition[k] = log( (v +1) / (self.tagcounts[k[0]] + 45) )
        #estimate B matrix (word given its tag)
        for k, v in self.wtcounts.iteritems():
            self.emissions[k] = log( v / self.tagcounts[k[1]] ) #+ self.capprobs[k[1]]

    def ComputePercentAmbiguous(self, data_set):
        """ Compute the percentage of tokens in data_set that have more than one tag according to self.dictionary. """
        num_ambig = 0.0
        num_toks = 0.0
        for sent in data_set:
            for tok in sent:
                num_toks +=1.0
                if len(self.dictionary[tok[0]]) > 1:
                    num_ambig += 1.0
            num_toks += -2.0
        return 100*num_ambig/num_toks
        
    def JointProbability(self, sent):
        """ Compute the joint probability of the words and tags of a tagged sentence. """
        last_tok = ("","")
        productA = 0.0
        productB = 0.0
        for token in sent:
            if last_tok != ("",""):
                productB += self.emissions[token]
                productA += self.transition[(last_tok[1],token[1])]
            last_tok = token
        final_prod = productA + productB
        return exp(final_prod) #need to exp this
        
    def Viterbi(self, sent):
        """ Find the probability and identity of the most likely tag sequence given the sentence. """
        tagged_sent = defaultdict(str)
        viterbi = defaultdict(float)
        backpointers = {}  #keys are states values are other states

        #handle the first step
        for tag in self.dictionary[sent[1][0]]:
            viterbi[(tag,1)] = self.transition[(sent[0][1], tag)] + self.emissions[(sent[1][0], tag)]
            backpointers[(tag,1)] = (start_token, 0)
        viterbi[(start_token, 0)] = 0

        #recursion step
        counter = 1
        for token in sent[2:]:
            counter += 1
            for tag in self.dictionary[token[0]]:
                find_max = defaultdict(float)
                for state in viterbi:
                    if (state[1] == counter-1):
                        find_max[state]= viterbi[state] + self.transition[(state[0],tag)] + self.emissions[(token[0],tag)]
                if len(find_max.values()) >= 1:
                    viterbi[(tag, counter)] = max(find_max.values())
                    backpointers[(tag, counter)] = max(find_max, key=lambda i: find_max[i])

        #termination step should be included in the recursion
        #traverse the backtrace and tag the sentence
        lookstate = (end_token,len(sent)-1)
        for k in backpointers.keys():
                tagged_sent[backpointers[lookstate][1]] = backpointers[lookstate][0]
                if backpointers[lookstate] != (start_token, 0):
                    lookstate = backpointers[lookstate]
        tagged_sent[0] = start_token

        final_tagged = []
        for count, k in enumerate(tagged_sent.values()):
            final_tagged.append((sent[count][0], k))
        final_tagged.append((end_token,end_token))
        return final_tagged

    def Test(self, test_set):
        """ Use Viterbi and predict the most likely tag sequence for every sentence. Return a re-tagged test_set. """
        tagged_set = []
        for sent in test_set:
            tagged_set.append(self.Viterbi(sent))
        return tagged_set

def MostCommonClassBaseline(training_set, test_set):
    """ Implement the most common class baseline for POS tagging. Return the test set tagged according to this baseline. """
    wtcounts = defaultdict(float)
    #take the counts
    for sent in training_set:
        for tok in sent:
            wtcounts[tok] += 1.0

    #tag the test by just looking at wtcounts
    tagged_test = []
    for sent in test_set:
        new_sent = []
        for tok in sent:
            max_key = FindMaxKey(tok, wtcounts)
            new_sent.append((tok[0],max_key[1]))
        tagged_test.append(new_sent)
    return tagged_test

def FindMaxKey(tok, wtcounts):
    maxlist = {}
    for k,v in wtcounts.iteritems():
        if (k[0] == tok[0]):
            maxlist[k] = v
    return max(maxlist, key=lambda i: maxlist[i])

    
def ComputeAccuracy(test_set, test_set_predicted, hmm):
    """ Using the gold standard tags in test_set, compute the sentence and tagging accuracy of test_set_predicted. """
    correct_sentences = 0.0
    correct_tags = 0.0
    num_toke = 0.0
    for count, sent in enumerate(test_set):
        if sent == test_set_predicted[count]:
            correct_sentences += 1
        for t_count, token in enumerate(sent):
            num_toke += 1
            if token == test_set_predicted[count][t_count]:
                correct_tags +=1
        correct_tags += -2
        num_toke += -2
    print "Percent sentence accuracy: " + str(100*correct_sentences/len(test_set))
    print "Percent tag accuracy: " + str(100*correct_tags/num_toke) 

def Confusion(test_set, test_set_predicted):
    confusion = defaultdict(float) #keys are incorrect comma correct tokens
    for count, sent in enumerate(test_set):
        for t_count, token in enumerate(sent):
            if token != test_set_predicted[count][t_count]:
                confusion[(token[1], test_set_predicted[count][t_count][1])] += 1
    return confusion

def main():
    treebank_tagged_sents = TreebankNoTraces()  # Remove trace tokens. 
    training_set = treebank_tagged_sents[:3000]  # This is the train-test split that we will use. 
    test_set = treebank_tagged_sents[3000:]
    
    """ Transform the data sets by eliminating unknown words and adding sentence boundary tokens.
    """
    vocabulary = BuildVocab(training_set)
    training_set_prep = PreprocessText(training_set, vocabulary)
    test_set_prep = PreprocessText(test_set, vocabulary)
    
    """ Print the first sentence of each data set.
    """
    print " ".join(untag(training_set_prep[0]))  # See nltk.tag.util module.
    print " ".join(untag(test_set_prep[0]))

    """ Estimate Bigram HMM from the training set, report level of ambiguity.
    """
    bigram_hmm = BigramHMM()
    bigram_hmm.Train(training_set_prep)

    test_set_predicted_baseline = MostCommonClassBaseline(training_set_prep, test_set_prep)
    print "--- Most common class baseline accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_predicted_baseline, bigram_hmm)

    print "Percent tag ambiguity in training set is %.2f%%." %bigram_hmm.ComputePercentAmbiguous(training_set_prep)

    print "Joint probability of the first sentence is %s." %bigram_hmm.JointProbability(training_set_prep[0])
    """print "sanity check: "
    product = 0.0
    for sent in training_set_prep:
        product += bigram_hmm.JointProbability(sent)
    print product"""
    #print bigram_hmm.Viterbi(test_set_prep[0])

    test_set_predicted_bigram_hmm = bigram_hmm.Test(test_set_prep)
    print "--- Bigram HMM accuracy ---"                                 
    ComputeAccuracy(test_set_prep, test_set_predicted_bigram_hmm, bigram_hmm)
    confusion = Confusion(test_set_prep, test_set_predicted_bigram_hmm)
    print max(confusion.values())
    print max(confusion, key=lambda i: confusion[i])
    print confusion[(u'DT', u'LS')]
    print confusion

"""    
    

    #Use the Bigram HMM to predict tags for the test set. Report accuracy of the predicted tags.
    
    test_set_predicted_bigram_hmm = bigram_hmm.Test(test_set_prep)
    print "--- Bigram HMM accuracy ---"                                 
    ComputeAccuracy(test_set_prep, test_set_predicted_bigram_hmm)    """

if __name__ == "__main__": 
    main()
    