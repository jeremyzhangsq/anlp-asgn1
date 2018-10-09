# Here are some libraries you're likely to use. You might want/need others as well.
import re
import sys
import random
from math import log
import numpy as np
from collections import defaultdict

'''
============================================================
variable declaration part
============================================================
'''
VOCABULARY_SIZE = 30
'''
============================================================
function declaration part
============================================================
'''


'''
Read in file and split into training, validation, test set, 
also store trigram count into dictionary 'tri_counts' 
@:param infile: input file name
@:param ratios: a list contains split ratio [train_ratio, val_ratio, test_ratio,]  
@:returns: tri_counts, bi_counts, validation_list, test_list
'''
def read_and_store(infile, ratios):

    if sum(ratios) != 1:
        raise Exception("The sum of split ratio should be one` ")
    # dictionary to store counts of all trigrams in input
    # key (string): trigram e.g "and"
    # value (int): counts of the key
    tri_counts = defaultdict(int)

    # dictionary to store counts of previous two chars of all trigrams in input
    # key (string): previous two chars e.g "an"
    # value (int): counts of the key
    bi_counts = defaultdict(int)

    # dictionary to store relation graph of third char and previous two chars
    # key (string): previous two chars e.g "an"
    # value (set): all third char
    # adjacent_map = defaultdict(set)

    # validation list
    validation_list = []

    # test list
    test_list = []

    bins = np.cumsum(ratios)

    with open(infile) as f:
        for line in f:
            line = preprocess_line(line)  # implemented already.
            idx = np.digitize(random.random(), bins)
            # idx = 0 means the random variable is a training item
            if idx == 0:
                # include '##<start char>' and '<end char>##'
                for j in range(len(line) - (2)):
                    trigram = line[j:j + 3]
                    pre = line[j:j + 2]
                    # adjacent_map[pre].add(trigram[2])
                    tri_counts[trigram] += 1
                    bi_counts[pre] += 1
            # idx = 1 means the random variable is a validation item
            elif idx == 1:
                validation_list.append(line)
            # idx = 2 means the random variable is a testing item
            else:
                test_list.append(line)

    # new_map = defaultdict(list)
    # for key in adjacent_map:
    #     new_map[key] = list(adjacent_map[key])
    #
    # del adjacent_map

    return tri_counts, bi_counts, validation_list, test_list

'''
Task 1: removing unnecessary characters from each line
@:param line: a line of string from input file
@:return: a new line only with English alphabet, space, digits and dot character.
'''
def preprocess_line(line):
    rule = re.compile("[^\s.A-Za-z0-9]")
    line = rule.sub('', line)  # newline with only digit, alphabet, space and dot.
    line = re.sub("[1-9]", "0", line)  # replace 1-9 to 0
    line = "##"+line.lower()[:-1]+"#"  # add character'#' to specify start and stop
    return line

'''
Task 3: Estimate trigram probabilities by trigram count in training set. 
Store probabilities into global dictionary 'train_model'
The calculation formula is lec6, slide 13:

P(w3 | w1, w2) = ( Count(w1, w2, w3) + alpha ) / (Count(w1, w2) + alpha * v)  
where alpha is a tunable smoothing parameter, and v is the size of
@:param tri_cnts: a dictionary containing all tri-gram counts
@:param bi_cnts: a dictionary containing all bi-gram counts
@:param alpha: smoothing parameter alpha
@:return: language model
'''
def add_alpha_estimate(tri_cnts, bi_cnts, alpha):
    v = len    # dictionary to store the probability of each trigram
    model = defaultdict(float)
    for k in tri_counts:
        # TODO: the detail estimation need discussing
        pre = bi_cnts[k[:-1]]
        tri = tri_cnts[k]
        model[k] = (tri + alpha) / (pre + alpha * VOCABULARY_SIZE)

    return model
'''
Normalize given model such that the sum of probability is 1
@:param model: the distribution of model
@:return: the normalized distribution
'''
def normalize_model(model):
    total = sum(model.values())
    for k in model:
        model[k] = model[k] / total
    return model

'''
Task 3: Write back estimated probabilities to an output file
@:param outfile: filename
@:param train_model: language model to be written
@:return: write succeed or not
'''
def write_back_prob(outfile, train_model):
    with open(outfile, 'w') as f:
        for k in train_model:
            f.write("{}\t{:.3e}\n".format(k,train_model[k]))

'''
Read model from file and store into a dictionary
@:param infile: model name
@:return: a default dictionary containing the probability of trigrams 
'''
def read_model(infile):
    model = defaultdict(float)
    with open(infile) as f:
        for line in f:
            trigram, prob = line.split('\t')
            prob = float(prob.split('\n')[0])
            model[trigram] = prob
    return model

'''
Task 4: By using a LM and probability, generate a random sequence with length k
@:param lmodel: a language model stored in dictionary
@:param k: the size of output sequence
@:return: a random string sequence
'''
def generate_from_LM(lmodel, k):

    if k % 3 != 0:
        raise Exception("Character size k cannot be divided by 3")
    else:
        # As noted elsewhere, the ordering of keys and values accessed from
        # a dictionary is arbitrary. However we are guaranteed that keys()
        # and values() will use the *same* ordering, as long as we have not
        # modified the dictionary in between calling them.
        outcomes = np.array(list(lmodel.keys()))
        # normalize the probability distribution to 0-1
        total = sum(lmodel.values())
        alist = [a/total for a in lmodel.values()]
        probs = np.array(alist)

        # make an array with the cumulative sum of probabilities at each
        # index (ie prob. mass func)
        bins = np.cumsum(probs)
        # create N random #s from 0-1
        # digitize tells us which bin they fall into.
        idx = np.digitize(np.random.random_sample(int(k/3)), bins)
        # return the sequence of outcomes associated with that sequence of bins
        seq = ""
        for i in list(outcomes[idx]):
            seq += i
        return seq


'''
Task 5: Given a language model and a test paragraph, calculate the perplexity.
@:param model: a language model stored in dictionary
@:param testfile: the name of test file
@:param flag: flag is 0 means test data is from file, 1 means it is from dictionary
@:return: the perplexity
'''
def get_perplexity(model, bi_gram, alpha, testfile, flag):
    logsum = 0
    cnt = 0
    if flag == 0:
        fo = open(testfile, "r")
        lines = fo.readlines()
        fo.close()
    else:
        lines = testfile
    # calculate the log probability of each sentence and sum them up
    for line in lines:
        if flag == 0:
            line = preprocess_line(line)
        cnt += 1
        logp = get_sentence_log_prob(model, bi_gram, alpha, line)
        logsum += logp
    # get cross entropy
    cross_entropy = -logsum / cnt
    # get perplexity of whole test paragraph
    pp = np.exp2(cross_entropy)
    return pp

'''
Given a language model and a sentence, calculate the log probablity.
@:param model: a language model stored in dictionary
@:param line: the sentence
@:return: the log probability of this sentence
'''
def get_sentence_log_prob(model,bi_gram, alpha, line):
    p = 0
    for j in range(len(line) - (2)):
        trigram = line[j:j + 3]
        # TODO: what if current trigram is not in training model? We skip unknown word here
        if model[trigram] != 0:
            prob = model[trigram]
        else:
            prob = alpha / (bi_gram[trigram[:-1]] + VOCABULARY_SIZE*alpha)
        p += np.log2(prob)
    return p


'''
Try different alpha and return the language model with least perplexity
@:param tri_cnts: a dictionary containing all tri-gram counts
@:param bi_cnts: a dictionary containing all bi-gram counts
@:param validation_set: list of lines for validation
@:return best_alpha, best_perplexity, best_model
'''
def adding_alpha_training_LM(tri_counts, bi_counts, validaion_set):

    best_alpha = 0
    best_perplexity = np.inf
    best_model = -1
    for a in np.arange(0.01, 1, 0.01):
        training_model = add_alpha_estimate(tri_counts, bi_counts, alpha=a)
        cur = get_perplexity(training_model, bi_counts, a, validaion_set, flag=1)
        print("alpha:",round(a,2),"perplexity:",cur)
        if cur < best_perplexity:
            best_alpha = a
            best_perplexity = cur
            best_model = training_model

    print("======================best===========================")
    print("alpha:", round(best_alpha, 2), "perplexity:", best_perplexity)
    return best_alpha, best_perplexity, best_model

# TODO: Johanna's task
# iteratively test all possible lambda1,2,3
def interpolation_training_LM(model, validation_set):

    pass

# TODO: Johanna's task
def interpolation_estimate(tri_counts, bi_counts, lam1, lam2):

    pass

'''
Some example code that prints out the counts. For small input files
the counts are easy to look at but for larger files you can redirect
to an output file (see Lab 1).
@:param param1: input file name 
@:return void
'''
def show(infile):
    print("Trigram counts in ", infile, ", sorted alphabetically:")
    for trigram in sorted(tri_counts.keys()):
        print(trigram, ": ", tri_counts[trigram])
    print("Trigram counts in ", infile, ", sorted numerically:")
    for tri_count in sorted(tri_counts.items(), key=lambda x: x[1], reverse=True):
        print(tri_count[0], ": ", str(tri_count[1]))






'''
============================================================
program running part
============================================================
'''
if __name__ == '__main__':

    # here we make sure the user provides a training filename when
    # calling this program, otherwise exit with a usage error.
    if len(sys.argv) != 2:
        print("Usage: ", sys.argv[0], "<training_file>")
        sys.exit(1)

    infile = sys.argv[1]  # get input argument: the training file

    tri_counts, bi_counts, validation_list, test_list = read_and_store(infile, [0.8,0.1,0.1])
    best_alpha, best_perplexity, best_model = adding_alpha_training_LM(tri_counts, bi_counts, validation_list)
    write_back_prob("outfile.txt", best_model)
    model = read_model("model-br.en")
    seq = generate_from_LM(best_model, 300)
    # print(training_model)
    # show(infile)







