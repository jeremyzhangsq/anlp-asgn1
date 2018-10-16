# Here are some libraries you're likely to use. You might want/need others as well.
import re
import sys
import random
from math import log
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import itertools
import string
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
        raise Exception("The sum of split ratio should be 1")
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

    uni_counts = defaultdict(int)

    # validation list
    validation_list = []

    # test list
    test_list = []

    train_list = []

    bins = np.cumsum(ratios)

    with open(infile) as f:
        for line in f:
            line = preprocess_line(line)  # implemented already.
            idx = np.digitize(random.random(), bins)
            # idx = 0 means the random variable is a training item
            if idx == 0:
                train_list.append(line)
                # include '##<start char>' and '<end char>##' -- included in preprocessing?
                for j in range(len(line) - (2)):
                    trigram = line[j:j + 3]
                    pre = line[j:j + 2]
                    uni = line[j:j + 1]
                    # adjacent_map[pre].add(trigram[2])
                    tri_counts[trigram] += 1
                    bi_counts[pre] += 1
                    uni_counts[uni] += 1
            # idx = 1 means the random variable is a validation item
            elif idx == 1:
                validation_list.append(line)
            # idx = 2 means the random variable is a testing item
            else:
                test_list.append(line)


    return train_list, tri_counts, bi_counts, uni_counts, validation_list, test_list

"""
Inserts missing trigrams and assigns all impossible combinations to <UNK>
@:param counts: counts dictionary
@:return counts: counts dictionary with possible but unseen trigrams added with count 0
"""

def missing_items (tri_counts, bi_counts, uni_counts):

    all_letters = list(string.ascii_lowercase)
    all_nonletters = [" ", "#", "."]
    all_letters.extend(all_nonletters)
    all_combs_bi = itertools.product(all_letters, repeat=2)
    all_combs_tri = itertools.product(all_letters, repeat=3)

    all_combs_bi_list = []
    all_combs_tri_list = []

    for combo in all_combs_bi:
        combo = "".join(list(combo))
        all_combs_bi_list.append(combo)

    for combo in all_combs_tri:
        combo = "".join(list(combo))
        all_combs_tri_list.append(combo)

    sonority_constraint1 = ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "t", "x"]
    sonority_constraint2 = ["b", "d", "g", "p", "t", "k", "q", "x"] # Stops and affricates
    impossible_comb = []
    for char1 in sonority_constraint1:
        for char2 in sonority_constraint2:
            impossible_comb.extend(["#", char1, char2])
            impossible_comb.extend([char2, char1, "#"])

    non_letter_comb = itertools.product(all_nonletters, repeat=3)
    impossible_comb.append(non_letter_comb)

    for i in all_combs_bi_list:
        if i not in bi_counts.keys():
            bi_counts[i] = 0
    for j in all_letters:
        if j not in uni_counts.keys():
            uni_counts[j] = 0
    for k in all_combs_tri_list:
        if k not in tri_counts:
            if k not in impossible_comb:
                tri_counts[k] = 0
            else:
                tri_counts["<UNK>"] += 1
                bi_counts["<UNK>"] += 1
                uni_counts["<UNK>"] += 1
    adj_map = defaultdict(set)
    for i in tri_counts:
        adj_map[i[:-1]].add(i[2])
    return adj_map, tri_counts, bi_counts, uni_counts

'''
Task 1: removing unnecessary characters from each line
@:param line: a line of string from input file
@:return: a new line only with English alphabet, space, digits and dot character.
'''
def preprocess_line(line):
    rule = re.compile("[^\s.A-Za-z0-9]")
    # newline with only digit, alphabet, space and dot.
    line = rule.sub('', line)
    # replace excess number of white spaces with one white space
    line = re.sub('\s{2,}', ' ', line)
    # Remove abbreviations enclosed by whitespace (substitue with whitespace)
    line = re.sub('\s[A-Z]{2,}\s', ' ', line)
    # Remove abbreviations that precede all punctuation (substitute with .)
    line = re.sub("\s([A-Z]{2,}).", '.', line)
    line = re.sub("[0-9]{1,}", "0", line)  # replace 1-9 with 0
    line = "##"+line.lower()[:-1]+"#"  # add character'##' to specify start and # to specify stop
    return line


'''
Normalize given model such that the sum of probability is 1. Add all missing items to dictionaries even if not possible.
@:param model: the distribution of model
@:return: the normalized distribution
'''
def normalize_model(model, adj_map):
    for bi in adj_map:
        total = 0
        for t in adj_map[bi]:
            total += model[bi+t]
        for t in adj_map[bi]:
            model[bi+t] /= total
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
    adj_map = defaultdict(set)
    with open(infile) as f:
        for line in f:
            trigram, prob = line.split('\t')
            prob = float(prob.split('\n')[0])
            model[trigram] = prob
            adj_map[trigram[:-1]].add(trigram[2])
    return model, adj_map


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
def add_alpha_estimate( tri_cnts, bi_cnts, alpha):
    model = defaultdict(float)
    for k in tri_counts:
        pre = bi_cnts[k[:-1]]
        tri = tri_cnts[k]
        model[k] = (tri + alpha) / (pre + alpha * VOCABULARY_SIZE)
    return model

'''
Try different alpha and return the language model with least perplexity
@:param tri_cnts: a dictionary containing all tri-gram counts
@:param bi_cnts: a dictionary containing all bi-gram counts
@:param validation_set: list of lines for validation
@:return best_alpha, best_perplexity, best_model
'''
def adding_alpha_training_LM(adj_map, tri_counts, bi_counts, validation_set):

    best_alpha = 0
    best_perplexity = np.inf
    best_model = -1
    for a in np.arange(0.01, 1, 0.01):
        training_model = add_alpha_estimate(tri_counts, bi_counts, alpha=a)
        cur = get_perplexity(training_model, validation_set, flag=1)
        print("alpha:",round(a,2),"perplexity:",cur)
        if cur < best_perplexity:
            best_alpha = a
            best_perplexity = cur
            best_model = training_model

    print("======================best===========================")
    print("alpha:", round(best_alpha, 2), "perplexity:", best_perplexity)
    return best_alpha, best_perplexity, normalize_model(best_model, adj_map)


"""
Task 3: Estimate trigram probabilities using interpolation.
@:param normalized_tri: dictionary containing the normalized distribution of trigram counts
@:param normalized_bi: dictionary containing the normalized distribution of bigram counts
@:param normalized_uni: dictionary containing the normalized distribution of unigram counts
@:param lam1: interpolation parameter used with trigram probabilities
@:param lam2: interpolation parameter used with bigram probabilities
@:param lam3: interpolation parameter used with unigram probabilities
@:return model: dictionary containing all theoretically possible trigrams and their estimated probabilities
"""

def interpolation_estimate(tri_counts, bi_counts, uni_counts, lam1, lam2, lam3):

    model = defaultdict(float)
    total = sum(uni_counts.values())
    for i in tri_counts.keys():
        last_two = i[1:]
        first_two = i[:-1]
        uni_key = i[1]
        if bi_counts[first_two] > 0:
            p_tri = tri_counts[i]/bi_counts[first_two]
        else:
            p_tri = 0
        if uni_counts[uni_key] > 0:
            p_bi = bi_counts[last_two]/uni_counts[uni_key]
        else:
            p_bi = 0
        p_uni = uni_counts[uni_key] / total
        val = lam1*p_tri+lam2*p_bi+lam3*p_uni
        model[i] = val
    return model

"""
Find the best values for interpolation parameters lambda1, lambda2, lambda3. 
@:param normalized_tri: dictionary containing the normalized distribution of trigram counts
@:param normalized_bi: dictionary containing the normalized distribution of bigram counts
@:param normalized_uni: dictionary containing the normalized distribution of unigram counts
@:param validation_list: held-out set from training data
@:return best_lam1, best_lam2, best_lam3, best_perplexity, best_model: the best estimates for lambda1, lambda2 and lambda 3, the associated model and its perplexity
"""


def interpolation_training_LM(adj_map, tri_counts, bi_counts, uni_counts, validation_list):

    l1 = np.arange(0, 1, 0.1)
    l2 = np.arange(0, 1, 0.1)
    l3 = np.arange(0, 1, 0.1)
    best_perplexity = np.inf
    for lam1 in l1:
        for lam2 in l2:
            for lam3 in l3:
                if lam1+lam2+lam3 == 1:
                    training_model = interpolation_estimate(tri_counts, bi_counts, uni_counts, lam1, lam2, lam3)
                    cur = get_perplexity(training_model, validation_list, flag=1)
                    # print("lam1:", round(lam1, 2), "lam2:", round(lam2, 2), "lam3:", round(lam3, 2),
                    #       "perplexity:", best_perplexity)
                    if cur < best_perplexity:
                        best_lam1 = lam1
                        best_lam2 = lam2
                        best_lam3 = lam3
                        best_perplexity = cur
                        best_model = training_model

    print("======================best===========================")
    print("lam1:", round(best_lam1, 2), "lam2:", round(best_lam2, 2), "lam3:", round(best_lam3, 2), "perplexity:",
          best_perplexity)
    return best_lam1, best_lam2, best_lam3, best_perplexity, normalize_model(best_model, adjcent_map)


'''
Task 4: Generate a random sequence of length k character by character.
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
select next char based on previous two,
the higher prob of next char, the higher prob to be select as next,
restart generator while meeting stop mark #.
'''
def generate_from_LM_random(model,map, k):
    if k < 3:
        raise Exception("Please specify a sequence of at least three characters.") # Not needed?
    else:
        seq = ""
        sentence = "##"
        cnt = 0
        while cnt < k:
            prev2 = sentence[-2:]
            thirds = list(map[prev2])
            probs = [model[prev2+ch] for ch in thirds]
            total = sum(probs)
            alist = [a / total for a in probs]
            probs = np.array(alist)
            bins = np.cumsum(probs)
            # if bins is zero, it means no following char and might be a finished sentence.
            # We restart the generator again
            if len(bins) == 0 or prev2[-1] == ".":
                seq += sentence
                stop = sentence[-1]
                sentence = "##"
                # since # is only an auxiliary char, we roll back our cnt
                if stop == "#":
                    cnt -= 1
                continue
            idx = np.digitize(np.random.random_sample(1), bins)
            next_char = thirds[idx[0]]
            sentence += next_char
            cnt += 1
        seq += sentence
        return seq

'''
select next char based on previous two,
the higher prob of next char, the higher prob to be select as next,
apply the idea of page rank.
'''
def generate_from_LM_pr(model,map, k, alpha):
    if k < 3:
        raise Exception("Please specify a sequence of at least three characters.") # Not needed?
    else:
        seq = ""
        sentence = "##"
        cnt = 0
        while cnt < k:
            prev2 = sentence[-2:]
            thirds = list(map[prev2])
            probs = [model[prev2+ch] for ch in thirds]
            total = sum(probs)
            alist = [a / total for a in probs]
            probs = np.array(alist)
            bins = np.cumsum(probs)
            if random.random() < alpha and len(bins) > 0:
                idx = np.digitize(np.random.random_sample(1), bins)[0]
                next_char = thirds[idx]
            else:
                idx = random.randint(0, len(model)-1)
                next_char = list(model.keys())[idx]
            sentence += next_char
            cnt += 1
        seq += sentence
        return seq

'''
select best next char based on previous two, randomly break tie
restart generator while meeting stop mark #.
'''
def generate_from_LM_greedy(model,map, k):
    if k < 3:
        raise Exception("Please specify a sequence of at least three characters.") # Not needed?
    else:
        seq = ""
        sentence = "##"
        cnt = 0
        while cnt < k:
            prev2 = sentence[-2:]
            thirds = list(map[prev2])
            probs = [model[prev2+ch] for ch in thirds]
            prob_char = dict(zip(probs, thirds))
            probs = sorted(probs,reverse=True)
            idxs = []
            for i, p in enumerate(probs):
                if p == probs[0]:
                    idxs.append(i)
                else:
                    break
            idx = random.sample(idxs, 1)[0]
            next_char = prob_char[probs[idx]]

            sentence += next_char
            cnt += 1
        seq += sentence
        return seq


'''
apply both random and greedy method to generate sequence
category for trigram: connector trigram and inner trigram
e.g., in 'this is a book', 'thi' is inner and 's i' is connector.
We randomly select connector based on LM distribution and generate inner(word) by greedy algorithm
'''


def generate_from_LM_rand_greedy(model, map, k):
    if k < 3:
        raise Exception("Please specify a sequence of at least three characters.")  # Not needed?
    else:
        seq = ""
        sentence = "##"
        cnt = 0
        while cnt < k:
            prev2 = sentence[-2:]
            thirds = list(map[prev2])
            probs = [model[prev2 + ch] for ch in thirds]
            if len(thirds) == 0:
                seq += sentence
                sentence = "##"
                continue
            elif prev2 == "##" or prev2[-1] == " " or prev2[-1] == ".":
                probs = np.array(probs)
                bins = np.cumsum(probs)
                idx = np.digitize(np.random.random_sample(1), bins)[0]
                next_char = thirds[idx]
                if next_char == ".":
                    sentence += next_char
                    seq += sentence
                    sentence = "##"
                    continue
            else:
                prob_char = defaultdict(set)
                for i in range(len(probs)):
                    prob_char[probs[i]].add(thirds[i])
                alist = sorted(prob_char.keys(), reverse=True)
                idxs = prob_char[alist[0]]
                next_char = random.sample(idxs, 1)[0]
            sentence += next_char
            cnt += 1
        seq += sentence
        return seq

'''
Make the generated sequence easier to read by removing non-character #
@:param generated: Randomly generated string
@:return: readable sequence
'''
def readable_generated_seq (generated):
    sequence_sub = re.compile("#")
    sequence_print = sequence_sub.sub("", generated)
    return sequence_print


'''
Task 5: Given a language model and a test paragraph, calculate the perplexity.
@:param model: a language model stored in dictionary
@:param testfile: name of the testfile
@:param flag: flag is 0 means test data is from file, 1 means it is from dictionary
@:return: perplexity of testfile under model
'''
def get_perplexity(model, testfile, flag):

    if flag == 0:
        fo = open(testfile, "r")
        lines = fo.readlines()
        fo.close()
    else:
        lines = testfile
    # calculate the perplexity of each line and sum them up
    pp = 0
    for line in lines:
        if flag == 0:
            line = preprocess_line(line)
        logp = get_sequence_log_prob(model, line)
        # get cross entropy
        cross_entropy = -logp / len(line)
        # get perplexity of this line
        pp += np.exp2(cross_entropy)

    return pp

'''
Given a language model and a sequence. calculate the log probablity.
@:param model: a language model stored in dictionary
@:param line: the sentence
@:return: the log probability of this sentence
'''
def get_sequence_log_prob(model, line):
    p = 0
    for j in range(len(line) - (2)):
        trigram = line[j:j + 3]
        prob = model[trigram]
        if prob > 0: # Is this the way to go?
            p += np.log2(prob)
    return p




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
    random.seed(1) # fix random seed
    # infile = "training.en"
    train_list, tri_counts, bi_counts, uni_counts, validation_list, test_list\
        = read_and_store(infile, [0.8, 0.1, 0.1])
    adjcent_map, full_tri_counts, full_bi_counts, full_uni_counts\
        = missing_items(tri_counts, bi_counts, uni_counts)
    # best_alpha, best_perplexity, best_model = adding_alpha_training_LM(adjcent_map, tri_counts, bi_counts, validation_list)
    best_lam1, best_lam2, best_lam3, best_perplexity, best_model\
        = interpolation_training_LM(adjcent_map, full_tri_counts, full_bi_counts, full_uni_counts, validation_list)
    write_back_prob("outfile.txt", best_model)
    model, model_map = read_model("model-br.en")
    print("Our model in test set:", get_perplexity(best_model, test_list, flag=1))
    print("Given model in test set:", get_perplexity(model, test_list, flag=1))
    print("=======================================")
    print("Our model in test file:", get_perplexity(best_model, "test", flag=0))
    print("Given model in test file:", get_perplexity(model, "test", flag=0))
    # #seq = generate_from_LM(best_model, 300)
    print("=======================================")
    # best_model, adjcent_map = read_model("outfile.txt")
    for i in range(5):
        # seq = generate_from_LM_random(best_model, adjcent_map, 100)
        # seq = readable_generated_seq(seq)
        # print("generator v1:", seq)
        # seq = generate_from_LM_greedy(best_model, adjcent_map, 100)
        # seq = readable_generated_seq(seq)
        # print("generator v2:", seq)
        # seq = generate_from_LM_random(best_model, adjcent_map, 150)
        # seq = readable_generated_seq(seq)
        # print("our model in generator v1:", seq)
        # seq = generate_from_LM_random(model, model_map, 150)
        # seq = readable_generated_seq(seq)
        # print("given model in generator v1:", seq)
        seq = generate_from_LM_rand_greedy(best_model, adjcent_map, 150)
        # seq = readable_generated_seq(seq)
        print("our model in generator v3:", seq)
        seq = generate_from_LM_rand_greedy(model, model_map, 150)
        # seq = readable_generated_seq(seq)
        print("given model in generator v3:", seq)
        print("=========================")






