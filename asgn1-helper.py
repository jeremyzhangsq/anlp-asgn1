# Here are some libraries you're likely to use. You might want/need others as well.
import re
import sys
import random
from math import log
import numpy as np
from collections import defaultdict
import itertools
import string
import matplotlib.pyplot as plt
'''
============================================================
variable declaration part
============================================================
'''
VOCABULARY_SIZE = 30
SONORITY = {'++': ['a','e','i','o','u'],
            '+': ['l','r'],
            '-': ['f','v','z','s','h','y'],
            '--': ['b','c','d','g','k','m','n','p','q','t','x','w','j']
            }

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
def read_and_store(infile, ratios, sonority):

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

    sonority_count = defaultdict(int)
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
                # include '##<start char>' and '<end char>##' -- included in preprocessing?
                prev_letter = '#'
                for j in range(len(line) - (2)):
                    trigram = line[j:j + 3]
                    pre = line[j:j + 2]
                    uni = line[j:j + 1]
                    # adjacent_map[pre].add(trigram[2])
                    tri_counts[trigram] += 1
                    bi_counts[pre] += 1
                    uni_counts[uni] += 1
                    # add sonority cnt here
                    if sonority:
                        sec = 'unk'
                        fst = 'unk'
                        # we record the bigram count for sonority
                        for son in SONORITY:
                            if line[j] in SONORITY[son]:
                                sec = son
                            if prev_letter in SONORITY[son]:
                                fst = son
                        # sonority[(fst,sec)] means the count of sonority sec given sonority fst
                        key = (fst, sec)
                        sonority_count[key] += 1
                    prev_letter = line[j]

            # idx = 1 means the random variable is a validation item
            elif idx == 1:
                validation_list.append(line)
            # idx = 2 means the random variable is a testing item
            else:
                test_list.append(line)

        # normalize the distribution of sonority
        if sonority:
            total = sum(sonority_count.values())
            for son in sonority_count:
                sonority_count[son] /= total

    return sonority_count, train_list, tri_counts, bi_counts, uni_counts, validation_list, test_list

"""
Inserts missing trigrams and assigns all impossible combinations to <UNK>
@:param counts: counts dictionary
@:return counts: counts dictionary with possible but unseen trigrams added with count 0
"""

def missing_items (tri_counts, bi_counts, uni_counts):

    all_letters = list(string.ascii_lowercase)
    all_nonletters = [" ", "#", "."]
    all_letters.extend(all_nonletters)
    # list all bigram combinations
    all_combs_bi = itertools.product(all_letters, repeat=2)
    # list all trigram combinations
    all_combs_tri = itertools.product(all_letters, repeat=3)

    all_combs_bi_list = []
    all_combs_tri_list = []

    for combo in all_combs_bi:
        combo = "".join(list(combo))
        all_combs_bi_list.append(combo)

    for combo in all_combs_tri:
        combo = "".join(list(combo))
        all_combs_tri_list.append(combo)

    # given default value 0 for not occur trigrams
    for i in all_combs_bi_list:
        if i not in bi_counts.keys():
            bi_counts[i] = 0
    for j in all_letters:
        if j not in uni_counts.keys():
            uni_counts[j] = 0
    for k in all_combs_tri_list:
        if k not in tri_counts:
            tri_counts[k] = 0

    adj_map = defaultdict(set)
    for i in tri_counts:
        adj_map[i[:-1]].add(i[2])
    return adj_map, tri_counts, bi_counts, uni_counts

'''
Task 1: Treating each line as separate sequence, remove all letters that are not in the English alphabet, 
all punctuation except '.', convert all digits to '0' and lowercase the whole line. 
Extra steps taken not specified in task: 
Remove excess white spaces and abbreviations.
@:param line: A line of string from the input file.
@:return: Modified line containing only lowercase characters found in the English alphabet, 
white spaces, 0, dot character and line start and stop markers. 
'''
def preprocess_line(line):
    rule = re.compile("[^\s.A-Za-z0-9]")
    # new line with only digit, English alphabet, white space and dot
    line = rule.sub('', line)
    # replace excess number of whitespaces with one white space
    line = re.sub('\s{2,}', ' ', line)
    # Remove abbreviations enclosed by whitespace (substitute with whitespace)
    line = re.sub('\s[A-Z]{2,}\s', ' ', line)
    # Remove abbreviations that precede all punctuation (substitute with dot)
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
        # get sum of p(*|bigram)
        for t in adj_map[bi]:
            total += model[bi+t]
        # normalize
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


"""
Task 3: Estimate trigram probabilities using interpolation.
@:param tri_counts: dictionary containing the distribution of trigram counts
@:param bi_counts: dictionary containing the distribution of bigram counts
@:param uni_counts: dictionary containing the distribution of unigram counts
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
        p_tri = 0
        p_bi = 0
        # p(w3 | w2)
        if bi_counts[first_two] > 0:
            p_tri = tri_counts[i]/bi_counts[first_two]
        # p(w3 | w1, w2)
        if uni_counts[uni_key] > 0:
            p_bi = bi_counts[last_two]/uni_counts[uni_key]
        # p(w3)
        p_uni = uni_counts[uni_key] / total
        val = lam1*p_tri+lam2*p_bi+lam3*p_uni
        model[i] = val
    return model

"""
Find the best values for interpolation parameters lambda1, lambda2, lambda3. 
@:param tri_counts: dictionary containing the distribution of trigram counts
@:param bi_counts: dictionary containing the distribution of bigram counts
@:param uni_counts: dictionary containing the distribution of unigram counts
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
                    # record current best
                    if cur < best_perplexity:
                        best_lam1 = lam1
                        best_lam2 = lam2
                        best_lam3 = lam3
                        best_perplexity = cur
                        best_model = training_model

    print("======================best===========================")
    print("lam1:", round(best_lam1, 2), "lam2:", round(best_lam2, 2), "lam3:", round(best_lam3, 2), "perplexity:",
          best_perplexity)
    # before return LM, normalize all trigram based on previous two
    return best_lam1, best_lam2, best_lam3, best_perplexity, normalize_model(best_model, adj_map)


'''
Task 4 (verision 1): Generate a random sequence of length k character by character.
select next char based on previous two,
the higher prob of next char, the higher prob to be select as next,
restart generator while meeting stop mark #.
@:param model: a language model stored in dictionary
@:param map: an adjacent map. key is bigram and value is every possible next char
@:param k: the size of output sequence
@:return: a random string sequence

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
            # normalize the probability distribution to 0-1
            alist = [a / total for a in probs]
            probs = np.array(alist)
            # make an array with the cumulative sum of probabilities at each
            # index (ie prob. mass func)
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

            # digitize tells us which bin they fall into.
            idx = np.digitize(np.random.random_sample(1), bins)
            next_char = thirds[idx[0]]
            sentence += next_char
            cnt += 1
        seq += sentence
        return seq


'''
Task 4 (report version): Generate a random sequence of length k character by character.
apply both random and greedy method to generate sequence
category for trigram: connector trigram and inner trigram
e.g., in 'this is a book', 'thi' is inner and 's i' is connector.
We randomly select connector based on LM distribution and generate inner(word) by greedy algorithm
Also, it support the sonority optimization for task 6
@:param model: a language model stored in dictionary
@:param map: an adjacent map. key is bigram and value is also possible next char
@:param map: a sonority map. key is bigram of sonority classes and value is the probability of next sonority class given previous class
@:param k: the size of output sequence
@:param sonority: the sonority optimization flag
@:return: a random string sequence

'''

def generate_from_LM_rand_greedy(model, map, son_map, k, sonority):
    if k < 3:
        raise Exception("Please specify a sequence of at least three characters.")
    else:
        sentence = "##"
        neat_seq = ""
        while len(neat_seq) < k:
            prev2 = sentence[-2:]
            thirds = list(map[prev2])
            probs = [model[prev2 + ch] for ch in thirds]
            if len(thirds) == 0:
                sentence += "##"
                continue
            # if it is a connector trigram, we random select next char according to prob
            elif prev2 == "##" or prev2[-1] == " " or prev2[-1] == ".":
                probs = np.array(probs)
                bins = np.cumsum(probs)
                idx = np.digitize(np.random.random_sample(1), bins)[0]
                next_char = thirds[idx]
                # avoid repetition dot mark and white space
                if next_char == "." or next_char == " ":
                    sentence += next_char
                    sentence += "##"
                    continue
            else:
                prob_char = defaultdict(set)
                if sonority:
                    # optimization for task 6
                    for i in range(len(probs)):
                        fst = 'unk'
                        snd = 'unk'
                        # identify the sonority tag for last char prev[-1] and current char third[i]
                        if str(prev2[-1]).isalpha():
                            for key in SONORITY:
                                if prev2[-1] in SONORITY[key]:
                                    fst = key
                                    break
                        if str(thirds[i]).isalpha():
                            for key in SONORITY:
                                if thirds[i] in SONORITY[key]:
                                    snd = key
                                    break
                        # the final probability equals to language model probability times sonority probability
                        p = probs[i]*son_map[(fst,snd)]
                        prob_char[p].add(thirds[i])

                else:
                    for i in range(len(probs)):
                        prob_char[probs[i]].add(thirds[i])
                # greedy select next only with highesr prob
                alist = sorted(prob_char.keys(), reverse=True)
                idxs = prob_char[alist[0]]
                # if there are more than more chars, randomly select one to break the tie
                next_char = random.sample(idxs, 1)[0]
            sentence += next_char
            neat_seq = readable_generated_seq(sentence)
        return sentence



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
Task 3: Excerpt from our model displaying all ng* combinations and their associated probabilities
@:param best_model: our language model
@:return ng_dict:
'''
def ng_excerpt (best_model):
    ng_dict = defaultdict(float)
    for key in best_model.keys():
        if key[0] == "n":
            if key[1] == "g":
                ng_dict[key] = best_model[key]
        continue
    return ng_dict

'''
Task 3: Print alphabetically sorted excerpt from model, here ng excerpt
@:param ng_dict: dictionary of model probabilities, here excerpt for ng
'''

def print_sorted_ng(ng_dict):
    for key, value in sorted(ng_dict.items()):
        print (key, round(value, 3))

'''
Task 3: Plot log ng* probabilities 
@:param ng_dict: dictionary of model probabilities, here excerpt for ng
'''
def plot_ng (ng_dict, outname):
    x = []
    y = []
    dictlist = []
    for key, value in ng_dict.items():
        dictlist.append([value, key])
        dictlist.sort(reverse=True)
    for i in range(len(dictlist)):
        x.append(dictlist[i][1])
        y.append(dictlist[i][0])
    x_labels = []
    for tri in x:
        if tri[-1] == " ":
            x_labels.append("_")
        else:
            x_labels.append(tri[-1])
    plt.bar(x, y, width=0.5, tick_label=x_labels)
    plt.yscale('log')
    plt.savefig(outname)



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

def run(infile, sonority):

    sonority_counts, train_list, tri_counts, bi_counts, uni_counts, validation_list, test_list \
            = read_and_store(infile, [0.8, 0.1, 0.1], sonority)
    adjcent_map, full_tri_counts, full_bi_counts, full_uni_counts \
        = missing_items(tri_counts, bi_counts, uni_counts)
    # best_alpha, best_perplexity, best_model = adding_alpha_training_LM(adjcent_map, tri_counts, bi_counts, validation_list)
    best_lam1, best_lam2, best_lam3, best_perplexity, best_model \
        = interpolation_training_LM(adjcent_map, full_tri_counts, full_bi_counts, full_uni_counts, validation_list)
    write_back_prob("outfile.txt", best_model)
    model, model_map = read_model("model-br.en")
    print("=======================================")
    print("Our model in test set:", get_perplexity(best_model, test_list, flag=1))
    print("Given model in test set:", get_perplexity(model, test_list, flag=1))
    print("=======================================")
    print("Our model in test file:", get_perplexity(best_model, "test", flag=0))
    print("Given model in test file:", get_perplexity(model, "test", flag=0))
    # best_model, adjcent_map = read_model("outfile.txt")
    for i in range(5):
        print("=========================")
        seq = generate_from_LM_rand_greedy(best_model, adjcent_map, sonority_counts, 300, sonority)
        seq = readable_generated_seq(seq)
        print("our model in generator v3:", seq)
        seq = generate_from_LM_rand_greedy(model, model_map, sonority_counts, 300, sonority)
        seq = readable_generated_seq(seq)
        print("given model in generator v3:", seq)
    return best_model

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
    random.seed(1)  # fix random seed
    bmodel = run(infile=infile, sonority=False)
    print("=============================ng* prob=================================")
    ng_dict = ng_excerpt(bmodel)
    print_sorted_ng(ng_dict)
    plot_ng(ng_dict,"ng_prob.png")
    print("======================Optimization: Sonority===========================")
    run(infile=infile, sonority=True)






