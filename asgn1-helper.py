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

'''
dictionary to store counts of all trigrams in input
key (string): trigram e.g "and"
value (int): counts of the key
'''
tri_counts = defaultdict(int)

'''
dictionary to store counts of previous two chars of all trigrams in input
key (string): previous two chars e.g "an"
value (int): counts of the key
'''
bi_counts = defaultdict(int)

'''
set to store all third char in trigrams, e.g. 'd' in "and"
'''
vocabulary = set()

'''
dictionary to store the probability of each trigram
'''
train_model = defaultdict(float)

'''
============================================================
function declaration part
============================================================
'''


'''
Read in training file and store trigram count into dictionary 'tri_counts' 
@:param infile: input file name
@:return: void
'''
def read_and_store(infile):
    with open(infile) as f:
        for line in f:
            line = preprocess_line(line)  # implemented already.
            # TODO: include last 2 characters or not?
            for j in range(len(line) - (3)):
                trigram = line[j:j + 3]
                pre = line[j:j + 2]
                vocabulary.add(trigram[2])
                tri_counts[trigram] += 1
                bi_counts[pre] += 1

'''
Task 1: removing unnecessary characters from each line
@:param line: a line of string from input file
@:return: a new line only with English alphabet, space, digits and dot character.
'''
def preprocess_line(line):
    rule = re.compile("[^\s.A-Za-z0-9]")
    line = rule.sub('', line)  # newline with only digit, alphabet, space and dot.
    line = re.sub("[1-9]", "0", line)  # replace 1-9 to 0
    return line.lower()

'''
Task 3: Estimate trigram probabilities by trigram count in training set. 
Store probabilities into global dictionary 'train_model'
The calculation formula is lec6, slide 13:

P(w3 | w1, w2) = ( Count(w1, w2, w3) + alpha ) / (Count(w1, w2) + alpha * v)  
where alpha is a tunable smoothing parameter, and v is the size of vocabulary

@:param alpha: smoothing parameter alpha
@:return: void
'''
def estimate_tri_prob(alpha = 0):
    v = len(vocabulary)
    for k in tri_counts:
        # TODO: the detail estimation need discussing
        pre = bi_counts[k[:2]]
        tri = tri_counts[k]
        train_model[k] = (tri + alpha) / (pre + alpha * v)

'''
Task 3: Write back estimated probabilities to an output file
@:param outfile: filename
@:return: write succeed or not
'''
def write_back_prob(outfile):
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
        group = int(k / 3)
        sequence = ""
        # this is a totally random version for sentence generation
        # TODO: need to discuss how to generate sequence
        trigrams = list(lmodel.keys())
        lens = len(trigrams)
        for i in range(group):
            # random select trigrams and concat them together
            idx = random.randint(0, lens-1)
            key = trigrams[idx]
            sequence += key
        return sequence


'''
Task 5: Given a language model and a test paragraph, calculate the perplexity.
@:param model: a language model stored in dictionary
@:param testfile: the name of test file
@:return: the perplexity
'''
def get_perplexity(model, testfile):
    logsum = 0
    cnt = 0
    with open(testfile) as f:
        # calculate the log probability of each sentence and sum them up
        for line in f:
            line = preprocess_line(line)
            cnt += 1
            p = get_sentence_prob(model, line)
            logsum += np.log2(p)
        # get cross entropy
        cross_entropy = -logsum / cnt
        # get perplexity of whole test paragraph
        pp = np.exp2(cross_entropy)
        return pp

'''
Given a language model and a sentence, calculate the probablity.
@:param model: a language model stored in dictionary
@:param line: the sentence
@:return: the probability of this sentence
'''
def get_sentence_prob(model, line):
    p = 1
    for j in range(len(line) - (3)):
        trigram = line[j:j + 3]
        prob = model[trigram]
        # TODO: what if current trigram is not in training model? We skip unknown word here
        if prob != 0:
            p *= prob
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

    read_and_store(infile)
    estimate_tri_prob()
    write_back_prob("outfile.txt")
    model = read_model("model-br.en")
    seq = generate_from_LM(train_model, 300)
    print(get_perplexity(train_model,"test"))
    # print(train_model)
    # show(infile)





