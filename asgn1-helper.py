# Here are some libraries you're likely to use. You might want/need others as well.
import re
import sys
from random import random
from math import log
from collections import defaultdict

'''
============================================================
variable declaration part
============================================================
'''
# dictionary to store counts of all trigrams in input
# key (string): trigram e.g "and"
# value (int): counts of the key
tri_counts = defaultdict(int)
tri_probability = defaultdict(float)

'''
============================================================
function declaration part
============================================================
'''


'''
This bit of code gives an example of how you might extract trigram counts
from a file, line by line. If you plan to use or modify this code,
please ensure you understand what it is actually doing, especially at the
beginning and end of each line. Depending on how you write the rest of
your program, you may need to modify this code.
@:param param1: input file name
@:return: void
'''
def read_and_store(infile):
    with open(infile) as f:
        for line in f:
            line = preprocess_line(line)  # implemented already.
            for j in range(len(line) - (3)):
                trigram = line[j:j + 3]
                tri_counts[trigram] += 1

'''
Task 1: removing unnecessary characters from each line
@:param param1: a line of string from input file
@:return: a new line only with English alphabet, space, digits and dot character.
'''
def preprocess_line(line):
    rule = re.compile("[^\s.A-Za-z0-9]")
    line = rule.sub('', line)  # newline with only digit, alphabet, space and dot.
    line = re.sub("[1-9]", "0", line)  # replace 1-9 to 0
    return line.lower()

'''
Estimate trigram probabilities by trigram count in training set. 
Store probabilities into global dictionary 'global probability'
@:param: None
@:return: void
'''
def estimate_tri_prob():
    total = sum(tri_counts.values())
    for k in tri_counts:
        tri_probability[k] = tri_counts[k] /float(total)


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
    # print(tri_probability)
    # show(infile)





