import sys
import numpy as np
import random

#### test command
#### directory at F19_10601_HW7
# python code_test/learnhmm.py handout/fulldata/trainwords.txt handout/fulldata/index_to_word.txt handout/fulldata/index_to_tag.txt result/fulldata/hmmprior.txt result/fulldata/hmmemit.txt result/fulldata/hmmtrans.txt

# python code/learnhmm.py handout/toydata/toytrain.txt handout/toydata/toy_index_to_word.txt handout/toydata/toy_index_to_tag.txt result/toydata/hmmprior.txt result/toydata/hmmemit.txt result/toydata/hmmtrans.txt

# tar -cvf hmm.tar learnhmm.py viterbi.py python3.txt

# load data
# train_input = sys.argv[1]
# index_to_word = sys.argv[2]
# index_to_tag = sys.argv[3]
# hmmprior = sys.argv[4]
# hmmemit = sys.argv[5]
# hmmtrans = sys.argv[6]

# train_input = "../handout/toydata/toytrain.txt"
# index_to_word = "../handout/toydata/toy_index_to_word.txt"
# index_to_tag = "../handout/toydata/toy_index_to_tag.txt"
# hmmprior = "../result/toydata/hmmprior.txt"
# hmmemit = "../result/toydata/hmmemit.txt"
# hmmtrans = "../result/toydata/hmmtrans.txt"

# train_input = "../handout/fulldata/trainwords.txt"
# index_to_word = "../handout/fulldata/index_to_word.txt"
# index_to_tag = "../handout/fulldata/index_to_tag.txt"
# hmmprior = "../result/fulldata/hmmprior.txt"
# hmmemit = "../result/fulldata/hmmemit.txt"
# hmmtrans = "../result/fulldata/hmmtrans.txt"

# for plots under different settings
sequence = 10000
train_input = "../handout/fulldata/trainwords.txt"
index_to_word = "../handout/fulldata/index_to_word.txt"
index_to_tag = "../handout/fulldata/index_to_tag.txt"
hmmprior = "../result/fulldata" + str(sequence) + "/hmmprior.txt"
hmmemit = "../result/fulldata" + str(sequence) + "/hmmemit.txt"
hmmtrans = "../result/fulldata" + str(sequence) + "/hmmtrans.txt"

# read files
def readFile(path):
    with open(path, "rt") as f:
        return f.read()


# write files
def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)


def convert_to_num(item_lst, dict):
    num_lst = []
    for i in item_lst:
        num_lst.append(dict[i])
    return num_lst


def strip_lst(lst):
    res = []
    for i in range(len(lst)):
        tmp = lst[i].strip()
        if len(tmp)>0:
            res.append(tmp)
    return res

# read train/test data
train_input = readFile(train_input)
index_to_word = readFile(index_to_word)
index_to_tag = readFile(index_to_tag)

index_to_word = index_to_word.split("\n")
index_to_tag = index_to_tag.split("\n")
index_to_word = strip_lst(index_to_word)
index_to_tag = strip_lst(index_to_tag)

word_dict = dict(zip(index_to_word,range(len(index_to_word))))
tag_dict = dict(zip(index_to_tag, range(len(index_to_tag))))

pi = np.zeros(len(index_to_tag))
A = np.zeros((len(index_to_tag), len(index_to_tag)))
B = np.zeros((len(index_to_tag), len(index_to_word)))

tmp_data = train_input.split("\n")
tmp_data = tmp_data[:sequence]
first_state_tag = []
for i in range(len(tmp_data)):  # every line of sentence
    tmp_token = tmp_data[i].split()
    tmp_token = strip_lst(tmp_token)
    
    tmp_lst_tag = []
    tmp_lst_word = []
    for j in range(len(tmp_token)):  # words in same line
        word = tmp_token[j].split("_")[0]
        tag = tmp_token[j].split("_")[1]
        tmp_lst_tag.append(tag)
        tmp_lst_word.append(word)
        if j==0:
            first_state_tag.append(tag)

    # A & B matrix
    tmp_lst_tag = convert_to_num(tmp_lst_tag, tag_dict)
    tmp_lst_word = convert_to_num(tmp_lst_word, word_dict)

    for j in range(len(tmp_lst_tag)-1):  # final state doesn't have a next state
        A[tmp_lst_tag[j], tmp_lst_tag[j+1]] += 1
    for j in range(len(tmp_lst_tag)):
        B[tmp_lst_tag[j], tmp_lst_word[j]] += 1


# get pi matrix
first_state_tag = convert_to_num(first_state_tag, tag_dict)
for i in range(len(first_state_tag)):
    pi[first_state_tag[i]] += 1
pi += 1
pi = pi / pi.sum(axis=0)

# get A matrix
A += 1
A = A / A.sum(axis=1, keepdims=True)

# get B matrix
B += 1
B = B / B.sum(axis=1, keepdims=True)


# output results
pi_str = ""
for i in range(len(pi)):
    pi_str += str(pi[i]) + "\n"

A_str = ""
for i in range(len(A)):
    for j in range(len(A[i])):
        A_str += str(A[i][j]) + " "
    A_str = A_str[:-1]
    A_str += "\n"

B_str = ""
for i in range(len(B)):
    for j in range(len(B[i])):
        B_str += str(B[i][j]) + " "
    B_str = B_str[:-1]
    B_str += "\n"

writeFile(hmmprior, pi_str)
writeFile(hmmtrans, A_str)
writeFile(hmmemit, B_str)