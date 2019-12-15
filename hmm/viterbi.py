from learnhmm import *
import sys
import numpy as np
import random

#### test command
#### directory at F19_10601_HW7
# python code/viterbi.py handout/fulldata/testwords.txt handout/fulldata/index_to_word.txt handout/fulldata/index_to_tag.txt result/fulldata/hmmprior.txt result/fulldata/hmmemit.txt result/fulldata/hmmtrans.txt result/fulldata/predictedtest.txt result/fulldata/metrics.txt

# python3 code/viterbi.py handout/toydata/toytest.txt handout/toydata/toy_index_to_word.txt handout/toydata/toy_index_to_tag.txt result/toydata/hmmprior.txt result/toydata/hmmemit.txt result/toydata/hmmtrans.txt result/toydata/predictedtest.txt result/toydata/metrics.txt


# load data
# test_input = sys.argv[1]
# index_to_word = sys.argv[2]
# index_to_tag = sys.argv[3]
# hmmprior = sys.argv[4]
# hmmemit = sys.argv[5]
# hmmtrans = sys.argv[6]
# predicted_file = sys.argv[7]
# metric_file = sys.argv[8]

# test_input = "../handout/toydata/toytest.txt"
# index_to_word = "../handout/toydata/toy_index_to_word.txt"
# index_to_tag = "../handout/toydata/toy_index_to_tag.txt"
# hmmprior = "../handout/toydataoutputs/hmmprior.txt"
# hmmemit = "../handout/toydataoutputs/hmmemit.txt"
# hmmtrans = "../handout/toydataoutputs/hmmtrans.txt"
# predicted_file = "../result/toydata/predictedtest.txt"
# metric_file = "../result/toydata/metrics.txt"

# test_input = "../handout/fulldata/testwords.txt"
# index_to_word = "../handout/fulldata/index_to_word.txt"
# index_to_tag = "../handout/fulldata/index_to_tag.txt"
# hmmprior = "../result/fulldata/hmmprior.txt"
# hmmemit = "../result/fulldata/hmmemit.txt"
# hmmtrans = "../result/fulldata/hmmtrans.txt"
# predicted_file = "../result/fulldata/predictedtest.txt"
# metric_file = "../result/fulldata/metrics.txt"

# for plots under different settings
sequence = 10
test_input = "../handout/fulldata/trainwords.txt"
index_to_word = "../handout/fulldata/index_to_word.txt"
index_to_tag = "../handout/fulldata/index_to_tag.txt"
hmmprior = "../result/fulldata" + str(sequence) + "/hmmprior.txt"
hmmemit = "../result/fulldata" + str(sequence) + "/hmmemit.txt"
hmmtrans = "../result/fulldata" + str(sequence) + "/hmmtrans.txt"
predicted_file = "../result/fulldata" + str(sequence) + "/predictedtest.txt"
metric_file = "../result/fulldata" + str(sequence) + "/metrics.txt"


# read train/test data
test_input = readFile(test_input)
index_to_word = readFile(index_to_word)
index_to_tag = readFile(index_to_tag)
hmmprior = readFile(hmmprior)
hmmemit = readFile(hmmemit)
hmmtrans = readFile(hmmtrans)

index_to_word = index_to_word.split("\n")
index_to_tag = index_to_tag.split("\n")
index_to_word = strip_lst(index_to_word)
index_to_tag = strip_lst(index_to_tag)

hmmprior = hmmprior.split("\n")
hmmemit = hmmemit.split("\n")
hmmtrans = hmmtrans.split("\n")

word_dict = dict(zip(index_to_word,range(len(index_to_word))))
tag_dict = dict(zip(index_to_tag, range(len(index_to_tag))))

tag_dict_reverse = dict(zip(range(len(index_to_tag)), index_to_tag))

# get pi & A & B matrices from input files
pi = np.zeros(len(index_to_tag))
A = np.zeros((len(index_to_tag), len(index_to_tag)))
B = np.zeros((len(index_to_tag), len(index_to_word)))
for i in range(len(index_to_tag)):
    pi[i] = hmmprior[i]
    A[i,] = hmmtrans[i].split()
    B[i,] = hmmemit[i].split()


def predict_viterbi(tmp_word_lst, pi, A, B, index_to_tag):
    T = len(tmp_word_lst)
    lw = np.zeros((T, len(index_to_tag)))  # shape=number of states
    p = np.zeros((T, len(index_to_tag)))
    # initial lw
    for j in range(len(index_to_tag)):
        lw[0, j] = np.log(pi[j]) + np.log(B[j, tmp_word_lst[0]])
        p[0, j] = j
    for t in range(1, T):
        for j in range(len(index_to_tag)):
            tmp_prob_lst = []
            for k in range(len(index_to_tag)):
                tmp_prob = np.log(B[j, tmp_word_lst[t]]) + np.log(A[k, j]) + lw[t-1, k]
                tmp_prob_lst.append(tmp_prob)
            lw[t, j] = max(tmp_prob_lst)
            p[t, j] = np.argmax(tmp_prob_lst)

    pred_tag_lst = np.zeros(T)
    pred_tag_lst[T-1] = np.argmax(lw[T-1, ])
    for t in range(T-2, -1, -1):
        pred_tag_lst[t] = p[t+1, int(pred_tag_lst[t+1])]
    return pred_tag_lst


# test data
tmp_data = test_input.split("\n")
accuracy = 0
num_words = 0
predict_str = ""
for i in range(len(tmp_data)):  # every line of sentence (an example)
    # print(i)
    tmp_token = tmp_data[i].split()
    tmp_token = strip_lst(tmp_token)

    tmp_tag_lst = []
    tmp_word_lst = []
    for j in range(len(tmp_token)):  # words in same line
        word = tmp_token[j].split("_")[0]
        tag = tmp_token[j].split("_")[1]
        tmp_tag_lst.append(tag)
        tmp_word_lst.append(word)
    
    # make into numbers
    tmp_tag_lst_num = convert_to_num(tmp_tag_lst, tag_dict)
    tmp_word_lst_num = convert_to_num(tmp_word_lst, word_dict)

    # predict on this example
    pred_tag_lst = predict_viterbi(tmp_word_lst_num, pi, A, B, index_to_tag)
    for j in range(len(pred_tag_lst)):
        num_words += 1
        predict_str += str(tmp_word_lst[j])
        predict_str += "_"
        predict_str += str(tag_dict_reverse[pred_tag_lst[j]])
        predict_str += " "
        if pred_tag_lst[j] == tmp_tag_lst_num[j]:
            accuracy += 1
    predict_str = predict_str[:-1]
    predict_str += "\n"

accuracy /= num_words

metric_str = "Accuracy: " + str(accuracy) + "\n"
print(metric_str)
writeFile(metric_file, metric_str)

writeFile(predicted_file, predict_str)
