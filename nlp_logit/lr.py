import sys
import numpy as np

# test command
# python lr.py model1_formatted_train.tsv model1_formatted_valid.tsv model1_formatted_test.tsv dict.txt model1_train_out.labels model1_test_out.labels model1_metrics_out.txt 60


LEARNING_RATE = 0.1

# read files
def readFile(path):
    with open(path, "rt") as f:
        return f.read()


# write files
def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)


# load data
train_input = sys.argv[1]
validation_input = sys.argv[2]
test_input = sys.argv[3]
dict_input = sys.argv[4]
train_out = sys.argv[5]
test_out = sys.argv[6]
metrics_out = sys.argv[7]
num_epoch = int(sys.argv[8])


# read dict data
dict_data = readFile(dict_input).split("\n")
dict_data = dict_data[:-1]  # remove the last empty line
mydict = dict()
for i in range(len(dict_data)):
    tmp = dict_data[i].split()
    mydict[tmp[0]] = tmp[1]

# read train/valid/test data
# remove the last empty line
train_data = readFile(train_input).split("\n")[:-1] 
valid_data = readFile(validation_input).split("\n")[:-1]
test_data = readFile(test_input).split("\n")[:-1]


# contruct feature dict from train data
def get_feat_dict(mydata):
    feat_dict = dict()  # feature: index
    feat_dict_inverse = dict()  # index: feature
    for i in range(len(mydata)):
        tmp_line = mydata[i].split("\t")
        tmp_feat = tmp_line[1:]
        for j in range(len(tmp_feat)):
            tmp_feat_idx = tmp_feat[j].split(":")[0]
            if tmp_feat_idx not in feat_dict:
                key = len(feat_dict) + 1
                feat_dict[tmp_feat_idx] = key
                feat_dict_inverse[key] = tmp_feat_idx
    return (feat_dict, feat_dict_inverse)


# parse data into feature matrix and label vector
def get_X_label(mydata, m, feat_dict_inverse):
    n = len(mydata)
    label = []
    X = np.zeros((n, m))  # design matrix
    X[:, 0] = 1
    for i in range(n):
        tmp_line = mydata[i].split("\t")
        label.append(int(tmp_line[0]))
        tmp_feat = tmp_line[1:]
        tmp_feat_set = set()
        for j in range(len(tmp_feat)): # collect all available features in sample i
            tmp_feat_set.add(tmp_feat[j].split(":")[0])
        for j in range(len(feat_dict_inverse)): # make into design matrix
            if j == 0:
                continue
            else:
                if feat_dict_inverse[j] in tmp_feat_set:
                    X[i, j] = 1
    return (X, label)


# train model
def train_logit(n, m, X, label, num_epoch, LEARNING_RATE, X_valid, label_valid):
    param = np.zeros(m)
    train_ll = []
    valid_ll = []
    for k in range(num_epoch):
        print(k)
        for i in range(n):
            activation = sigmoid(param, X[i,:])
            grad = np.multiply(X[i,:], (label[i] - activation))
            param += LEARNING_RATE * grad
        train_ll.append(compute_likelihood(X, label, param))
        valid_ll.append(compute_likelihood(X_valid, label_valid, param))
    return param, train_ll, valid_ll


# compute negative condition log-likelihood
def compute_likelihood(X, label, param):
    J = 0
    n = len(label)
    for i in range(n):
        dot_prod = np.dot(X[i,:], param)
        J += -label[i] * dot_prod + np.log(1 + np.exp(dot_prod))
    return J/n


# sigmoid function
def sigmoid(param, X):
    dot_prod = np.dot(X, param)
    return 1 / (1 + np.exp(-dot_prod))


# prediction
def pred(param, X, label):
    pred_prob = sigmoid(param, X)
    n = len(X)
    pred_label = np.zeros(n)
    error_rate = 0
    for i in range(n):
        if pred_prob[i] >= 0.5:
            pred_label[i] = 1
        else:
            pred_label[i] = 0
        if pred_label[i] != label[i]:
            error_rate += 1
    error_rate /= n
    print(error_rate)
    return (error_rate, pred_label)


# main function
feat_dict, feat_dict_inverse = get_feat_dict(train_data)

# number of features in train data + intercept term
m = len(feat_dict) + 1  # col
n = len(train_data)  # row

# training
train_X, train_label = get_X_label(train_data, m, feat_dict_inverse)
valid_X, valid_label = get_X_label(valid_data, m, feat_dict_inverse)
param_train, train_ll, valid_ll = train_logit(n, m, train_X, train_label, num_epoch, LEARNING_RATE, valid_X, valid_label)
# likelihood = compute_likelihood(n, m, train_X, train_label, num_epoch, LEARNING_RATE)
error_train, pred_train = pred(param_train, train_X, train_label)

# testing
test_X, test_label = get_X_label(test_data, m, feat_dict_inverse)
error_test, pred_test = pred(param_train, test_X, test_label)

# output results
pred_train_str = ""
for i in range(len(pred_train)):
    pred_train_str += str(int(pred_train[i])) + "\n"
pred_test_str = ""
for i in range(len(pred_test)):
    pred_test_str += str(int(pred_test[i])) + "\n"
writeFile(train_out, pred_train_str)
writeFile(test_out, pred_test_str)
metrics_str = "error(train): " + str(error_train) + "\nerror(test): " + str(error_test) + "\n"
writeFile(metrics_out, metrics_str)

# plot conditional negative log-likelihood
import matplotlib.pyplot as plt
epoch = list(range(1, 201))
plt.plot(epoch, train_ll, '-b', label='training data')
plt.plot(epoch, valid_ll, '-r', label='validation data')

plt.xlabel("number of epoch")
plt.ylabel("negative log likelihood")
plt.legend(loc='upper right')
plt.title("average negative log likelihood for Model 2")

# save image
plt.savefig("model2.png")  # should before show method

# show
plt.show()
