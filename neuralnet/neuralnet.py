import sys
import numpy as np
import random

# test command
# python feature.py smalltrain_data.tsv smallvalid_data.tsv smalltest_data.tsv dict.txt model1_formatted_train.tsv model1_formatted_valid.tsv model1_formatted_test.tsv 1


# read files
def readFile(path):
    with open(path, "rt") as f:
        return f.read()


# write files
def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)


# load data
# train_input = sys.argv[1]
# test_input = sys.argv[2]
# train_out = sys.argv[3]
# test_out = sys.argv[4]
# metrics_out = sys.argv[5]
# num_epoch = int(sys.argv[6])
# hidden_units = int(sys.argv[7])
# init_flag = int(sys.argv[8])
# learning_rate = float(sys.argv[9])

folder = "../handout/"
train_input = folder + "largeTrain.csv"
test_input = folder + "largeValidation.csv"
train_out = "model1train_out.labels"
test_out = "model1test_out.labels"
metrics_out = "model1metrics_out.txt"
num_epoch = 100
hidden_units = 50
init_flag = 1
learning_rate = 0.001


# parse data into feature matrix and label vector
def get_X_label(mydata):
    tmp_data = mydata.split("\n")[:-1]  # remove the last line
    N = len(tmp_data)
    M = len(tmp_data[0].split(","))  # don't need to remove 1 (for all 1 col)
    X = np.zeros((N, M))  # design matrix
    X[:, 0] = 1  # first col = 1
    label = []
    for i in range(N):
        tmp_line = tmp_data[i].split(",")
        label.append(int(tmp_line[0]))
        tmp_feat = tmp_line[1:]
        X[i, 1:] = list(map(int, tmp_feat)) 
    return (X, label, M-1)  # M-1 is number of features


# get y to be one hot encoding
# https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
def get_one_hot(y, classes):
    y = np.array(y)
    res = np.eye(classes)[np.array(y).reshape(-1)]
    return res.reshape(list(y.shape)+[classes])


# initialize parameters alpha & beta
def init(init_flag, M, D, K):
    if init_flag == 1:
        alpha = np.random.uniform(-0.1, 0.1, (D, M+1))
        beta = np.random.uniform(-0.1, 0.1, (K, D+1))
    else:
        alpha = np.zeros((D, M+1))
        beta = np.zeros((K, D+1))
        # print(alpha.shape)
    return (alpha, beta)


# read train/test data
train_data = readFile(train_input)
test_data = readFile(test_input)

K = 10  # number of outputs (0-9)
train_X, train_label, M = get_X_label(train_data)
train_y_onehot = get_one_hot(train_label, K)
test_X, test_label, m2 = get_X_label(test_data)
test_y_onehot = get_one_hot(test_label, K)
assert(M==m2)  # number of features

alpha_init, beta_init = init(init_flag, M, hidden_units, K)

# forward computation
def NN_forward(X, y, alpha, beta):
    a = linear_forward(X, alpha)
    z = sigmoid_forward(a)
    z = np.hstack((1, z))
    b = linear_forward(z, beta)
    y_hat = softmax_forward(b)
    J = cross_entropy_forward(y, y_hat)
    interm = (a, z, b, y_hat, J)
    return interm

# backpropagation
def NN_backward(X, label, alpha, beta, interm):
    a, z, b, y_hat, J = interm
    g_j = 1
    g_y_hat = cross_entropy_backward(label, y_hat, J, g_j)  # shape=10
    g_b = softmax_backward(y_hat, g_y_hat)
    g_beta, g_z = linear_backward(z, beta, g_b)  # g_beta:10*5
    g_a = sigmoid_backward(z, g_z)
    g_a = g_a[1:]  # remove the first element
    g_a = np.reshape(g_a, (1, g_a.shape[0]))  # g_a: 1*4
    g_alpha, g_x = linear_backward(X, alpha, g_a)
    return (g_alpha, g_beta)

# linear function in forward
def linear_forward(X, alpha):
    return np.matmul(alpha, X)

# linear function in backward
# z: 5*1
# beta: 10*5
# g_b: 1*10
def linear_backward(z, beta, g_b):
    z = np.reshape(z, (z.shape[0], 1))
    g_beta = np.matmul(g_b.T, z.T)
    g_z = np.dot(beta.T, g_b.T)
    return (g_beta, g_z)

# sigmoid function in forward
def sigmoid_forward(a):
    return 1 / (1 + np.exp(-a))

# sigmoid function in backward; z: 5*1; g_z: 5*1
def sigmoid_backward(z, g_z):
    z = np.reshape(z, (z.shape[0], 1))
    return np.multiply(g_z, np.multiply(z, 1-z))

# softmax function in forward
def softmax_forward(b):
    exp_b = np.exp(b)
    # print(exp_b / exp_b.sum())
    return exp_b / exp_b.sum()

# softmax function in backward  ##
def softmax_backward(y_hat, g_y_hat):
    g_y_hat = np.reshape(g_y_hat, (len(g_y_hat),1))
    b_dash = np.reshape(y_hat, (len(y_hat),1))
    return np.matmul(g_y_hat.T, np.subtract(np.diag(y_hat), np.matmul(b_dash,b_dash.T)))


# cross entropy in forward
def cross_entropy_forward(label, y_hat):
    return -np.dot(label.T, np.log(y_hat))

# cross entropy in backward
def cross_entropy_backward(label, y_hat, J, g_j):
    return -np.divide(label, y_hat)  # shape = 10

# compute average cross entropy
def compute_avg_entropy(X, label, alpha, beta):  # label is one hot
    n = len(label)
    entropy = 0
    for i in range(n):
        a = linear_forward(X[i, :], alpha)
        z = sigmoid_forward(a)
        z = np.hstack((1, z))
        b = linear_forward(z, beta)
        y_hat = softmax_forward(b)
        entropy += np.dot(label[i], np.log(y_hat))
    entropy = (-1) * entropy / n
    return entropy

# training
def SGD(train_X, train_label, test_X, test_label, alpha, beta, num_epoch, learning_rate):
    n = len(train_label)  # number of sample size
    train_cross_entropy = []
    test_cross_entropy = []
    for k in range(num_epoch):
        print(k)
        for i in range(n):
            interm = NN_forward(train_X[i, :], train_label[i], alpha, beta)
            g_alpha, g_beta = NN_backward(train_X[i, :], train_label[i], alpha, beta, interm)
            alpha = alpha - learning_rate * g_alpha
            beta = beta - learning_rate * g_beta
        train_cross_entropy.append(compute_avg_entropy(train_X, train_label, alpha, beta))
        test_cross_entropy.append(compute_avg_entropy(test_X, test_label, alpha, beta))
    return (alpha, beta, train_cross_entropy, test_cross_entropy)

# prediction for one sample
def get_y_hat(X, alpha, beta):
    a = linear_forward(X, alpha)
    z = sigmoid_forward(a)
    z = np.hstack((1, z))
    b = linear_forward(z, beta)
    y_hat = softmax_forward(b)
    return y_hat


# predict the test data
def predict(test_X, test_label, alpha, beta):
    n = len(test_label)
    error_rate = 0
    test_y_out = []
    for i in range(n):
        y_hat = get_y_hat(test_X[i, :], alpha, beta)
        y_pred = np.argmax(y_hat)
        # print(y_pred)
        test_y_out.append(y_pred)
        if y_pred != test_label[i]:
            error_rate += 1
    error_rate /= n
    return (test_y_out, error_rate)

        

# main
alpha, beta, train_cross_entropy, test_cross_entropy = SGD(train_X, train_y_onehot, test_X, test_y_onehot, alpha_init, beta_init, num_epoch, learning_rate)
pred_train, error_train = predict(train_X, train_label, alpha, beta)
pred_test, error_test = predict(test_X, test_label, alpha, beta)
# print(error_train, error_test)
# print(train_cross_entropy, test_cross_entropy)

# output results
# output label files
pred_train_str = ""
for i in range(len(pred_train)):
    pred_train_str += str(int(pred_train[i])) + "\n"
pred_test_str = ""
for i in range(len(pred_test)):
    pred_test_str += str(int(pred_test[i])) + "\n"
writeFile(train_out, pred_train_str)
writeFile(test_out, pred_test_str)

# output metrics file
epoch_str = ""
for i in range(num_epoch):
    epoch_str += "epoch=" + str(i+1) + " crossentropy(train): " + str(train_cross_entropy[i]) \
        + "\nepoch=" + str(i+1) + " crossentropy(test): " + str(test_cross_entropy[i]) + "\n"
error_str = "error(train): " + str(error_train) + "\nerror(test): " + str(error_test) + "\n"
metrics_str = epoch_str + error_str
writeFile(metrics_out, metrics_str)

print(train_cross_entropy[-1], test_cross_entropy[-1])
hidden_units = [5, 20, 50, 100, 200]
train = [0.5368527675493823, 0.13284138665486048, 0.053223347968168806, 0.04753874124552824, 0.046427876315143946]
test = [0.7015223942295157, 0.5294168576738014, 0.47391693320386075, 0.43430637776973446, 0.4121598427687438]


# plot average cross entropy vs learning rate
import matplotlib.pyplot as plt
epoch = list(range(1, num_epoch+1))
plt.plot(epoch, train_cross_entropy, '-b', label='train data')
plt.plot(epoch, test_cross_entropy, '-r', label='test data')
plt.xlabel("number of epochs")
plt.ylabel("average cross entropy")
plt.legend(loc='upper right')
plt.title("average cross entropy with learning rate of 0.001")
# save image
plot_name = "graph_learn_rate_" + str(learning_rate) + ".png"
plt.savefig(plot_name)  # should before show method
# show
plt.show()


# plot average cross entropy vs hidden units
hidden_units = [5, 20, 50, 100, 200]
train = [0.5368527675493823, 0.13284138665486048, 0.053223347968168806, 0.04753874124552824, 0.046427876315143946]
test = [0.7015223942295157, 0.5294168576738014, 0.47391693320386075, 0.43430637776973446, 0.4121598427687438]
plt.plot(hidden_units, train, color="b", marker="o", label="train data")
plt.plot(hidden_units, test, color="r", marker="v", label="test data")
plt.xlabel("number of hidden units")
plt.ylabel("average cross-entropy")
plt.legend(loc="upper right")
plt.title("average cross-entropy vs number of hidden units")

plot_name = "graph_hidden_units.png"
plt.savefig(plot_name)  # should before show method
plt.show()