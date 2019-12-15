import sys
import numpy as np
import math
import copy

# define a node data structure
class Node:
    def __init__(self, depth, train_data, feature_used):
        self.left = None
        self.right = None
        self.left_elem = None
        self.right_elem = None
        self.split_feature = None
        self.split_feature_index = None
        self.depth = depth
        self.train_data = train_data
        self.feature_used = feature_used  # 0: unused, 1: used
        self.is_leaf = False
        self.leaf_label = None


# compute mutual info of y and a specific feature
def compute_mutual_info(y, x_tmp, Hy):
    unique, counts = np.unique(x_tmp, return_counts=True)
    n_total = sum(counts)
    if (len(counts) > 1):  # 2 categories
        y_x0 = y[np.where(x_tmp==unique[0])]
        y_x1 = y[np.where(x_tmp==unique[1])]
        Hy_x0 = compute_entropy(y_x0)
        Hy_x1 = compute_entropy(y_x1)
        x_type0_prob = counts[0] / n_total
        x_type1_prob = counts[1] / n_total
        Hy_x = x_type0_prob * Hy_x0 + x_type1_prob * Hy_x1
    else:
        y_x0 = y[np.where(x_tmp==unique[0])]
        Hy_x0 = compute_entropy(y_x0)
        x_type0_prob = counts[0] / n_total
        Hy_x = x_type0_prob * Hy_x0
    info = Hy - Hy_x
    return(info)


# get the index with the largest mutual info
def get_largest_info_index(node):
    train_data = node.train_data
    y_train = train_data[:, -1]
    x_train = train_data[:, :-1]
    Hy = compute_entropy(y_train)
    mutual_info = np.zeros(len(x_train[0]))
    m, n = train_data.shape
    for i in range(len(x_train[0])):
        if node.feature_used[i] == 0:  # feature has not used
            mutual_info[i] = compute_mutual_info(y_train, train_data[:, i], Hy)
    # get index with the largest info
    largest_index = np.argmax(mutual_info)
    largest_info = max(mutual_info)
    if largest_info <=0:
        largest_index = -1
    return largest_index


# calculate the entropy of label
def compute_entropy(y):
    base = 2
    unique, counts = np.unique(y, return_counts=True)
    n_total = sum(counts)
    if (len(counts) > 1):  # 2 categories
        type1_prob = counts[0] / n_total
        type2_prob = counts[1] / n_total
        entropy = - type1_prob * math.log(type1_prob, base) -  \
            type2_prob * math.log(type2_prob, base)
    else:
        type1_prob = counts[0] / n_total
        entropy = - type1_prob * math.log(type1_prob, base)
    return(entropy)


# train a decision tree
def decisionTreeTrain(node, max_depth, feature_name):
    # stopping rules
    # base cases
    if (node.depth > max_depth) or (0 not in node.feature_used):
        majority_vote_node(node)
    else:  # build tree
        largest_index = get_largest_info_index(node)
        if largest_index == -1:
            majority_vote_node(node)
        else:
            node.feature_used[largest_index] = 1
            # split the node and data based on the current best feature
            train_data = node.train_data
            feature_col = train_data[:, largest_index]
            left_elem = train_data[0, largest_index]
            subdata0 = train_data[np.where(feature_col==left_elem)]
            subdata1 = train_data[np.where(feature_col!=left_elem)]
            right_elem = subdata1[0, largest_index]
            node.left_elem = left_elem
            node.right_elem = right_elem
            node.split_feature = feature_name[largest_index]
            node.split_feature_index = largest_index
            curr_depth = node.depth
            node.left = Node(curr_depth+1, subdata0, copy.copy(node.feature_used))
            node.right = Node(curr_depth+1, subdata1, copy.copy(node.feature_used))
            # left branch
            print_node_info(node, node.left_elem, node.left.train_data)
            decisionTreeTrain(node.left, max_depth, feature_name)
            # right branch
            print_node_info(node, node.right_elem, node.right.train_data)
            decisionTreeTrain(node.right, max_depth, feature_name)


# use majority vote to determine a leaf
def majority_vote_node(node):
    node.is_leaf = True
    train_data = node.train_data
    y_train = train_data[:, -1]
    unique, counts = np.unique(y_train, return_counts=True)
    node.leaf_label = unique[np.argmax(counts)]
    if (len(counts)==2) and (counts[0]==counts[1]):  
        # choose the first one if the number of two categories are the same
        node.leaf_label = y_train[0]


# get the prediction result of a specific test point
def decisionTreeTest(node, test_point, train_data):
    if node.is_leaf:
        return(node.leaf_label)
    else:
        x_train = train_data[:, :-1]
        if (node.left_elem == test_point[node.split_feature_index]):
            # left branch
            return decisionTreeTest(node.left, test_point, train_data)
        else:  # right branch
            return decisionTreeTest(node.right, test_point, train_data)


# get the prediction error
def pred_error(test_data, root, train_data):
    y_test = test_data[:, -1]
    pred_test = np.zeros(len(test_data), object)
    error_test = 0
    for i in range(len(test_data)):
        pred_test[i] = decisionTreeTest(root, test_data[i], train_data)
        if (y_test[i] != pred_test[i]):
            error_test += 1
    error_test /= len(test_data)
    return((pred_test, error_test))


# print first line of the tree
def print_first_line(train_data):
    y_train = train_data[:, -1]
    unique, counts = np.unique(y_train, return_counts=True)
    print("[" + str(counts[0]) + " " + unique[0] + " /" + str(counts[1]) \
        + " " + unique[1] + "]")


# print the whole tree
def print_node_info(node, edge_value, edge_data):
    print("| " * node.depth, end="")
    train_edge_data = edge_data
    y_train_edge = train_edge_data[:, -1]
    unique, counts = np.unique(y_train_edge, return_counts=True)
    feature = node.split_feature
    if len(counts)==1:
        counts = np.append(counts, 0)
        unique_tmp = np.unique(node.train_data)
        if unique_tmp[0] != unique[0]:
            unique = np.append(unique, unique_tmp[0])
        else:
            unique = np.append(unique, unique_tmp[1])
    print(feature + " = " + edge_value + ": [" + str(counts[0]) + " " + unique[0] + "/" + str(counts[1]) + " " + unique[1] + "]")


# main function
def main(train_input, test_input, max_depth, train_out, test_out, metrics_out):
    train_data = np.genfromtxt(train_input,dtype=np.dtype(np.unicode_))
    test_data = np.genfromtxt(test_input,dtype=np.dtype(np.unicode_))
    feature_name = train_data[0,:][:-1]
    feature_num = len(feature_name)
    feature_used = np.zeros(feature_num, dtype=int)  # 0: unused, 1: used
    train_data = train_data[1:]  # remove feature row
    test_data = test_data[1:]
    # creat root node and start training the decision tree
    print_first_line(train_data)
    root = Node(depth=1, train_data=train_data, feature_used=feature_used)
    decisionTreeTrain(root, max_depth, feature_name)

    # prediction
    pred_train, error_train = pred_error(train_data, root, train_data)
    pred_test, error_test = pred_error(test_data, root, train_data)
    np.savetxt(train_out, pred_train, delimiter=" ", fmt="%s")
    np.savetxt(test_out, pred_test, delimiter=" ", fmt="%s")
    output_list = list()
    output_list.append(["error(train): " + str(error_train)])
    output_list.append(["error(test): " + str(error_test)])
    np.savetxt(metrics_out, output_list, delimiter=" ", fmt="%s")


if __name__ == "__main__": 
    # train_input = "education_train.tsv"
    # test_input = "education_test.tsv"
    # max_depth = 4
    # train_out = "education_4_train.labels"
    # test_out = "education_4_test.labels"
    # metrics_out = "education_4_metrics.txt"
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    main(train_input, test_input, max_depth, train_out, test_out, metrics_out)



