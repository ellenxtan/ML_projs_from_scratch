import sys

# test command
# python feature.py smalltrain_data.tsv smallvalid_data.tsv smalltest_data.tsv dict.txt model1_formatted_train.tsv model1_formatted_valid.tsv model1_formatted_test.tsv 1

TRIM_THRESHOLD = 4

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
formatted_train_out = sys.argv[5]
formatted_validation_out = sys.argv[6]
formatted_test_out = sys.argv[7]
feature_flag = int(sys.argv[8])


# read dict data
dict_data = readFile(dict_input).split("\n")
dict_data = dict_data[:-1]  # remove the last empty line
mydict = dict()
for i in range(len(dict_data)):
    tmp = dict_data[i].split()
    mydict[tmp[0]] = tmp[1]


# read train/valid/test data
train_data = readFile(train_input)
valid_data = readFile(validation_input)
test_data = readFile(test_input)


# feature engineering on train/valid/test data for model 1
def feat_engin_model1(mydata, output_file_name):
    mydata = mydata.split("\n")
    mydata = mydata[:-1]  # remove the last empty line
    label = []
    feature = []
    for i in range(len(mydata)):
        tmp_line = mydata[i].split("\t")
        label.append(tmp_line[0])
        feature_str = tmp_line[1].split()
        tmp_dict = dict()
        for j in range(len(feature_str)):
            if (feature_str[j] in mydict) and (mydict[feature_str[j]] not in tmp_dict):
                tmp_dict[mydict[feature_str[j]]] = 1
        feature.append(tmp_dict)

    output_str = ""
    for i in range(len(label)):
        output_str += label[i] + "\t"
        feat_str = ":1\t".join(list(feature[i].keys()))
        output_str += feat_str + ":1\n"

    writeFile(output_file_name, output_str)


# feature engineering on train/valid/test data for model 2
def feat_engin_model2(mydata, output_file_name, TRIM_THRESHOLD):
    mydata = mydata.split("\n")
    mydata = mydata[:-1]  # remove the last empty line
    label = []
    feature = []
    for i in range(len(mydata)):
        tmp_line = mydata[i].split("\t")
        label.append(tmp_line[0])
        feature_str = tmp_line[1].split()
        tmp_dict = dict()
        for j in range(len(feature_str)):
            if (feature_str[j] in mydict) and (mydict[feature_str[j]] not in tmp_dict):
                tmp_dict[mydict[feature_str[j]]] = 1
            elif (feature_str[j] in mydict) and (mydict[feature_str[j]] in tmp_dict):
                # print(feature_str[j])
                # print(mydict[feature_str[j]])
                tmp_dict[mydict[feature_str[j]]] += 1
        tmp_dict2 = dict()
        for key in tmp_dict:
            if tmp_dict[key] < TRIM_THRESHOLD:
                tmp_dict2[key] = 1
        feature.append(tmp_dict2)

    output_str = ""
    for i in range(len(label)):
        output_str += label[i] + "\t"
        feat_str = ":1\t".join(list(feature[i].keys()))
        output_str += feat_str + ":1\n"

    writeFile(output_file_name, output_str)


# main function
if feature_flag == 1:  # model 1
    feat_engin_model1(train_data, formatted_train_out)
    feat_engin_model1(valid_data, formatted_validation_out)
    feat_engin_model1(test_data, formatted_test_out)
else:  # model 2
    feat_engin_model2(train_data, formatted_train_out, TRIM_THRESHOLD)
    feat_engin_model2(valid_data, formatted_validation_out, TRIM_THRESHOLD)
    feat_engin_model2(test_data, formatted_test_out, TRIM_THRESHOLD)
