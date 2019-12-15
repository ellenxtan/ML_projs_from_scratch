# test 10,100,1000,10000
# Accuracy: 0.8325026284023208
# Accuracy: 0.8335539893306335
# Accuracy: 0.8564697636384876
# Accuracy: 0.9225692145944473

# train 
# Accuracy: 0.8328168509141984
# Accuracy: 0.8336861129254841
# Accuracy: 0.8607658345651971
# Accuracy: 0.9378649549899077

train_acc = [0.8328168509141984, 0.8336861129254841, 0.8607658345651971, 0.9378649549899077]
test_acc = [0.8325026284023208, 0.8335539893306335, 0.8564697636384876, 0.9225692145944473]
sequence = [10, 100, 1000, 10000]

import matplotlib.pyplot as plt
plt.plot(sequence, train_acc, color="b", marker="o", label="train accuracy")
plt.plot(sequence, test_acc, color="r", marker="v", label="test accuracy")
plt.xlabel("number of sequences")
plt.ylabel("accuracy")
plt.legend(loc="upper left")
plt.title("number of sequences vs accuracy")

plot_name = "graph_hmm.png"
plt.savefig(plot_name)  # should before show method
plt.show()

[round(x,4) for x in train_acc]
[round(x,4) for x in test_acc]