import matplotlib.pyplot as plt

#%%
# plot average cross entropy
# plt.style.use("ggplot")
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


#%%


epoch = list(range(1, num_epoch+1))
plt.plot(epoch, train_cross_entropy, '-b', label='train data')
plt.plot(epoch, test_cross_entropy, '-r', label='test data')

plt.xlabel("number of hidden layers")
plt.ylabel("average cross entropy")
plt.legend(loc='upper right')
plt.title("average cross entropy with 5 hidden units")

# save image
plot_name = "hidden_units_" + str(hidden_units) + ".png"
plt.savefig(plot_name)  # should before show method

# show
plt.show()

#%%
