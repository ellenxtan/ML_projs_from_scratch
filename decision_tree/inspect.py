import sys
import numpy as np
import math

# calculate the entropy of the labels before any split
def inspect(infile, outfile):
	train_data = np.loadtxt(infile,dtype=str)

	y_train = train_data[:, -1]
	y_train = y_train[1:]  # remove the first one (attribute name)

	unique, counts = np.unique(y_train, return_counts=True)

	n_total = sum(counts)

	type1_prob = counts[0] / n_total
	type2_prob = counts[1] / n_total

	base = 2
	entropy = - type1_prob * math.log(type1_prob, base) - type2_prob * math.log(type2_prob, base)

	error = min(type1_prob, type2_prob)

	entropy_str = ["entropy:"] + [str(entropy)]
	error_str = ["error:"] + [str(error)]
	output_list = list()
	output_list.append(entropy_str)
	output_list.append(error_str)

	np.savetxt(outfile, output_list, delimiter=" ", fmt="%s")


if __name__ == "__main__": 
    infile = sys.argv[1]
    outfile = sys.argv[2]
    print("The input file is: %s" % (infile)) 
    print("The output file is: %s" % (outfile))
    inspect(infile, outfile)
