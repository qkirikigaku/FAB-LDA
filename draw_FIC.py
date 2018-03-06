import numpy as np
import matplotlib.pyplot as plt
import sys

def main():
    args = sys.argv
    # args[1] = num_doc [2] = num_word [3] = correct_topic [4] = num_K
    if (int(args[4]) >= 10):
        topic = args[4]
    else:
        topic = '0' + args[4]

    input_file = 'result/sample_M' + args[1] + '_nd' + args[2] + '_K' + args[3] + '/FIC_k' + topic + '.txt'
    lines = open(input_file, 'r').readlines()
    FIC = list()
    index = 0
    for line in lines:
        fic = line[:-1]
        if(index % 2 == 0):
            FIC.append(fic)
        index += 1
    iter = list()
    leng = len(FIC)
    for i in range(leng):
        iter.append(i+1)
    plt.figure()
    left = iter
    height = FIC
    plt.plot(left, height)
    plt.xlabel('iteration')
    plt.ylabel('FIC')
    plt.savefig('result/sample_M' + args[1] + '_nd' + args[2] + '_K' + args[3] + '/figure/FIC.png', bbox_inches='tight')

if(__name__ == '__main__'):
    main()
