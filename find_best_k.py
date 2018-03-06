import numpy as np
import sys
import os.path

def main():
    args =  sys.argv
    # args[1]:num_doc [2]:num_word [3]:correct_topic
    D = 30
    K = 30
    VLB_mat = np.zeros([D,K-1])
    for d in range(1,D+1):
        for k in range(2,31):
            if(k <= 9):
                num_topic = '0' + str(k)
            else:
                num_topic = str(k)
            FILE = 'result/VB_sample_M' + args[1] + '_nd' + args[2] + '_K' + args[3] + '_' + str(d) + '/result_k' + num_topic + '.txt'
            if os.path.exists(FILE):
                lines = open(FILE, 'r').readlines()
                VLB = lines[0]
            else:
                VLB = '-1000000000000000000.0\n'
            VLB_mat[d-1,k-2] = float(VLB[:-1])
    index_list = VLB_mat.argmax(0)
    index = VLB_mat.argmax()
    max_d = (index // 29) + 1
    max_k = (index % 29) + 2
    print('max_data : ' + str(max_d) + '\n')
    print('max_k : ' + str(max_k) + '\n')
    output = open('ref/VB_' + args[1] + '_' + args[2] + '_' + args[3] + '.txt', 'w')
    output.write(str(max_d) + '\n')
    output.write(str(max_k) + '\n')
    for d in range(29):
        output.write(str(index_list[d] + 1) + '\n')

if(__name__ == '__main__'):
    main()
