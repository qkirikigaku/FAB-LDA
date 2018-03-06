import sys
import networkx as nx
import numpy as np
from scipy import stats
from ortoolpy import stable_matching


def swap(i,list):
    temp = list[i]
    list[i] = list[i+1]
    list[i+1] = temp

args = sys.argv
"""args[1]:num_doc [2]:num_word [3]:correct_topic [4]:num_K"""

K = int(args[4])
if(K <= 9):
    topic = '0' + args[4]
else:
    topic = args[4]

input_file = 'result/sample_M' + args[1] + '_nd' + args[2] + '_K' + args[3] + '/result_k' + topic + '.txt'

ex = open(input_file)
co = open('data/signature_probability.txt')

data_ex = ex.readlines()
count_ex = 0
p_ex = np.zeros([K,96])
for line in data_ex:
    if((count_ex > 1) and (count_ex < K + 2)):
        words = line.split()
        for signature in range(96):
            p_ex[count_ex-2,signature] = float(words[signature])
    count_ex += 1

data_co = co.readlines()
count_co = 0
p_co = np.zeros([30,96])
for line in data_co:
    if(count_co != 0):
        words = line.split()
        for signature in range(33):
            if(signature >= 3):
                p_co[signature-3,count_co-1] = float(words[signature])
    count_co += 1

ex.close()
co.close()

JS_ex = np.zeros([K,30])
JS_co = np.zeros([30,K])
for i in range(K):
    for j in range(30):
        qx_ex = (p_ex[i]+p_co[j])/2
        JS_ex[i,j] = 0.5*(stats.entropy(p_ex[i],qx_ex,2) + stats.entropy(p_co[j],qx_ex,2))

for i in range(30):
    for j in range(K):
        qx_co = (p_co[i]+p_ex[j])/2
        JS_co[i,j] = 0.5*(stats.entropy(p_co[i],qx_co,2) + stats.entropy(p_ex[j],qx_co,2))

print(JS_ex)

Pre_ex = JS_ex.argsort()
Pre_co = JS_co.argsort()

print(Pre_ex)

match_list = list()
for i in range(K):
    min_j = 0
    for j in range(1,30):
        if(JS_ex[i,j] < JS_ex[i,min_j]):
            min_j = j
    match_list.append(min_j)

print(match_list)
flag = 0
while(flag == 0):
    pre_flag = 0
    for i in range(K):
        for j in range(i+1, K):
            if(match_list[i] == match_list[j]):
                print('duplicatd match:' + str(match_list[i]))
                temp_j_index = list(Pre_ex[j]).index(match_list[j])
                temp_i_index = list(Pre_ex[i]).index(match_list[i])
                j_candidate = temp_j_index + 1
                i_candidate = temp_i_index + 1
                first = JS_ex[i, match_list[i]] + JS_ex[j, Pre_ex[j, j_candidate]]
                second = JS_ex[j, match_list[j]] + JS_ex[i, Pre_ex[i, i_candidate]]
                print('j candidate:' + str(Pre_ex[j, j_candidate]))
                print('i candidate:' + str(Pre_ex[i, i_candidate]))
                if(first < second):
                    match_list[j] = Pre_ex[j, j_candidate]
                else:
                    match_list[i] = Pre_ex[i, i_candidate]
                print(match_list)
    for i in range(K):
        for j in range(i+1, K):
            if(match_list[i] == match_list[j]):
                pre_flag += 1
    if(pre_flag == 0):
        flag = 1

match = '{'
for i in range(K):
    match += str(i) + ': ' + str(match_list[i])
    if(i != K-1):
        match += ', '
match += '}'

output_file = "result/sample_M" + args[1] + '_nd' + args[2] + '_K' + args[3] +'/figure/' + args[4] + '/matching_improved.txt' 

output = open(output_file,'w')
output.write(match)
output.close()

outJS_file = "result/sample_M" + args[1] + '_nd' + args[2] + '_K' + args[3] + '/figure/' + args[4] + '/mean_JS.txt'
output = open(outJS_file, 'w')
mean_JS = 0
for i in range(K):
    mean_JS += JS_ex[i, match_list[i]]
mean_JS /= K
output.write('mean_JS : ' + str(mean_JS) + '\n')
miss = 0
for i in range(K):
    if(match_list[i] not in [0,1,2,3,4,5,6,7,8,9]):
        miss += 1
output.write('success_match : ' + str(K-miss) + '\n')
output.write('miss_match : ' + str(miss) + '\n')
output.close()
