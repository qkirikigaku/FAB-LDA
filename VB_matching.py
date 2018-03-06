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

input_file = 'result/VB_sample_M' + args[1] + '_nd' + args[2] + '_K' + args[3] + '/result_k' + topic + '.txt'

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

Pre_ex = np.argsort(JS_ex)
Pre_co = np.argsort(JS_co)

match = '{'
for i in range(K):
    min_j = 0
    for j in range(1,30):
        if(JS_ex[i,j] < JS_ex[i,min_j]):
            min_j = j
    match += str(i) + ': ' + str(min_j)
    if(i != K-1):
        match += ', '
match += '}'

output_file = "result/VB_sample_M" + args[1] + '_nd' + args[2] + '_K' + args[3] +'/figure/' + args[4] + '/matching.txt' 

output = open(output_file,'w')
output.write(match)
output.close()
