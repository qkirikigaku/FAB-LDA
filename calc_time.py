import sys

def main():
    args = sys.argv
    #[1] = num_doc [2] = num_word [3] = correct_topic
    
    times = list()
    for i in range(1,31):
        file = 'result/sample_M' + args[1] + '_nd' + args[2] + '_K' + args[3] + '_' + str(i) + '/execution_time.txt'
        times.append(float(open(file, 'r').readline()))
    sum_time = 0
    for i in range(len(times)):
        sum_time += times[i]
    mean_time = sum_time / 30
    out_file = 'ref/time_' + args[1] + '_' + args[2] + '_' + args[3] + '.txt'
    out = open(out_file, 'w')
    out.write(str(mean_time) + '\n')

if __name__ == '__main__':
    main()
