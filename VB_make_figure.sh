#!/bin/bash
num_doc=$1
num_word=$2
num_topic=$3
str1=idenken:project/FAB_LDA/code/result/VB_sample_M${num_doc}_nd${num_word}_K${num_topic}*
scp -r ${str1} result

#measure the time
python VB_calc_time.py ${num_doc} ${num_word} ${num_topic}

#make figure
python find_best_k.py ${num_doc} ${num_word} ${num_topic}
FILENAME=ref/VB_${num_doc}_${num_word}_${num_topic}.txt
cnt=0
array=()
while read line;
do
    cnt=$(expr $cnt + 1)
    if test $cnt -eq 1; then
	num_data=$line
    fi
    if test $cnt -eq 2; then
	number_of_topic=$line
    fi
    if test $cnt -ge 3; then
	array+=($line)
    fi
done<$FILENAME
echo ${number_of_topic}
for i in $(seq 2 30); do
    d_num=${array[i-2]}
    str1=result/VB_sample_M${num_doc}_nd${num_word}_K${num_topic}_${d_num}
    str2=../VB_sample_M${num_doc}_nd${num_word}_K${num_topic}_${num_data}
    cd ${str1}
    if test ${i} -le 9; then
	str3=result_k0${i}.txt
    else
	str3=result_k${i}.txt
    fi
    cp ${str3} ${str2}
    cd ../..
done
for i in $(seq 1 30); do
    str1=result/VB_sample_M${num_doc}_nd${num_word}_K${num_topic}_${i}
    if test ${i} -eq ${num_data}; then
	cd ${str1}
	mkdir figure
	cd ../..
	python comparison_K.py ${num_doc} ${num_word} ${num_topic} ${i}
	str2=result/VB_sample_M${num_doc}_nd${num_word}_K${num_topic}
	mv ${str1} ${str2}
    else
	rm -r ${str1}
    fi
done

str2=result/VB_sample_M${num_doc}_nd${num_word}_K${num_topic}
mv ref/VB_time_${num_doc}_${num_word}_${num_topic}.txt $str2

cd ${str2}
cd figure
mkdir ${number_of_topic}
cd ../../..
sh VB_make_figure_re.sh ${num_doc} ${num_word} ${num_topic} ${number_of_topic}