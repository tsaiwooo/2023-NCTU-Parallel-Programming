#!/bin/bash

# Array to store execution times
declare -a times1
declare -a times2
declare -a times3

# Run commands 50 times
for i in {1..50}
do
    output1=$(make clean && make && ./test_auto_vectorize -t 2 | tail -n 1)
    time1=$(echo $output1 | grep -oP "\d+\.\d+(?=sec)")
    times1+=($time1)

    output2=$(make clean && make VECTORIZE=1 && ./test_auto_vectorize -t 2 | tail -n 1)
    time2=$(echo $output2 | grep -oP "\d+\.\d+(?=sec)")
    times2+=($time2)

    output3=$(make clean && make VECTORIZE=1 AVX2=1 && ./test_auto_vectorize -t 2 | tail -n 1)
    time3=$(echo $output3 | grep -oP "\d+\.\d+(?=sec)")
    times3+=($time3)
done

# Sorting the arrays
sorted_times1=($(printf '%s\n' "${times1[@]}" | sort -n))
sorted_times2=($(printf '%s\n' "${times2[@]}" | sort -n))
sorted_times3=($(printf '%s\n' "${times3[@]}" | sort -n))

# Calculate the median
n=$((${#sorted_times1[@]} / 2))
median1=$(echo "scale=5; (${sorted_times1[$n]} + ${sorted_times1[$((n-1))]} ) / 2" | bc)
median2=$(echo "scale=5; (${sorted_times2[$n]} + ${sorted_times2[$((n-1))]} ) / 2" | bc)
median3=$(echo "scale=5; (${sorted_times3[$n]} + ${sorted_times3[$((n-1))]} ) / 2" | bc)

echo "Median for Command 1: $median1"
echo "Median for Command 2: $median2"
echo "Median for Command 3: $median3"
