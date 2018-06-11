# Q1. 輸入hw0_data.dat，請將指定column由小排到大並印出來到ans1.txt

import  sys

index = int(sys.argv[1])
input_file = str(sys.argv[2])

with open(input_file) as f:
    line = f.readlines()[index].strip()
    print(line)
    words = line.split(" ")
    nums = map(float,words)
    sorted_nums = sorted(nums)
    output = ",".join(map(str, sorted_nums))
    print(output)

open("ans1.txt", "w").write(output)