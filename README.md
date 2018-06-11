# HW
Homework 4 Lihongyi_2017-2018

## hw00

Q1. 輸入hw0_data.dat，請將指定column由小排到大並印出來到ans1.txt

A1:
```python
python q1.py 1 hwhw0_data.dat
```
Q2. 輸入一張圖，將圖上下顛倒，左右相反（旋轉180度），並輸出到ans2.png

A2: 
```python
python q2.py Lena.png
```

Q3. 使用豐原站的觀測記錄，分成train set跟test set，train set是豐原站每個月的前20天所有資料。test set則是從豐原站剩下的資料中取樣出來。

train.csv: 每個月前20天的完整資料。

test_X.csv: 從剩下的10天資料中取樣出連續的10小時為一筆，前九小時的所有觀測數據當作feature，第十小時的PM2.5當作answer。
一共取出240筆不重複的test data，請根據feauure預測這240筆的PM2.5。

A3: ...ing 