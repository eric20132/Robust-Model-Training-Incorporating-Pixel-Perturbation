import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import csv

with open("accuracy/accuracy1.csv", "r", encoding='utf-8-sig') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    accuracy1 = list(reader)
csv_file.close()

accuracy1 = [list( map(float,i) ) for i in accuracy1] # string to float

with open("accuracy/accuracy2.csv", "r", encoding='utf-8-sig') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    accuracy2 = list(reader)
csv_file.close()

accuracy2 = [list( map(float,i) ) for i in accuracy2] # string to float

with open("accuracy/accuracy3.csv", "r", encoding='utf-8-sig') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    accuracy3 = list(reader)
csv_file.close()

accuracy3 = [list( map(float,i) ) for i in accuracy3] # string to float
print(accuracy1[0])

fig = plt.figure()
K=len(accuracy1[0])
x=np.linspace(1.0, K, num=K)
plt.plot(x,accuracy1[0], label='w/o defend',linewidth=1)
plt.plot(x,accuracy2[0], 'r--',label='w/ noisy defend',linewidth=1)
plt.plot(x,accuracy3[0], label='w/ patch-rand defend',linewidth=1)

# plt.title(r'Accuracy vs Attacking iteration')
plt.legend(loc='upper right')
plt.xlabel('Num of iteraions')
plt.ylabel('Accuracy')

plt.savefig('accuracy.png')
plt.clf()