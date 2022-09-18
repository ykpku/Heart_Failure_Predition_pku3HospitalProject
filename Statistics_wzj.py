import csv
import json
import os
import pandas as pd
import numpy as np



# df = pd.read_excel("心衰最全6.17-ly删病人姓名ID.xlsx", engine='openpyxl')
# data = np.array(df)
# data_list = data.tolist()
# rows = []
# for item in data_list:
#     for i in range(len(item)):
#         if pd.isnull((item[i])):
#             item[i] = ""
#
#     rows.append(item)
# a=[]
# for i in range(644):
#     num = 0
#     for j in range(len(rows)):
#         if(rows[j][i]=="" or rows[j][i] == "."):
#             num+=1
#     a.append(num/1730)
# print(a)
# print(len(a))

# f1 = open('中间文件.csv', 'r')
# reader = csv.reader(f1)
# data = [row for row in reader]
# rows = []
# a=[]
# flag1=False
# flag2=False
# flag3=False
# flag4=False
# flag5=False
# flag6=False
# flag7=False
# for i in range(1,len(data)):
#     b=[]
#     for j in range(0,49,7):
#         if(data[i][j]=='1'):
#             flag1 = True
#     if(flag1==True):
#         b.append(1)
#     else:
#         b.append(2)
#     for j in range(1,49,7):
#         if(data[i][j]=='1'):
#             flag2 = True
#     if(flag2==True):
#         b.append(1)
#     else:
#         b.append(2)
#     for j in range(2,49,7):
#         if(data[i][j]=='1'):
#             flag3 = True
#     if(flag3==True):
#         b.append(1)
#     else:
#         b.append(2)
#     for j in range(3,49,7):
#         if(data[i][j]=='1'):
#             flag4 = True
#     if(flag4==True):
#         b.append(1)
#     else:
#         b.append(2)
#     for j in range(4,49,7):
#         if(data[i][j]=='1'):
#             flag5 = True
#     if(flag5==True):
#         b.append(1)
#     else:
#         b.append(2)
#     for j in range(5,49,7):
#         if(data[i][j]=='1'):
#             flag6 = True
#     if(flag6==True):
#         b.append(1)
#     else:
#         b.append(2)
#     for j in range(6,49,7):
#         if(data[i][j]=='1'):
#             flag7 = True
#     if(flag7==True):
#         b.append(1)
#     else:
#         b.append(2)
#
#     a.append(b)
#
# with open('中间结果.csv', 'w') as csvfile1:
#     writer = csv.writer(csvfile1)
#     for item in a:
#         writer.writerow(item)

# f1 = open('中间文件2.csv', 'r')
# reader = csv.reader(f1)
# data = [row for row in reader]
# rows = []
# a=[]
# for i in range(1,len(data)):
#     b=[]
#     if(data[i][0] == '1' or data[i][2] == '1' or data[i][3] == '1'):
#         b.append(1)
#     else:
#         b.append(2)
#     if(data[i][1] == '1' or data[i][4] == '1' or data[i][5] == '1' or data[i][6] == '1'):
#         b.append(1)
#     else:
#         b.append(2)
#     a.append(b)
# with open('中间结果2.csv', 'w') as csvfile1:
#     writer = csv.writer(csvfile1)
#     for item in a:
#         writer.writerow(item)

# f1 = open('中间文件3.csv', 'r')
# reader = csv.reader(f1)
# data = [row for row in reader]
# pre = ''
# pre1 = ''
# pre2 = ''
# pre3 = ''
# pre4 = ''
# for i in range(1,len(data)):
#     for j in range(0,35,5):
#         if(data[i][j]!=''):
#             pre = data[i][j]
#     for j in range(0,35,5):
#         if(data[i][j]==''):
#             data[i][j] = pre
#     for j in range(1,35,5):
#         if(data[i][j]!=''):
#             pre1 = data[i][j]
#     for j in range(1,35,5):
#         if(data[i][j]==''):
#             data[i][j] = pre1
#     for j in range(2,35,5):
#         if(data[i][j]!=''):
#             pre2 = data[i][j]
#     for j in range(2,35,5):
#         if(data[i][j]==''):
#             data[i][j] = pre2
#     for j in range(3,35,5):
#         if(data[i][j]!=''):
#             pre3 = data[i][j]
#     for j in range(3,35,5):
#         if(data[i][j]==''):
#             data[i][j] = pre3
#     for j in range(4,35,5):
#         if(data[i][j]!=''):
#             pre4 = data[i][j]
#     for j in range(4,35,5):
#         if(data[i][j]==''):
#             data[i][j] = pre4
#
# with open('中间结果3.csv', 'w') as csvfile1:
#     writer = csv.writer(csvfile1)
#     for item in data:
#         writer.writerow(item)


# f1 = open('中间文件4.csv', 'r')
# reader = csv.reader(f1)
# data = [row for row in reader]
# a=[]
#
# for i in range(1,len(data)):
#     max = 0
#     min = float(data[i][0])
#     for item in data[i]:
#         if (float(item) >= max):
#             max = float(item)
#         if (float(item) <= min):
#             min = float(item)
#     value = max - min
#     a.append(value)
#     print(a)
#
#
# with open('中间结果4.csv', 'w') as csvfile1:
#     writer = csv.writer(csvfile1)
#     for item in a:
#         writer.writerow([item])


# f1 = open('中间文件5.csv', 'r')
# reader = csv.reader(f1)
# data = [row for row in reader]
# a=[]
# for i in range(1,len(data)):
#      if(data[i][0] == '1' or data[i][1] == '1'):
#          a.append(1)
#      else:
#          a.append(2)
# with open('中间结果5.csv', 'w') as csvfile1:
#     writer = csv.writer(csvfile1)
#     for item in a:
#         writer.writerow([item])

f1 = open('origin_data/中间文件6.csv', 'r')
reader = csv.reader(f1)
data = [row for row in reader]
a=[]
for i in range(1,len(data)):
    b = []
    for item in data[i]:
        if(float(item)<400):
            b.append(1)
        else:
            b.append(0)
    a.append(b)
with open('origin_data/中间结果6.csv', 'w') as csvfile1:
    writer = csv.writer(csvfile1)
    for item in a:
        writer.writerow(item)

