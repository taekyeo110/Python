import csv
import pandas as pd
import sys
import numpy as np
import string
import random
import matplotlib.pyplot as plt


#data셋 만들기(count,s_address,s_name,items)

df = pd.read_csv('C:/py36/data.csv')
t=65003
i=1
sdf = pd.DataFrame(columns = ['Items'])
item_list = ['면세탁','드라이크리닝','운동화세탁','명품크리닝','흔적제거','정장세탁']
for i in range(t):
    sdf.loc[i,'Items'] = random.choice(item_list)
    if i%100 == 1:
        print(i,",")
    else:
        continue
df = pd.concat([df,sdf], axis=1)

is_keep = df['영업상태구분코드'] == 1
seoul_df = df['소재지전체주소'].str.startswith("서울")
finall_df = df[is_keep & seoul_df]
seoul_yes_df = finall_df[['번호','소재지전체주소','사업장명','Items']]
seoul_yes_df.to_csv('C:/py36/data_phase3.csv',encoding='utf-8-sig')




#data셋 그래프를 위한 정제(item,count) -> 돌리고 header 바꿔줘야함

'''

df = pd.read_csv('C:/py36/data_phase3.csv')
unique = pd.Series.unique(df['Items'])
unique_count = pd.Series.value_counts(df['Items'])
print(unique_count)
unique_count.to_csv('C:/py36/data_phase4.csv',encoding='utf-8-sig')

'''