import csv
import pandas as pd
import sys
import numpy as np
import string
import random
import glob
import os

#지역별 data셋 불러오기

jongro_df = pd.read_csv('C:/py36/data_jongro.csv')
'''
yongsan_df = pd.read_csv('C:/py36/data_yongsan.csv')
sungdong_df = pd.read_csv('C:/py36/data_sungdong.csv')
gwangjin_df = pd.read_csv('C:/py36/data_gwangjin.csv')
dongdaemun_df = pd.read_csv('C:/py36/data_dongdaemun.csv')
junglang_df = pd.read_csv('C:/py36/data_junglang.csv')
sungbuk_df = pd.read_csv('C:/py36/data_sungbuk.csv')
gangbuk_df = pd.read_csv('C:/py36/data_gangbuk.csv')
dobong_df = pd.read_csv('C:/py36/data_dobong.csv')
nowon_df = pd.read_csv('C:/py36/data_nowon.csv')
uenpueng_df = pd.read_csv('C:/py36/data_uenpueng.csv')
sudaemun_df = pd.read_csv('C:/py36/data_sudaemun.csv')
mapo_df = pd.read_csv('C:/py36/data_mapo.csv')
yangchun_df = pd.read_csv('C:/py36/data_yangchun.csv')
gangsu_df = pd.read_csv('C:/py36/data_gangsu.csv')
guro_df = pd.read_csv('C:/py36/data_guro.csv')
gumchun_df = pd.read_csv('C:/py36/data_gumchun.csv')
yungdungpo_df = pd.read_csv('C:/py36/data_yungdungpo.csv')
dongjak_df = pd.read_csv('C:/py36/data_dongjak.csv')
gwanak_df = pd.read_csv('C:/py36/data_gwanak.csv')
seocho_df = pd.read_csv('C:/py36/data_seocho.csv')
gangnam_df = pd.read_csv('C:/py36/data_gangnam.csv')
songpa_df = pd.read_csv('C:/py36/data_songpa.csv')
gangdong_df = pd.read_csv('C:/py36/data_gangdong.csv')
'''

t1=len(jongro_df)
df = jongro_df
#t1 = 1000

'''
t2=len(yongsan_df)
t3=len(sungdong_df)
t4=len(gwangjin_df)
t5=len(dongdaemun_df)
t6=len(junglang_df)
t7=len(sungbuk_df)
t8=len(gangbuk_df)
t9=len(dobong_df)
t10=len(nowon_df)
t11=len(uenpueng_df)
t12=len(sudaemun_df)
t13=len(mapo_df)
t14=len(yangchun_df)
t15=len(gangsu_df)
t16=len(guro_df)
t17=len(gumchun_df)
t18=len(yungdungpo_df)
t19=len(dongjak_df)
t20=len(gwanak_df)
t21=len(seocho_df)
t22=len(gangnam_df)
t23=len(songpa_df)
t24=len(gangdong_df)
'''

i=1

TotalPrice = 0
price1 = 500
price2 = 1000
price3 = 1500
price4 = 2000
price5 = 2500
price6 = 3000

#데이터프레임들 생성

adf = pd.DataFrame(columns = ['Item1'])
bdf = pd.DataFrame(columns = ['Item2'])
cdf = pd.DataFrame(columns = ['Item3'])
fdf = pd.DataFrame(columns = ['Item4'])
gdf = pd.DataFrame(columns = ['Item5'])
hdf = pd.DataFrame(columns = ['Item6'])
ddf = pd.DataFrame(columns = ['Price'])
edf = pd.DataFrame(columns = ['Date'])
idf = pd.DataFrame(columns = ['Items'])

item_list = ['면세탁','드라이크리닝','운동화세탁','명품크리닝','흔적제거','정장세탁','null','null','null','null','null']
first_item_list = ['면세탁','드라이크리닝','운동화세탁','명품크리닝','흔적제거','정장세탁']


#데이터 프레임들에 값 넣어주기

for i in range(t1):
    year = 2020
    month = random.randint(8, 9)
    day = random.randint(1, 28)

    adf.loc[i,'Item1'] = random.choice(first_item_list)
    bdf.loc[i,'Item2'] = random.choice(item_list)
    cdf.loc[i,'Item3'] = random.choice(item_list)
    fdf.loc[i,'Item4'] = random.choice(item_list)
    gdf.loc[i,'Item5'] = random.choice(item_list)
    hdf.loc[i,'Item6'] = random.choice(item_list)
    idf.loc[i,'Items'] = (adf.loc[i,'Item1'] +',' + bdf.loc[i,'Item2'] + ',' + cdf.loc[i,'Item3'] + ',' + fdf.loc[i,'Item4'] + ',' + gdf.loc[i,'Item5'] + ',' + hdf.loc[i,'Item6'])
    ddf.loc[i,'Price'] = 0
    edf.loc[i,'Date'] = (str(year) + '-' + str(month) + '-' + str(day))

    if adf.loc[i,'Item1'] == '면세탁':
        TotalPrice = TotalPrice + price1
    elif adf.loc[i,'Item1'] == '드라이크리닝':
        TotalPrice = TotalPrice + price2
    elif adf.loc[i,'Item1'] == '운동화세탁':
        TotalPrice = TotalPrice + price3
    elif adf.loc[i,'Item1'] == '명품크리닝':
        TotalPrice = TotalPrice + price4
    elif adf.loc[i,'Item1'] == '흔적제거':
        TotalPrice = TotalPrice + price5
    elif adf.loc[i,'Item1'] == '정장세탁':
        TotalPrice = TotalPrice + price6
    else:
        TotalPrice = TotalPrice + 0

    if bdf.loc[i,'Item2'] == '면세탁':
        TotalPrice = TotalPrice + price1
    elif bdf.loc[i,'Item2'] == '드라이크리닝':
        TotalPrice = TotalPrice + price2
    elif bdf.loc[i,'Item2'] == '운동화세탁':
        TotalPrice = TotalPrice + price3
    elif bdf.loc[i,'Item2'] == '명품크리닝':
        TotalPrice = TotalPrice + price4
    elif bdf.loc[i,'Item2'] == '흔적제거':
        TotalPrice = TotalPrice + price5
    elif bdf.loc[i,'Item2'] == '정장세탁':
        TotalPrice = TotalPrice + price6
    else:
        TotalPrice = TotalPrice + 0
    
    if cdf.loc[i,'Item3'] == '면세탁':
        TotalPrice = TotalPrice + price1
    elif cdf.loc[i,'Item3'] == '드라이크리닝':
        TotalPrice = TotalPrice + price2
    elif cdf.loc[i,'Item3'] == '운동화세탁':
        TotalPrice = TotalPrice + price3
    elif cdf.loc[i,'Item3'] == '명품크리닝':
        TotalPrice = TotalPrice + price4
    elif cdf.loc[i,'Item3'] == '흔적제거':
        TotalPrice = TotalPrice + price5
    elif cdf.loc[i,'Item3'] == '정장세탁':
        TotalPrice = TotalPrice + price6
    else:
        TotalPrice = TotalPrice + 0

    if fdf.loc[i,'Item4'] == '면세탁':
        TotalPrice = TotalPrice + price1
    elif fdf.loc[i,'Item4'] == '드라이크리닝':
        TotalPrice = TotalPrice + price2
    elif fdf.loc[i,'Item4'] == '운동화세탁':
        TotalPrice = TotalPrice + price3
    elif fdf.loc[i,'Item4'] == '명품크리닝':
        TotalPrice = TotalPrice + price4
    elif fdf.loc[i,'Item4'] == '흔적제거':
        TotalPrice = TotalPrice + price5
    elif fdf.loc[i,'Item4'] == '정장세탁':
        TotalPrice = TotalPrice + price6
    else:
        TotalPrice = TotalPrice + 0

    if gdf.loc[i,'Item5'] == '면세탁':
        TotalPrice = TotalPrice + price1
    elif gdf.loc[i,'Item5'] == '드라이크리닝':
        TotalPrice = TotalPrice + price2
    elif gdf.loc[i,'Item5'] == '운동화세탁':
        TotalPrice = TotalPrice + price3
    elif gdf.loc[i,'Item5'] == '명품크리닝':
        TotalPrice = TotalPrice + price4
    elif gdf.loc[i,'Item5'] == '흔적제거':
        TotalPrice = TotalPrice + price5
    elif gdf.loc[i,'Item5'] == '정장세탁':
        TotalPrice = TotalPrice + price6
    else:
        TotalPrice = TotalPrice + 0
    
    if hdf.loc[i,'Item6'] == '면세탁':
        TotalPrice = TotalPrice + price1
    elif hdf.loc[i,'Item6'] == '드라이크리닝':
        TotalPrice = TotalPrice + price2
    elif hdf.loc[i,'Item6'] == '운동화세탁':
        TotalPrice = TotalPrice + price3
    elif hdf.loc[i,'Item6'] == '명품크리닝':
        TotalPrice = TotalPrice + price4
    elif hdf.loc[i,'Item6'] == '흔적제거':
        TotalPrice = TotalPrice + price5
    elif hdf.loc[i,'Item6'] == '정장세탁':
        TotalPrice = TotalPrice + price6
    else:
        TotalPrice = TotalPrice + 0

    ddf.loc[i,'Price'] = TotalPrice
    TotalPrice = 0

    if i%1000 == 1:
        print(i,",")
    else:
        continue

#데이터프레임들을 붙이기

df = pd.concat([df,adf], axis=1)
df = pd.concat([df,bdf], axis=1)
df = pd.concat([df,cdf], axis=1)
df = pd.concat([df,fdf], axis=1)
df = pd.concat([df,gdf], axis=1)
df = pd.concat([df,hdf], axis=1)
df = pd.concat([df,idf], axis=1)
df = pd.concat([df,ddf], axis=1)
df = pd.concat([df,edf], axis=1)

df.index = jongro_df['사업장명']            #인덱스에 사업장명을 넣기
final_df = df.drop(df.columns[[0]],axis='columns')      #인덱스에 사업장명을 넣었기 때문에 열에 있는 사업장명을 제거

final_df.to_csv('C:/py36/data_jongro_2.csv',encoding='utf-8-sig')