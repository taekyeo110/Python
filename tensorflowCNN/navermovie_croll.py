import requests
from bs4 import BeautifulSoup
import sys
import string
import pandas as pd
import numpy as np
import csv

sys.stdout = open('C:/py36/example.csv','w',newline="")

df = pd.DataFrame(columns=['nickname','review','score'])

f=0


def get_data(url):
    resp = requests.get(url)
    html = BeautifulSoup(resp.content, 'html.parser')
    score_result = html.find('div', {'class': 'score_result'})
    lis = score_result.findAll('li')

    for li in lis:
        befornickname = li.find('dl')
        nickname = befornickname.findAll('a')[0].find('span').getText()
        review_text = li.find('p').getText()
        score = li.find('em').getText()
        nickname2 = str(nickname).replace('\n','').replace('\r','').replace('\t','').replace(',','') + ','
        review_text2 = str(review_text).replace('\n','').replace('\r','').replace('\t','').replace(',','') + ','

        row = [nickname2,review_text2,score]

        global df
        df = df.append(pd.Series(row, index=df.columns),ignore_index=True)

        print(nickname2, review_text2, end="")
        print(score)

#url 연결하기
test_url = 'https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=136990&type=after'
resp = requests.get(test_url)
html = BeautifulSoup(resp.content, 'html.parser')
result = html.find('div', {'class':'score_total'}).find('strong').findChildren('em')[0].getText()
total_count = int(result.replace(',', ''))
        
for i in range(1, int(total_count / 10) + 1):
    url = test_url + '&page=' + str(i)
    #print('url: "' + url + '" is parsing....')
    get_data(url)

#df.set_index('nickname',inplace=True)
#pd.set_option('display.max_rows',len(df))
#print(df)

#df.to_csv("C:/py36/example.csv", mode='a', header=False)