from konlpy.tag import Komoran
from summarizer import KeywordSummarizer
from summarizer import KeysentenceSummarizer
import pandas as pd
import string
import numpy as np
import sys

komoran = Komoran()

a = pd.read_csv('C:/py36/review_emotion.csv', encoding='utf-8')

'''
a.columns=['num','ID','review','score']
a['review'] = a['review'].str.replace(pat='[^\w\s]', repl= ' ')  # replace all special symbols to space
a['review'] = a['review'].str.replace(pat='[\s\s+]', repl= ' ', regex=True)  # replace multiple spaces with a single space
'''
sentss = list(np.array(a['review'].tolist()))
sents = []

for i in range(1000):
    sents.insert(i,sentss[i])

def komoran_tokenize(sent):
    words = komoran.pos(sent, join=True)
    words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
    return words

keyword_extractor = KeywordSummarizer(
    tokenize = komoran_tokenize,
    window = -1,
    verbose = False
)
keywords = keyword_extractor.summarize(sents, topk=10)

summarizer = KeysentenceSummarizer(tokenize = komoran_tokenize, min_sim = 0.5)
keysents = summarizer.summarize(sents, topk=10)

keywords2 = pd.DataFrame(keywords)
keywords2.columns = ["keywords","weights"]

keywords2.to_csv('C:/py36/review_emotion2.csv',encoding='utf-8-sig')