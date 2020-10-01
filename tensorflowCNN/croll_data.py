import requests
from bs4 import BeautifulSoup
import sys
import string
import pandas as pd
import numpy as np
import csv

df = pd.read_csv("C:/py36/example2.csv" , encoding = 'cp949')

df2 = df.replace('\*',' ')

df2.to_csv("C:/py36/example3.csv", encoding = 'utf-8-sig' ,header=False)