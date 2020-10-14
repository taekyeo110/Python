# Textrank



TextRank 는 키워드 추출 기능과 핵심 문장 추출 기능, 두 가지를 제공합니다. 

키워드를 추출하기 위해서 먼저 단어 그래프를 만들어야 합니다. 
마디인 단어는 주어진 문서 집합에서 최소 빈도수 min_count 이상 등장한 단어들 입니다. 
sents 는 list of str 형식의 문장들이며, tokenize 는 str 형식의 문장을 list of str 형식의 단어열로 나누는 토크나이저 입니다.


```
from collections import Counter

def scan_vocabulary(sents, tokenize, min_count=2):
  counter = Counter(w for sent in sents for w in tokenize(sent))
  counter = {w:c for w,c in counter.items() if c >= min_count}
  idx_to_vocab = [w for w, _ in sorted(counter.items(), key=lambda x:-x[1])]
  vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
  return idx_to_vocab, vocab_to_idx
```
    
  
TextRank 에서 두 단어 간의 유사도를 정의하기 위해서는 두 단어의 co-occurrence 를 계산해야 합니다. 
Co-occurrence 는 문장 내에서 두 단어의 간격이 window 인 횟수입니다. 논문에서는 2 ~ 8 사이의 값을 이용하기를 추천하였습니다. 
여기에 하나 더하여, 문장 내에 함께 등장한 모든 경우를 co-occurrence 로 정의하기 위하여 window 에 -1 을 입력할 수 있도록 합니다. 
또한 그래프가 지나치게 dense 해지는 것을 방지하고 싶다면 min_coocurrence 를 이용하여 그래프를 sparse 하게 만들 수도 있습니다.

```
from collections import defaultdict

def cooccurrence(tokens, vocab_to_idx, window=2, min_cooccurrence=2):
    counter = defaultdict(int)
    for s, tokens_i in enumerate(tokens):
        vocabs = [vocab_to_idx[w] for w in tokens_i if w in vocab_to_idx]
        n = len(vocabs)
        for i, v in enumerate(vocabs):
            if window <= 0:
                b, e = 0, n
            else:
                b = max(0, i - window)
                e = min(i + window, n)
            for j in range(b, e):
                if i == j:
                    continue
                counter[(v, vocabs[j])] += 1
                counter[(vocabs[j], v)] += 1
    counter = {k:v for k,v in counter.items() if v >= min_cooccurrence}
    n_vocabs = len(vocab_to_idx)
    return dict_to_mat(counter, n_vocabs, n_vocabs)
```

dict_to_mat 함수는 dict of dict 형식의 그래프를 scipy 의 sparse matrix 로 변환하는 함수입니다.

```
from scipy.sparse import csr_matrix

def dict_to_mat(d, n_rows, n_cols):
    rows, cols, data = [], [], []
    for (i, j), v in d.items():
        rows.append(i)
        cols.append(j)
        data.append(v)
    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
```

TextRank 에서는 명사, 동사, 형용사와 같은 단어만 단어 그래프를 만드는데 이용합니다. 
모든 종류의 단어를 이용하면 ‘a’, ‘the’ 와 같은 단어들이 다른 단어들과 압도적인 co-occurrence 를 지니기 때문입니다. 
즉, stopwords 를 지정할 필요가 있다면 지정하여 키워드 후보만 그래프에 남겨둬야 한다는 의미입니다. 
그러므로 입력되는 tokenize 함수는 불필요한 단어를 모두 걸러내고, 필요한 단어 혹은 품사만 return 하는 함수이어야 합니다.

이 과정을 정리하면 아래와 같은 word_graph 함수를 만들 수 있습니다.

```
def word_graph(sents, tokenize=None, min_count=2, window=2, min_cooccurrence=2):
    idx_to_vocab, vocab_to_idx = scan_vocabulary(sents, tokenize, min_count)
    tokens = [tokenize(sent) for sent in sents]
    g = cooccurrence(tokens, vocab_to_idx, window, min_cooccurrence, verbose)
    return g, idx_to_vocab
```

그 뒤 만들어진 그래프에 PageRank 를 학습하는 함수를 만듭니다. 
입력되는 x 는 co-occurrence 그래프일 수 있으니, column sum 이 1 이 되도록 L1 normalization 을 합니다. 이를 A 라 합니다. 
A * R 은 column j에서 row i로의 랭킹 R_j의 전달되는 값을 의미합니다. 
이 값에 df 를 곱하고, 모든 마디에 1 - df 를 더합니다. 이를 max_iter 만큼 반복합니다.

```
import numpy as np
from sklearn.preprocessing import normalize

def pagerank(x, df=0.85, max_iter=30):
    assert 0 < df < 1

    # initialize
    A = normalize(x, axis=0, norm='l1')
    R = np.ones(A.shape[0]).reshape(-1,1)
    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)

    # iteration
    for _ in range(max_iter):
        R = df * (A * R) + bias

    return R
```

이 과정을 정리하면 아래와 같은 textrank_keyword 함수를 만들 수 있습니다.

```
def textrank_keyword(sents, tokenize, min_count, window, min_cooccurrence, df=0.85, max_iter=30, topk=30):
    g, idx_to_vocab = word_graph(sents, tokenize, min_count, window, min_cooccurrence)
    R = pagerank(g, df, max_iter).reshape(-1)
    idxs = R.argsort()[-topk:]
    keywords = [(idx_to_vocab[idx], R[idx]) for idx in reversed(idxs)]
    return keywords
```



##	적용

크롤링으로 수집한 인크레더블 영화 댓글 10267개의 keyword를 구합니다.

크롤링한 10267개의 댓글입니다.
```
	|num|	ID|	review|	score
0|	1|	yski****|	 잭잭이랑 에드나 케미 미쳤닼ㅋㅋㅋ| 	10
1|	2|	배센도(bbtj****)|	 어릴 때 1편을 보고 성인이 된 올해 2편을 봤다  또 보고 싶다  3편도 나오면 좋겠다|  	10
2|  3|	space(tmd5****)|	 속편도 이리 완벽할 수 있구나 |   	10
3|	4|	황진이의두번째팬티(sion****)|	 중심히어로와 빌런이 여성이라는 점 그 둘의 대화 내용 아내에게 열등감을 느끼던 남편이 육아를 도맡고 아내의 바깥일을 내조하며 그녀를 진짜 히어로로 인정하는과정이 인상 깊었다  꿈이 많은 내게 선물같은 영화였다|   	10
4|	5|	불(catc****)|	 이런 게 최고의 애니매이션이 아니면 뭐란 말인가   미취학아동 때 1을 보고 대학생이 되어서 2를 보는 기분이란   ㅠㅠㅠ헬렌의 활약과 잭잭의 귀여움 그리고 개인적으로는 에드나의 매력까지 올해 본 영화 중 최고|     	10
5|	6|	쿠앤크(zhfl****)|	 잭잭 납치하러 갈 파티원 구합니다 1 10000 | 	8
6|	7|	아머두어라두(dkqj****)|	 관람객내 어릴적 베스트 영화의 속편이 너무 잘만들어져서 울컥했습니다|  	10
7|	8|	일산빵셔틀(rnra****)|	 평론가 임수연씨의 마블보다 재밌다는 평을보고 코웃음치고 보러갔는데 마블빠인 내가봐도 마블보다 재밌었다 히어로물은 언제까지나 마블의 독주일꺼라는 착각을 씻어내준 가족액션 히어로물 진짜 너무재밌다| 	10
8|	9|	물짱이(pres****)|	 관람객꼭 보세요 개 쩜  일라스티걸이 미스터인크레더블의 그늘에서 벗어나 활약하는 것도 멋지지만 미스터 인크레더블이 바뀐 역할을 얕보지 않고 가족을 위해서 열심히 노력하는 모습도 멋집니다  가족 구성원의 어떤 역할이든 중요|   	10
9|	10|	임태준(kota****)|	 일라스틱걸 사랑해요 ㅜㅜ| 	10
......

```
문장들의 가중치를 계산할 때, '~가, ~도'와 같은 불필요하게 가중치가 높아질 수 있는 단어들은 stopword라는 변수로 제외시킵니다.
아직 세탁소 댓글을 다 만들지 못해 stopword는 지정하지 않았습니다.
위 데이터에서 가중치가 높은 sents와 keyword를 뽑아 csv파일로 저장합니다.
```
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
```
추출한 keyword입니다. 명사만 뽑을 수 있지만 혹시 몰라 형용사와 동사도 뽑았습니다. 차후에 세탁소 댓글이 완성되면 조정할 계획입니다.
```
--------|keywords|weights
--------|--------|----------
0|보/VV|36.62739196
1|관람객/NNG|30.84064907
2|잭/NNP|21.10252649
3|재밌/VA|17.58688637
4|편/NNB|14.03999885
5|영화/NNG|10.25931933
6|나오/VV|8.003162485
7|좋/VA|7.613481931
8|인크레더블/NNP|7.292621322
9|있/VV|7.158753249
```



References
Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank citation ranking: Bringing order to the web. Stanford InfoLab
Mihalcea, R., & Tarau, P. (2004). Textrank: Bringing order into text. In Proceedings of the 2004 conference on empirical methods in natural language processing
Erkan, G., & Radev, D. R. (2004). Lexrank: Graph-based lexical centrality as salience in text summarization. Journal of Artificial Intelligence Research, 22, 457-479
Barrios, F., López, F., Argerich, L., & Wachenchauzer, R. (2016). Variations of the similarity function of textrank for automated summarization. arXiv preprint arXiv:1602.03606.
