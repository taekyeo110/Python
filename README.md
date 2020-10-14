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

