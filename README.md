# Textrank



TextRank 는 키워드 추출 기능과 핵심 문장 추출 기능, 두 가지를 제공합니다. 

키워드를 추출하기 위해서 먼저 단어 그래프를 만들어야 합니다. 
마디인 단어는 주어진 문서 집합에서 최소 빈도수 min_count 이상 등장한 단어들 입니다. 
sents 는 list of str 형식의 문장들이며, tokenize 는 str 형식의 문장을 list of str 형식의 단어열로 나누는 토크나이저 입니다.

'''C

  from collections import Counter

  def scan_vocabulary(sents, tokenize, min_count=2):
    counter = Counter(w for sent in sents for w in tokenize(sent))
    counter = {w:c for w,c in counter.items() if c >= min_count}
    idx_to_vocab = [w for w, _ in sorted(counter.items(), key=lambda x:-x[1])]
    vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
    return idx_to_vocab, vocab_to_idx
    
 '''
    
TextRank 에서 두 단어 간의 유사도를 정의하기 위해서는 두 단어의 co-occurrence 를 계산해야 합니다. 
Co-occurrence 는 문장 내에서 두 단어의 간격이 window 인 횟수입니다. 논문에서는 2 ~ 8 사이의 값을 이용하기를 추천하였습니다. 
여기에 하나 더하여, 문장 내에 함께 등장한 모든 경우를 co-occurrence 로 정의하기 위하여 window 에 -1 을 입력할 수 있도록 합니다. 
또한 그래프가 지나치게 dense 해지는 것을 방지하고 싶다면 min_coocurrence 를 이용하여 그래프를 sparse 하게 만들 수도 있습니다.

