# %%
from gensim.models import KeyedVectors

# w2v_file = '../../dataset/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'
# #载入词向量
# w2v_model = KeyedVectors.load_word2vec_format(w2v_file, binary=False)

# 直接加载w2v模型
w2v_model = KeyedVectors.load('../data/word2vec.model')

# %%
