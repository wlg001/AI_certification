import gensim
from gensim.models import Word2Vec
import jieba
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
from transformers import BertConfig
from gensim.models import KeyedVectors

model = gensim.models.Word2Vec.load('words.model')
result = model.wv.n_similarity('轨道炮', '轰20')
print(result)