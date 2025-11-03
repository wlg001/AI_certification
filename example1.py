import gensim
from gensim.models import Word2Vec
import jieba

datas = open('text.txt', 'r', encoding='utf-8').read().split('\n')

words_datas = [[i for i in jieba.cut(data) if i != ' ' ] for data in datas]
print(words_datas)

model = Word2Vec(words_datas, vector_size=10, window=2, min_count=1, workers=8, sg=0, epochs=10)

model.wv.save_word2vec_format('words_data.vector', binary=False)  # 保存为文本格式
model.save('words.model')


