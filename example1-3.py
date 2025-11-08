from gensim.models import KeyedVectors
import os

def create_small_model(input_file, output_file, vocab_size=40000):
    print(f"正在从{input_file}创建小型模型...")
    wv = KeyedVectors.load_word2vec_format(input_file, binary=False)
    # 获取词频排序后的前N个词
    words_to_keep = sorted(wv.key_to_index.keys(), 
                         key=lambda w: wv.get_vecattr(w, 'count'), 
                         reverse=True)[:vocab_size]
    # 创建新的词向量，只包含选定的词
    limited_wv = KeyedVectors(wv.vector_size)
    for word in words_to_keep:
        limited_wv.add_vector(word, wv[word])
    limited_wv.fill_norms()
    limited_wv.save_word2vec_format(output_file, binary=True)
    print(f"已生成小型模型：{output_file}")
    return limited_wv

# 文件路径
small_model_path = 'small_weibo_vectors.bin'
full_model_path = 'sgns.weibo.word'

# 加载或创建模型
if os.path.exists(small_model_path):
    print("找到小型模型，正在加载...")
    wv = KeyedVectors.load_word2vec_format(small_model_path, binary=True)
elif os.path.exists(full_model_path):
    print("未找到小型模型，但找到完整模型，将创建小型模型...")
    wv = create_small_model(full_model_path, small_model_path)
else:
    raise FileNotFoundError("既找不到小型模型也找不到完整模型，请确保至少有一个模型文件存在")

# 测试模型
similarity = wv.similarity('浙江', '北京')
print(f"相似度：浙江 和 北京 = {similarity}")

# 可以添加更多的测试
similar_words = wv.most_similar('浙江', topn=5)
print("\n与'浙江'最相似的词：")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")

# 计算两个词组的相似度
phrase1 = ['浙江', '台州']
phrase2 = ['湖北', '武汉']
sim2 = wv.n_similarity(phrase1, phrase2)
print(f"\n词组相似度：{' '.join(phrase1)} 和 {' '.join(phrase2)} = {sim2}") 

# 找出最不相似的词
least_similar = wv.most_similar(negative=['浙江'], topn=5)
print("\n与'浙江'最不相似的词：")
for word, score in least_similar:
    print(f"{word}: {score:.4f}")

# 找出给定词组中最不相似的词
words = ['浙江', '北京', '上海', '广州', '深圳', '西班牙']
outlier = wv.doesnt_match(words)
print(f"\n在词组 {words} 中，最不相似的词是：{outlier}")

# 计算词向量的加减
result = wv.most_similar(positive=['北京', '浙江'], negative=['上海'], topn=5)
print("\n词向量计算结果（北京 + 浙江 - 上海）：")
for word, score in result:
    print(f"{word}: {score:.4f}")   