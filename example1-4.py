from gensim.models import KeyedVectors
w2v_model = KeyedVectors.load('small_weibo_vectors.txt', mmap='r')