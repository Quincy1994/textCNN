# coding=utf-8

import multiprocessing  # 多任务调度包
from gensim.models import word2vec
import data_helpers

FILE_TO_LOAD = '../model/w2vmodel/w2v.model'
FILE_TO_SAVE = '../model/w2vmodel/w2v.model'

def embedding_sentences(sentences , embedding_size=128, window=5, min_count =5, file_to_load = None, file_to_save=None):

    # 加载或训练word2vec模型
    if file_to_load is not None:
        w2vModel = word2vec.Word2Vec.load(file_to_load)
    else:
        w2vModel = word2vec.Word2Vec(sentences=sentences, size= embedding_size, window= window, min_count= min_count, workers= multiprocessing.cpu_count())
        if file_to_save is not None:
            w2vModel.save(file_to_save)

    # 将句子转化为word2vec矩阵
    all_vectors = []
    embedding_dim = w2vModel.vector_size
    embedding_unknown = [0 for i in range(embedding_dim)]  ## 未登录词做0填充
    for sentence in sentences:
        this_vector = []
        for word in sentence:
            if word in w2vModel.wv.vocab:
                this_vector.append(w2vModel[word])
            else:
                this_vector.append(embedding_unknown)
        all_vectors.append(this_vector)
    return all_vectors

# 训练word2vec语料
def train_w2vmodel():
    sentences = word2vec.Text8Corpus('/home/iiip/Pictures/text8')
    embedding_sentences(sentences,embedding_size=128, window=5,min_count=5, file_to_save=FILE_TO_SAVE)
    print 'w2vmodel is ok!'
    # print x_text[0]

# train_w2vmodel()