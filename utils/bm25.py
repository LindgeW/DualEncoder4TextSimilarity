import numpy as np
from collections import Counter
# 和TF-IDF一样，最好先去掉停用词


class BM25(object):
    def __init__(self, docs):
        self.D = len(docs)
        self.avgdl = sum([len(doc)+0.0 for doc in docs]) / self.D
        self.docs = docs
        self.f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {}  # 存储每个词及出现了该词的文档数量
        self.idf = {}  # 存储每个词的idf值
        self.k1 = 1.5  # 调节特征词文本频率尺度的作用 [1.2, 2]
        self.b = 0.75  # 调整文档长度对相关性影响的大小
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = np.log(1 + (self.D - v + 0.5) / (v + 0.5))
            # self.idf[k] = np.log(self.D + 1) - np.log(v + 0.5)

    def sim(self, doc, index):
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word] * (self.f[index][word] * (self.k1 + 1) / (self.f[index][word] + self.k1 * (1 - self.b + self.b * d / self.avgdl)) + 0.001))
        return score

    def simall(self, doc):
        scores = []
        for index in range(self.D):
            score = self.sim(doc, index)
            scores.append(score)
        return scores


class BM25Model(object):
    def __init__(self, documents_list, k1=1.5, k2=1, b=0.75):
        # 文本列表，内部每个文本需要事先分好词
        self.documents_list = documents_list
        # 文本总个数
        self.documents_number = len(documents_list)
        # 文本库中文本的平均长度
        self.avg_documents_len = sum([len(document) for document in documents_list]) / self.documents_number
        # 存储每个文本中每个词的词频
        self.f = []
        # 存储每个词汇的逆文档频率
        self.idf = {}
        self.k1 = k1  # [1.2, 2]
        self.k2 = k2
        self.b = b  # 0.75
        # 类初始化
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                # 存储每个文档中每个词的词频
                temp[word] = temp.get(word, 0) + 1
            self.f.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            # 每个词的逆文档频率
            self.idf[key] = np.log(1 + (self.documents_number - value + 0.5) / (value + 0.5))
            # self.idf[key] = np.log((self.documents_number + 1) / (value + 0.5))

    def get_score(self, index, query):
        score = 0.0
        document_len = len(self.f[index])
        qf = Counter(query)
        for q in query:
            if q not in self.f[index]:
                continue
            # score += self.idf[q] * ((self.f[index][q] * (self.k1 + 1) / (self.f[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) * (qf[q] * (self.k2 + 1) / (qf[q] + self.k2)) + 0.001)
            score += self.idf[q] * ((self.f[index][q] * (self.k1 + 1) / (self.f[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) + 0.001)
        return score

    def get_documents_score(self, query):
        score_list = []
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query))
        return score_list


if __name__ == '__main__':
    bm25 = BM25Model(['NLP是一门比较有意思的研究学科',
                      'NLP是一门比较有趣的研究学科',
                      'NLP是一门很有意思的研究方向'])
    s1 = bm25.get_documents_score('它是一门有意思的研究学科')
    s2 = bm25.get_documents_score('NLP是一个有意思的研究方向。，')
    print('BM25 Testing1:', s1, s2)
    bm25 = BM25(['NLP是一门比较有意思的研究学科',
                 'NLP是一门比较有趣的研究学科',
                 'NLP是一门很有意思的研究方向'])
    s3 = bm25.simall('NLP是一个有意思的研究热点')
    s4 = bm25.simall('这是一个有意思的研究方向')
    print('BM25 Testing2:', s3, s4)