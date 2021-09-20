import matplotlib.pyplot
import numpy
from gensim.models import word2vec
import numpy as np

'''生成长序列串的子序列集，k为kmer长度'''
def k_mergenerator(WaiTingKmer, k):
    result = open("result.txt", mode="w")
    DNA = []
    for i in WaiTingKmer:
        line = []
        for j in range(len(i)-k+1):
            line.append(i[j:j+k])
            result.write(i[j:j+k])
            result.write(" ")
        DNA.append(line)
        result.write("\n")
    return DNA

'''生成整条基因数据的向量，生成方式为将每个kmer向量相加'''
def DNA_to_array(DNA, module):
    DNA_array = []
    sum = 0
    for i in DNA:
        for j in i:
            sum = sum + module.wv[j]
        DNA_array.append(sum)
    return DNA_array


if __name__ == '__main__':
    splice = open("splice.data")
    lable = []
    DNA = []
    for i in splice:
        lable.append(i.split(",")[0])
        DNA.append(i.split(",")[2].replace(" ", "").replace("\n", ""))
    splice.close()
    # 获取Kmer
    DNAkmer = k_mergenerator(DNA, 3)
    # 使用word2vec出基因数据
    genes = word2vec.Text8Corpus("result.txt")
    model = word2vec.Word2Vec(genes, sg=1, min_count=1, vector_size=100)
    print(model)

    model.wv.save_word2vec_format("VectorForGene", binary=False)

    """转化成其他模块能用向量数据"""
    # word_list = model.wv.index_to_key
    # # print(word_list)
    # embeddings = np.array([model.wv[word] for word in word_list])
    # print(embeddings)
    """把DNA文件转化为向量（还是列表）"""
    lable = np.array(lable)
    DNAarray = DNA_to_array(DNAkmer, model)

    '''试试看能不能用kmeans'''
    # # print(npemd)
    # from sklearn.cluster import k_means
    # fuck = k_means(DNAarray,n_clusters=3)
    # result = numpy.array(fuck[1])
    # print(result)
    # fu = []
    # for cluster in range(3):
    #     fu.append(list(lable[np.where(fuck[1] == cluster)]))
    # print("第一类 EI",fu[0].count("EI")," IE ",fu[0].count("IE")," N ",fu[0].count("N"),)
    # print("第二类 EI",fu[1].count("EI")," IE ",fu[1].count("IE")," N ",fu[1].count("N"),)
    # print("第三类 EI",fu[2].count("EI")," IE ",fu[2].count("IE")," N ",fu[2].count("N"),)
    # """kmeans 结果大概78左右"""

    '''试试看能不能svm'''
    DNAarray = np.array(DNAarray)
    print(DNAarray.shape)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(np.array(DNAarray),np.array(lable),test_size=0.2)
    from sklearn.svm import SVC
    svmclass = SVC(kernel="rbf")
    print(len(X_train),len(y_train))
    svmclass.fit(X_train,y_train)
    res = svmclass.predict(X_test)
    import sklearn.metrics as sm
    bg = sm.classification_report(y_test, res)
    SB = sm.confusion_matrix(y_test,res)
    print(SB)
    sm.ConfusionMatrixDisplay
    # for i in range(len(y_test)):
    #     print(y_test[i],res[i])
    print('分类报告：', bg, sep='\n')
