from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse
from django.http import JsonResponse

from search.models import Paper

import nltk
from nltk.corpus import stopwords
import numpy as np
import math
import re
from sklearn.cluster import KMeans


def prepro(document):
    """文書前処理"""
    tmp = document.replace('.', '').replace(',', '').replace('\"', '').replace('\'', '').replace('¥n', '') #句読点や改行などを削除
    tmp = re.sub(re.compile("[!-/:-@[-`{-~]"), '', tmp) #半角記号を削除
    tmp = tmp.split(" ") # スペースを区切りにリスト化
    return tmp


def removeStoplist(document):
  """ストップワードを取り除く"""
  stoplist_removed_document = []
  for word in document:
      if word not in stopwords.words('english'):
        stoplist_removed_document.append(word)
  return stoplist_removed_document


def stemming(document):
    """ステミング処理"""
    stemmer = nltk.PorterStemmer()
    stem = []
    for word in document:
        stem.append(stemmer.stem(word))
    return stem


def make_term(documents):
    """全文書による単語リストを作成"""
    all_term=[]
    for i in documents:
        for j in i:
            if j not in all_term:
                all_term.append(j)
    return(all_term)


def tf(terms, document):
    """任意の文書のTF値の計算"""
    tf_values = [document.count(term) for term in terms]
    return list(map(lambda x: x/sum(tf_values), tf_values))


def idf(terms, documents):
    """任意の文書のIDF値の計算"""
    return list(map(lambda x : x+1, [math.log10(len(documents)/sum([bool(term in document) for document in documents])) for term in terms]))


def tf_idf(terms, documents):
    """文章毎にTF-IDF値を計算"""
    return [[_tf*_idf for _tf, _idf in zip(tf(terms, document), idf(terms, documents))] for document in documents]


def center(doc_vector, k, pred):
    """クラスタの重心ベクトル算出"""
    center_vec = np.zeros((k,len(doc_vector[0]))) # 重心ベクトルの初期値(クラスタの数*ベクトルの次元数)
    for i in range(k):
        counter = 0
        for j in range(len(doc_vector)):
            if i == pred[j]:
                center_vec[i] += doc_vector[j]
                counter += 1
        center_vec[i] = center_vec[i] / counter
    return center_vec


def clst(doc_vector):
    """文章ベクトルのクラスタリング処理"""
    k = 3
    pred = KMeans(n_clusters=k,random_state=56).fit_predict(doc_vector)
    center_vec = center(doc_vector,k,pred)
    return pred, center_vec


def cos_sim(x,y):
    """コサイン類似度の計算"""
    if np.linalg.norm(x)==0 or np.linalg.norm(y)==0:
        return 0
    else:
        return np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))


# 行に文書,列に単語のリストを作成
doc = []
# for i in range(20): # file_number
for i in Paper.objects.all():
    tmp = stemming(removeStoplist(prepro(i.abst)))
    doc.append(tmp)

#全出現単語
all_terms = make_term(doc)

# tf-idfを計算し文書ベクトルへ(行に各文書ベクトル、列はベクトルの各次元の要素)
doc_vec = tf_idf(all_terms,doc)

#文書ベクトルをクラスタリングし、クラスタの重心ベクトルを算出
clst_pred, center_vector = clst(doc_vec)
print(clst_pred)


def Index(request):
    """検索結果"""

    result_title = []
    result_abst = []
    len_abst = []
    p_id = []
    search_flag = '0'

    if request.method == 'GET':

        query = request.GET.get('query') #検索ワード：query

        if query:
            query_tmp = stemming(removeStoplist(prepro(query)))

            # クエリのベクトル生成
            query_vec = np.zeros(len(all_terms))
            for i in query_tmp:
                if i in all_terms:
                    query_vec[all_terms.index(i)] = 1

            #クエリベクトルがどのクラスタの重心ベクトルと類似しているかを計算
            near_clst_sim = 0
            for i in range(len(center_vector)):
                clst_sim = cos_sim(query_vec,center_vector[i])
                if near_clst_sim <= clst_sim :
                    near_clst_sim = clst_sim
                    clst_number = i

            # クエリベクトルと類似クラスタ内の各文書ベクトルとのコサイン類似度を算出し、
            # 文書ベクトルと対応するリストの位置に類似度の値を代入
            sim_array = np.zeros(len(doc_vec))
            for i in range(len(doc_vec)):
                if clst_number == clst_pred[i]:
                    sim_array[i] = cos_sim(query_vec,doc_vec[i])

            # 検索ワードとの類似度が高い順にソート(ランキング化)し、その元のindexをリスト化
            dec_doc=[]
            for i in range(len(sim_array)):
                if np.sort(sim_array)[::-1][i] != 0:
                    dec_doc += [np.argsort(sim_array)[::-1][i]]

            # ランキングが高い順にリストに追加
            for i in dec_doc:
                 for j in Paper.objects.all():
                     if (i+1) == j.id:
                         result_title += [j.title]
                         result_abst += [j.abst[:200]]
                         len_abst += [len(j.abst)]
                         p_id += [str(j.id)]

            search_flag = '1'

    d = {
        'query':query,
        'search_flag':search_flag,
        'result_title':result_title,
        'result_abst':result_abst,
        'len_abst':len_abst,
        'p_id':p_id,
        'result_num':str(len(result_abst)),
    }
    return render(request, 'search/index.html',d)


def Content(request,paper_id):

    cont = get_object_or_404(Paper, pk=paper_id)
    contexts = {
        'title':cont.title,
        'abst':cont.abst,
    }
    return render(request,'search/content.html',contexts)
