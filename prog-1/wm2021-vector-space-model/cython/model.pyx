import os
import math
import time
import argparse
import xml.etree.ElementTree as ET
from functools import reduce
from itertools import compress, repeat
import numpy as np
cimport numpy as np
from tqdm import tqdm
# import matplotlib.pyplot as plt

alpha = 3
beta = 1
gamma = 0
r = 6
ir = 0
g = 100
k1 = 2 # 1-2
k3 = 100 # 0-1000
b = 0.75 # 0.75

cdef class Retriever:
    cdef dict __dict__
    def __init__(self, relevance, input_path, output_path, model_path, dir_path):
        self.eps = 1e-12
        self.relevance = relevance
        self.output_path = output_path
        self.build_vocab(model_path) # Must be called first
        self.build_doc_list(model_path, dir_path)
        self.build_stopword_list('model/stopwords')
        self.build_tfidf(model_path) # Must be called after build_doc_list
        self.build_query_list(input_path)
        self.build_query_tfidf() # Must be called after build_query_list & build_tfidf
        self.get_query_score()
        
    
    def build_vocab(self, model_path):
        self.bi_vocab_dict = {}
        with open(os.path.join(model_path, "vocab.all")) as f:
            vocab_list = f.readlines()
            self.uni_vocab_dict = {k[:-1]: i for i,k in enumerate(vocab_list)}
        self.vocab_size = len(self.uni_vocab_dict)

    def tokenize(self, text):
        index_list = []
        english_vocab = ""
        for i, c in enumerate(text):
            if c.encode().isalnum():
                english_vocab = english_vocab + c
            else:
                if not english_vocab:
                    index_list.append(self.uni_vocab_dict.get(c, 0))
                else:
                    index_list.append(self.uni_vocab_dict.get(english_vocab, 0))
                    english_vocab = ""
        return index_list


    def build_query_list(self, input_path):
        input_tree = ET.parse(input_path)
        self.query_list = []
        self.query_id_list = []
        for topic in input_tree.getroot().findall('topic'):
            number = topic.find('number').text[-3:]

            # text =  topic.find('title').text.strip() + \
            #         topic.find('question').text.strip() + \
            #         topic.find('narrative').text.strip() + \
            #         topic.find('concepts').text.strip()
            text =  topic.find('title').text.strip() + \
                    topic.find('concepts').text.strip()

            self.query_list.append(self.tokenize(text))
            self.query_id_list.append(number)
        self.query_size = len(self.query_list)

    def build_doc_list(self, model_path, dir_path):
        print("Building document list...")
        with open(os.path.join(model_path, 'file-list')) as f:
            file_list = f.readlines()
        # self.doc_size = len(file_list)
        self.doc_list = []
        self.doc_id_list = []
        self.doc_title_ids = []
        self.doc_title_tfs = []
        doc_length = []
        for i, file_path in enumerate(tqdm(file_list)):
            # doc_tree = ET.parse(os.path.join(dir_path, file_path.strip()))
            # doc = doc_tree.getroot().find('doc')
            # self.doc_id_list.append(doc.find('id').text.lower())
            self.doc_id_list.append(file_path.strip().split('/')[-1].lower())
        self.doc_size = len(self.doc_id_list)
        # self.doc_length = np.array(doc_length)
        print("Document list built!")

    def build_stopword_list(self, stopword_path):
        self.stopwords = [0]
        with open(stopword_path) as f:
            stopwords = [w.strip() for w in f.readlines()]
        for w in stopwords:
            if w in self.uni_vocab_dict:
                self.stopwords.append(self.uni_vocab_dict[w])

    def build_tfidf(self, model_path):
        print("Building tfidf...")
        self.doc_ids = [[] for _ in range(self.doc_size)]
        self.doc_tfs = [[] for _ in range(self.doc_size)]
        self.doc_df = [0] * self.vocab_size
        with open(os.path.join(model_path, 'inverted-file')) as f:
            file_list = f.readlines()
        # inverted_file = open(os.path.join(model_path, 'inverted-file'))
        vocab = 0
        skip = False
        for line in tqdm(file_list):
            line_list = line.strip().split()
            if len(line_list) == 3:
                if line_list[1] == "-1":
                    vocab = int(line_list[0])
                    self.doc_df[vocab] = int(line_list[2])
                    if vocab in self.stopwords:
                        skip = True
                    else:
                        skip = False
                else:
                    self.bi_vocab_dict[(int(line_list[0]), int(line_list[1]))] = self.vocab_size
                    vocab = self.vocab_size
                    self.doc_df.append( int(line_list[2]) )
                    self.vocab_size += 1
                    skip = False
            else:
                if skip:
                    continue
                self.doc_ids[int(line_list[0])].append(vocab)
                self.doc_tfs[int(line_list[0])].append( int(line_list[1]) )

        print("Transforming into numpy array...")
        self.doc_length = []
        for i in tqdm(range(self.doc_size)):
            self.doc_ids[i] = np.array(self.doc_ids[i])
            self.doc_tfs[i] = np.array(self.doc_tfs[i])
            p = np.argsort(self.doc_ids[i])
            self.doc_ids[i] = self.doc_ids[i][p]
            self.doc_tfs[i] = self.doc_tfs[i][p]
            length = self.doc_tfs[i].sum()
            if length == 0:
                self.doc_ids[i] = np.array([0])
                self.doc_tfs[i] = np.array([1])
                length = 1
            self.doc_length.append(length)
        self.doc_length = np.array(self.doc_length)
        self.avdl = self.doc_length.mean()
        self.doc_df = np.array(self.doc_df)
        print("Tfidf built!")

    def add_numpy_bow(self, ids1, tfs1, ids2, tfs2):
        union_ids = np.union1d(ids1, ids2)
        union_tfs = np.zeros(len(union_ids))
        union_tfs[np.in1d(union_ids, ids1)] += tfs1
        union_tfs[np.in1d(union_ids, ids2)] += tfs2
        return union_ids, union_tfs

    def precal_bm25_terms(self):
        print("pre-calculate bm25 terms...")
        self.pre_df()
        self.doc_tfs_bm25 = list(map(self.pre_tf, tqdm(range(self.doc_size))))
        self.query_tfs_bm25 = list(map(self.pre_qtf, tqdm(range(self.query_size))))
        print("Done!!")

    def pre_df(self):
        self.doc_df_bm25 = np.log((self.doc_size-self.doc_df+0.5)/(self.doc_df+0.5))
    
    def pre_tf(self, index):        
        return (k1+1)*self.doc_tfs[index]/(k1*(1-b+b*self.doc_length[index]/self.avdl)+self.doc_tfs[index])

    def pre_qtf(self, index):
        return (k3+1)*self.query_tfs[index]/(k3+self.query_tfs[index])

    cdef cal_bm25_score(self,    np.ndarray[np.int_t, ndim=1] qids, \
                                np.ndarray[np.float_t, ndim=1] qtfs, \
                                np.ndarray[np.int_t, ndim=1] ids, \
                                np.ndarray[np.float_t, ndim=1] tfs):
        cdef int p1=0, p2=0
        cdef int len1 = qids.shape[0], len2 = ids.shape[0]
        cdef double score = 0
        while p1 < len1 and p2 < len2:
            if qids[p1] < ids[p2]:
                p1 += 1
            elif qids[p1] > ids[p2]:
                p2 += 1
            else:
                score += self.doc_df_bm25[qids[p1]] * tfs[p2] * qtfs[p1]
                p1 += 1
                p2 += 1

        return score

    def build_query_tfidf(self):
        query_tf = np.zeros((self.query_size, self.vocab_size))
        for index, query in enumerate(self.query_list):
            length = len(query)
            jump = False
            for i, token in enumerate(query):
                if jump:
                    jump = False
                    continue
                if i < length - 1:
                    if (token, query[i+1]) in self.bi_vocab_dict:
                        query_tf[index][self.bi_vocab_dict.get((token, query[i+1]), 0)] += 1
                        jump = True
                    else:
                        query_tf[index][self.uni_vocab_dict.get(token, 0)] += 1
                else:
                    query_tf[index][self.uni_vocab_dict.get(token, 0)] += 1

        self.query_ids = []
        self.query_tfs = []
        for i in range(self.query_size):
            p = query_tf[i] > self.eps
            self.query_ids.append(np.arange(len(p))[p])
            self.query_tfs.append(query_tf[i][p])

    def get_query_score(self):
        self.query_score = np.zeros((self.query_size, self.doc_size))
        if self.relevance:
            print("Calculating query scores...")
            self.precal_bm25_terms()
            for i in tqdm(range(self.query_size)):
                self.query_score[i] = np.array([self.cal_bm25_score(self.query_ids[i], self.query_tfs_bm25[i], self.doc_ids[doc], self.doc_tfs_bm25[doc]) for doc in tqdm(range(self.doc_size))])
                # self.query_score[i] = np.fromiter(map(self.cal_bm25_score, repeat(self.query_ids[i]), repeat(self.query_tfs_bm25[i]), tqdm(self.doc_ids), self.doc_tfs_bm25), dtype=float)
            max_scores = self.query_score.max(axis=-1)
            self.result = np.argsort(self.query_score, axis=-1)[:,::-1]
            print("Done!")

            if r < 1:
                filter_ids = []
                for i in range(self.query_size):
                    filter_ids.append(self.query_score[i] > max_scores[i]*r)
            elif type(r) == int:
                filter_ids = np.zeros((self.query_size, self.doc_size), dtype=bool)
                for i in range(self.query_size):
                    filter_ids[i][self.result[i][:r]] = True
            else:
                print("Value of r should be either int or < 1")
                raise ValueError
            print("Doing Rocchio Feedback...")
            
            for i in tqdm(range(self.query_size)):
                relevant_num = filter_ids[i].sum()
                irrelevant_num = ir
                relevant_query_ids = np.array([], dtype=int)
                relevant_query_tfs = np.array([], dtype=float)
                irrelevant_query_ids = np.array([], dtype=int)
                irrelevant_query_tfs = np.array([], dtype=float)
                for j in range(relevant_num):
                    relevant_query_ids, relevant_query_tfs = self.add_numpy_bow(relevant_query_ids, relevant_query_tfs, self.doc_ids[self.result[i][j]], self.doc_tfs[self.result[i][j]])
                for j in range(-1,-irrelevant_num-1,-1):
                    irrelevant_query_ids, irrelevant_query_tfs = self.add_numpy_bow(irrelevant_query_ids, irrelevant_query_tfs, self.doc_ids[self.result[i][j]], self.doc_tfs[self.result[i][j]])
                
                # relevant_query = reduce(self.add_numpy_bow, compress(self.doc_ids, filter_ids[i]), compress(self.doc_tfs, filter_ids[i]))
                # relevant_query = []
                # for j in range(relevant_num):
                #     relevant_query = self.add_bow_list(relevant_query, self.doc_weight[self.result[i][j]])
                # self.query_weight[i] = self.add_bow_list(self.query_weight[i], relevant_query, alpha, beta/relevant_num)
                self.query_ids[i], self.query_tfs[i] = self.add_numpy_bow(self.query_ids[i], self.query_tfs[i]*alpha, relevant_query_ids, relevant_query_tfs*beta/relevant_num)
                self.query_ids[i], self.query_tfs[i] = self.add_numpy_bow(self.query_ids[i], self.query_tfs[i], irrelevant_query_ids, -irrelevant_query_tfs*gamma/irrelevant_num)
                threshold = np.sort(self.query_tfs[i])[::-1][:200][-1]
                p = self.query_tfs[i] >= threshold
                self.query_ids[i] = self.query_ids[i][p]
                self.query_tfs[i] = self.query_tfs[i][p]

            print("Rocchio Feedback done!")

        print("Calculating query scores...")
        self.precal_bm25_terms()
        for i in tqdm(range(self.query_size)):
            self.query_score[i] = np.array([self.cal_bm25_score(self.query_ids[i], self.query_tfs_bm25[i], self.doc_ids[doc], self.doc_tfs_bm25[doc]) for doc in tqdm(range(self.doc_size))])
            # self.query_score[i] = np.fromiter(map(self.cal_bm25_score, repeat(self.query_ids[i]), repeat(self.query_tfs_bm25[i]), tqdm(self.doc_ids), self.doc_tfs_bm25,), dtype=float)
        sorted_score = np.sort(self.query_score, axis=-1)[:,::-1]
        self.result = np.argsort(self.query_score, axis=-1)[:,::-1]

        if g < 1:
            filter_ids = []
            for i in range(self.query_size):
                filter_ids.append(sorted_score[i] > sorted_score[i][0]*g)
        elif type(g) == int:
            filter_ids = np.zeros((self.query_size, self.doc_size), dtype=bool)
            for i in range(self.query_size):
                filter_ids[i][:g] = True

        # fig, axs = plt.subplots(self.query_size)
        # fig.suptitle("top 100 query scores")
        # for i in range(self.query_size):
        #     axs[i].bar(np.arange(100), sorted_score[i][:100])
        # plt.savefig(f"{alpha}_{beta}_{r}_{g}_{k1}_{k3}_{b}_{args.output_path}.png")
        
        with open(f"{self.output_path}", 'w') as f:
            f.write("query_id,retrieved_docs\n")
            for i in range(self.query_size):
                relevant_num = filter_ids[i].sum()
                f.write(f"{self.query_id_list[i]},")
                filtered_docs = self.result[i][:relevant_num]
                for j in range(len(filtered_docs)-1):
                    f.write(f"{self.doc_id_list[filtered_docs[j]]} ")
                f.write(f"{self.doc_id_list[len(filtered_docs)-1]}\n")
        print(f"{self.output_path} saved!")
