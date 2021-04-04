import os
import math
import argparse
import xml.etree.ElementTree as ET
from functools import reduce
from itertools import compress, repeat
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class Retriever:
    def __init__(self, relevance, input_path, output_path, model_path, dir_path):
        self.eps = 1e-12
        self.build_vocab(model_path) # Must be called first
        self.build_doc_list(model_path, dir_path)
        self.build_stopword_list('model/stopwords')
        self.build_tfidf(model_path) # Must be called after build_doc_list
        self.doc_arrangement() # Must be called after build_tfidf
        self.build_query_list(input_path)
        self.build_query_tfidf() # Must be called after build_query_list & build_tfidf
        self.query_arrangement()
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
        doc_length = []
        for i, file_path in enumerate(tqdm(file_list)):
            doc_tree = ET.parse(os.path.join(dir_path, file_path.strip()))
            doc = doc_tree.getroot().find('doc')
            # text = doc.find('title').text or ""
            # for p in doc.find('text').findall('p'):
            #     text = text + p.text
            # tokenized = self.tokenize(text)
            # self.doc_list.append(tokenized)
            # doc_length.append(len(tokenized))
            self.doc_id_list.append(doc.find('id').text.lower())
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
        self.doc_idf = [0] * self.vocab_size
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
                    self.doc_idf[vocab] = math.log2(self.doc_size/int(line_list[2]))
                    if vocab in self.stopwords:
                        skip = True
                    else:
                        skip = False
                else:
                    self.bi_vocab_dict[(int(line_list[0]), int(line_list[1]))] = self.vocab_size
                    vocab = self.vocab_size
                    self.doc_idf.append( math.log2(self.doc_size/int(line_list[2])) )
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
            # p = self.doc_ids[i].argsort()
            # self.doc_ids[i] = self.doc_ids[i][p]
            # self.doc_tfs[i] = self.doc_tfs[i][p]
            length = self.doc_tfs[i].sum()
            if length == 0:
                self.doc_ids[i] = np.array([0])
                self.doc_tfs[i] = np.array([1])
                length = 1
            self.doc_length.append(length)
        self.doc_length = np.array(self.doc_length)
        self.doc_idf = np.array(self.doc_idf)
        print("Tfidf built!")
        

    def list2array(self, bow_list):
        # if len(bow_list) != 0:
        #     termids, tfs = zip(*bow_list)
        # else:
        #     termids, tfs = [],[]
        # return termids, tfs

        termid_array, tf_array = [], []
        for termid, tf in bow_list:
            termid_array.append(termid)
            tf_array.append(tf)
        return termid_array, tf_array

    def array2list(self, termid_array, tf_array):
        # bow_list = []
        # for termid, tf in zip(termid_array, tf_array):
        #     if tf > self.eps:
        #         bow_list.append((termid, tf))
        
        bow_list = [(termid, tf) for (termid, tf) in zip(termid_array, tf_array) if tf > self.eps]
        return bow_list

    def add_numpy_bow(self, ids1, tfs1, ids2, tfs2):
        union_ids = np.union1d(ids1, ids2)
        union_tfs = np.zeros(len(union_ids))
        union_tfs[np.in1d(union_ids, ids1)] += tfs1
        union_tfs[np.in1d(union_ids, ids2)] += tfs2
        return union_ids, union_tfs


    def add_bow_list(self, list1, list2, alpha=1, beta=1):
        # alpha*list1 + beta*list2
        p1, p2 = 0, 0
        score = 0
        new_list = []
        len1, len2 = len(list1), len(list2)
        list1.append((self.vocab_size, 0))
        list2.append((self.vocab_size, 0))
        while (p1 < len1 or p2 < len2):
            if list1[p1][0] < list2[p2][0]:
                new_list.append((list1[p1][0], alpha*list1[p1][1]))
                p1 += 1
            elif list1[p1][0] > list2[p2][0]:
                new_list.append((list2[p2][0], beta*list2[p2][1]))
                p2 += 1
            else:
                new_list.append((list1[p1][0], alpha*list1[p1][1] + beta*list2[p2][1]))
                p1 += 1
                p2 += 1
        list1.pop()
        list2.pop()
        return new_list

    def logtf(self, bow_list):
        ids, array = self.list2array(bow_list)
        array = np.array(array)
        tf_avg = array.mean()
        array = (1 + np.log2(array)) / (1 + np.log2(tf_avg))
        return self.array2list(ids, array)

    def cal_doc_weight_and_norm(self, ids, tfs):
        tf_avg = tfs.mean()
        tfs = (1 + np.log2(tfs)) / (1 + np.log2(tf_avg))
        tfs = tfs * self.doc_idf[ids]
        norm = np.linalg.norm(tfs)
        norm = (1.0 - self.slope) * self.pivot + self.slope * norm
        tfs = tfs / norm
        return tfs

    def doc_arrangement(self):
        print("Start SMART-IR arrangement...")
        self.slope = 0.2
        self.pivot = self.doc_length.mean()
        # Implementation of Ltc SMART-IR
        # tf: L
        # for i in tqdm(range(self.doc_size)):
        #     ids, array = self.list2array(self.doc_tf[i])
        #     array = np.array(array)
        #     tf_avg = array.mean()
        #     array = (1 + np.log2(array)) / (1 + np.log2(tf_avg))
        #     self.doc_tf[i] = self.array2list(ids, array)
        # self.doc_tf = map(self.logtf, self.doc_tf)
        # df: t
        # zero-corrected idf
        # dl: c
        # self.doc_norm = np.zeros(self.doc_size)
        # self.doc_weight = [[] for _ in range(self.doc_size)]
        # for i in tqdm(range(self.doc_size)):
        #     ids, array = self.list2array(self.doc_tf[i])
        #     array = np.array(array)
        #     weight = array * self.doc_idf[list(ids)]
        #     norm = np.linalg.norm(weight)
        #     self.doc_weight[i] = self.array2list(ids, weight)
        #     self.doc_norm[i] = norm
        self.doc_tfs = list(map(self.cal_doc_weight_and_norm, self.doc_ids, tqdm(self.doc_tfs)))

        # # pivot normalization
        # self.doc_norm = (1.0 - self.slope) * self.pivot + self.slope * self.doc_norm
        # for i in tqdm(range(self.doc_size)):
        #     ids, array = self.list2array(self.doc_weight[i])
        #     array = np.array(array)
        #     array = array / self.doc_norm[i]
        #     self.doc_weight[i] = self.array2list(ids, array)
        
        print("SMART-IR arrangement done!")

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

    def query_arrangement(self):
        # SMART-IR
        for i in range(self.query_size):
            tf_avg = self.query_tfs[i].mean()
            self.query_tfs[i] = (1 + np.log2(self.query_tfs[i])) / (1 + np.log2(tf_avg))
            self.query_tfs[i] = self.query_tfs[i] * self.doc_idf[self.query_ids[i]]
            norm = np.linalg.norm(self.query_tfs[i])
            norm = (1.0 - self.slope) * self.pivot + self.slope * norm
            self.query_tfs[i] = self.query_tfs[i] / norm

    def get_similarity_sparse(self, list1, list2):
        p1, p2 = 0, 0
        score = 0
        while (p1 < len(list1) and p2 < len(list2)):
            if list1[p1][0] < list2[p2][0]:
                p1 += 1
            elif list1[p1][0] > list2[p2][0]:
                p2 += 1
            else:
                score += list1[p1][1] * list2[p2][1]
                p1 += 1
                p2 += 1
        return score

    def get_similarity_dense(self, list1, list2):
        array1 = np.zeros(self.vocab_size)
        array2 = np.zeros(self.vocab_size)
        ids1, tfs1 = self.list2array(list1)
        array1[list(ids1)] = list(tfs1)
        ids2, tfs2 = self.list2array(list2)
        array2[list(ids2)] = list(tfs2)
        return (array1*array2).sum()

    def get_similarity_numpy(self, ids1, tfs1, ids2, tfs2):
        return (tfs1[np.in1d(ids1, ids2)] * tfs2[np.in1d(ids2, ids1)]).sum()

    def get_query_score(self):
        print("Calculating query scores...")
        self.query_score = np.zeros((self.query_size, self.doc_size))
        for i in tqdm(range(self.query_size)):
            self.query_score[i] = np.fromiter(map(self.get_similarity_numpy, tqdm(self.doc_ids), self.doc_tfs, repeat(self.query_ids[i]), repeat(self.query_tfs[i])), dtype=float)
        max_scores = self.query_score.max(axis=-1)
        self.result = np.argsort(self.query_score, axis=-1)[:,::-1]
        print("Done!")

        # Use score > 0.8
        # filter_ids = []
        # for i in range(self.query_size):
        #     filter_ids.append(self.query_score[i] > max_scores[i]*0.8)

        # Use top 10 result
        filter_ids = np.zeros((self.query_size, self.doc_size), dtype=int)
        for i in range(self.query_size):
            filter_ids[i][self.result[i][:10]] = 1

        if args.relevance:
            print("Doing Rocchio Feedback...")
            alpha = 0.5
            beta = 1
            gamma = 0
            for i in tqdm(range(self.query_size)):
                relevant_num = filter_ids[i].sum()
                relevant_query_ids = np.array([])
                relevant_query_tfs = np.array([])
                for j in range(relevant_num):
                    relevant_query_ids, relevant_query_tfs = self.add_numpy_bow(relevant_query_ids, relevant_query_tfs, self.doc_ids[self.result[i][j]], self.doc_tfs[self.result[i][j]])
                # relevant_query = reduce(self.add_numpy_bow, compress(self.doc_ids, filter_ids[i]), compress(self.doc_tfs, filter_ids[i]))
                # relevant_query = []
                # for j in range(relevant_num):
                #     relevant_query = self.add_bow_list(relevant_query, self.doc_weight[self.result[i][j]])
                # self.query_weight[i] = self.add_bow_list(self.query_weight[i], relevant_query, alpha, beta/relevant_num)
                relevant_query_ids, relevant_query_tfs = self.add_numpy_bow(self.query_ids[i], self.query_tfs[i], relevant_query_ids, relevant_query_tfs*beta/relevant_num)
            print("Rocchio Feedback done!")

        print("Calculating query scores...")
        for i in tqdm(range(self.query_size)):
            self.query_score[i] = np.fromiter(map(self.get_similarity_numpy, tqdm(self.doc_ids), self.doc_tfs, repeat(self.query_ids[i]), repeat(self.query_tfs[i])), dtype=float)

        sorted_score = np.sort(self.query_score, axis=-1)[:,::-1]
        filter_ids = []
        for i in range(self.query_size):
            filter_ids.append(sorted_score[i] > sorted_score[i][0]*0.5)
        self.result = np.argsort(self.query_score, axis=-1)[:,::-1]
        print("Done!")

        fig, axs = plt.subplots(self.query_size)
        fig.suptitle("top 100 query scores")
        for i in range(self.query_size):
            axs[i].bar(np.arange(100), sorted_score[i][:100])
        plt.show()
        
        with open(args.output_path, 'w') as f:
            f.write("query_id,retrieved_docs\n")
            for i in range(self.query_size):
                f.write(f"{self.query_id_list[i]},")
                filtered_docs = self.result[i][filter_ids[i]]
                for j in range(len(filtered_docs)-1):
                    f.write(f"{self.doc_id_list[filtered_docs[j]]} ")
                f.write(f"{self.doc_id_list[len(filtered_docs)-1]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--relevance", action="store_true")
    parser.add_argument("-i", "--input_path", type=str)
    parser.add_argument("-o", "--output_path", type=str)
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument("-d", "--dir_path", type=str)
    args = parser.parse_args()

    retriever = Retriever(**vars(args))