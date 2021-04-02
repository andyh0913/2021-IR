import os
import math
import argparse
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm


class Retriever:
    def __init__(self, relevance, input_path, output_path, model_path, dir_path):
        self.build_vocab(model_path) # Must be called first
        self.build_doc_list(model_path, dir_path)
        self.build_tfidf(model_path) # Must be called after build_doc_list
        self.arrangement() # Must be called after build_tfidf
        self.build_query_list(input_path)
        self.build_query_tfidf() # Must be called after build_query_list & build_tfidf
        
    
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
            if c.isalnum():
                english_vocab = english_vocab + c
            else:
                if not english_vocab:
                    index_list.append(self.uni_vocab_dict.get(english_vocab, 0))
                    english_vocab = ""
                else:
                    index_list.append(self.uni_vocab_dict.get(c, 0))
        return index_list


    def build_query_list(self, input_path):
        input_tree = ET.parse(input_path)
        self.query_list = []
        self.query_id_list = []
        for topic in input_tree.getroot().findall('topic'):
            number = topic.find('number').text[-3:]
            text =  topic.find('title').text + \
                    topic.find('question').text + \
                    topic.find('narrative').text + \
                    topic.find('concepts').text
            self.query_list.append(self.tokenize(text))
            self.query_id_list.append(number)

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
            text = doc.find('title').text or ""
            for p in doc.find('text').findall('p'):
                text = text + p.text
            tokenized = self.tokenize(text)
            self.doc_list.append(tokenized)
            doc_length.append(len(tokenized))
            self.doc_id_list.append(doc.find('id').text.lower())
        self.doc_size = len(self.doc_list)
        self.doc_length = np.array(doc_length)
        self.doc_length_avg = self.doc_length.mean()
        print("Document list built!")

    def build_tfidf(self, model_path):
        print("Building tfidf...")
        print(self.doc_size, self.vocab_size)
        self.doc_tf = np.zeros((self.doc_size, self.vocab_size))
        self.doc_idf = np.zeros(self.vocab_size)
        with open(os.path.join(model_path, 'inverted-file')) as f:
            file_list = f.readlines()
        vocab = 0
        for line in tqdm(file_list):
            line_list = line.strip().split()
            if len(line_list) == 3:
                if line_list[1] == "-1":
                    vocab = int(line_list[0])
                    self.doc_idf[vocab] = math.log2(self.doc_size/float(line_list[2]))
                else:
                    self.bi_vocab_dict[(int(line_list[0]), int(line_list[1]))] = self.vocab_size
                    vocab = self.vocab_size
                    np.append(self.doc_tf, np.zeros((self.doc_size, 1)), axis=1)
                    np.append(self.doc_idf, [0])
                    self.doc_idf[vocab] = math.log2(self.doc_size/float(line_list[2]))
                    self.vocab_size += 1
            else:
                self.doc_tf[int(line_list[0])][vocab] = int(line_list[1])
        self.doc_weight = self.doc_tf * self.doc_idf / self.doc_length
        print("Tfidf built!")

    def arrangement(self):
        # Implementation of Ltc SMART-IR
        # tf: L
        tf_avg = self.doc_tf.mean()
        self.doc_tf = (1 + np.log2(self.doc_tf)) / (1 + np.log2(tf_avg))
        # df: t
        # zero-corrected idf
        # dl: c
        self.doc_weight = self.doc_tf * self.doc_idf
        self.doc_norm = (self.doc_weight ** 2).sum(axis=1) ** 0.5

        # pivot normalization
        slope = 0.2
        pivot = self.doc_length_avg
        self.doc_norm = (1.0 - slope) * pivot + slope * self.doc_norm
        self.doc_weight = self.doc_tf * self.doc_idf / self.doc_norm

    def build_query_tfidf(self):
        self.query_tf = np.zeros((len(self.query_list), self.vocab_size))
        for index, query in enumerate(self.query_list):
            length = len(query)
            jump = False
            for i, token in enumerate(query):
                if jump:
                    continue
                if i < length - 1:
                    if (token, token_list[i+1]) in self.bi_vocab_dict:
                        self.query_tf[index][self.bi_vocab_dict.get((token, token_list[i+1]), 0)] += 1
                        jump = True
                    else:
                        self.query_tf[index][self.uni_vocab_dict.get(token, 0)] += 1
                else:
                    self.query_tf[index][self.uni_vocab_dict.get(token, 0)] += 1
        
        self.query_weight = self.query_tf * self.doc_idf
        self.query_norm = (self.query_weight ** 2).sum(axis=1) ** 0.5

        # pivot normalization
        slope = 0.2
        pivot = self.doc_length_avg
        self.query_norm = (1.0 - slope) * pivot + slope * self.query_norm
        self.query_weight = self.query_tf * self.doc_idf / self.query_norm

    def get_query_score(self):
        scores = np.matmul(self.query_weight, self.doc_weight.T)
        self.result = np.argsort(scores, axis=-1)[:,::-1]
        print(self.result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--relevance", action="store_true")
    parser.add_argument("-i", "--input_path", type=str)
    parser.add_argument("-o", "--output_path", type=str)
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument("-d", "--dir_path", type=str)
    args = parser.parse_args()

    retriever = Retriever(**vars(args))