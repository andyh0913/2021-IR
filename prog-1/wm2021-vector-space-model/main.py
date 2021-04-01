import os
import argparse
import xml.etree.ElementTree as ET
import numpy as np


class Retriever:
    def __init__(self, do_relevance, input_path, output_path, model_path, dir_path):
        self.build_vocab(model_path) # The index started from 1
        self.build_query_vector(input_path)
    
    def build_vocab(self, model_path):
        with open(os.path.join(model_path, "vocab.all")) as f:
            vocab_list = f.readlines()
            self.vocab_dict = {k[:-1]: i for i,k in enumerate(vocab_list)}
    
    def tokenize(self, text):
        index_list = []
        english_vocab = ""
        for i, c in enumerate(text):
            if c.isalnum():
                english_vocab = english_vocab + c
            else:
                if not english_vocab:
                    index_list.append(self.vocab_dict.get(english_vocab, 0))
                    english_vocab = ""
                else:
                    index_list.append(self.vocab_dict.get(c, 0))
        return index_list


    def build_query_vector(self, input_path):
        input_tree = ET.parse(args.input)
        topic_text_dict = {}
        for topic in input_tree.findall('topic'):
            number = topic.find('number').text[-3:]
            text =  topic.find('title').text +
                    topic.find('question').text +
                    topic.find('narrative').text +
                    topic.find('concepts').text
            topic_text_dict[number] = self.tokenize(text)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--relevance", action="store_true")
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-d", "--directory", type=str)
    args = parser.parse_args()

    retriever = Retriever(*vars(args))

    input_tree = ET.parse(args.input)
    # print(input_tree.findall('topic'))
    # for topic in input_tree.findall('topic'):
    #     print(topic.find('number').text)
