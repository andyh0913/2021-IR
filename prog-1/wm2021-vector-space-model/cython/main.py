import argparse
from model import Retriever
import time

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--relevance", action="store_true")
    parser.add_argument("-i", "--input_path", type=str)
    parser.add_argument("-o", "--output_path", type=str)
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument("-d", "--dir_path", type=str)
    args = parser.parse_args()

    retriever = Retriever(**vars(args))
    print(f"Execution time: {time.time()-start} sec")