import argparse
import xml.etree.ElementTree as ET

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--relevance", action="store_true")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-d", "--directory", type=str, required=True)
    args = parser.parse_args()

    input_xml = ET.parse(args.input).getroot()
    