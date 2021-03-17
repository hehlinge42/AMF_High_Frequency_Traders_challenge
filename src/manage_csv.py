import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Change separator of csv from ; to ,')
parser.add_argument('-f', '--file', type=str, help='path to the file')
args = parser.parse_args()

csv = pd.read_csv(args.file, sep=';')
csv.to_csv(args.file, index=False)