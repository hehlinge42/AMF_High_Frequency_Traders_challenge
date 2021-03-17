import argparse
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Changes separator of csv file')
    parser.add_argument('-f', '--files', nargs='*', type=str, help='paths to the csv')
    args = parser.parse_args()

    for file in args.files:
        df = pd.read_csv(file, sep=';', index_col='Trader')
        df.to_csv(file, index=True)