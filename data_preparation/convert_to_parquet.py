# Steal from https://github.com/ExplainableML/flair/blob/main/preprocess/convert_to_parquet.py

import pandas as pd
import argparse


def csv_to_parquet(input_path: str, output_path: str):
    print("Start converting, this may take a while...")
    df = pd.read_csv(input_path)
    if output_path is None:
        output_path = input_path.replace(".csv", ".parquet")
    df.to_parquet(output_path, index=False)
    print("Conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a CSV file to Parquet format.")
    parser.add_argument("-i", "--input_path", required=True, type=str, help="Path to the input CSV file")
    parser.add_argument("-o", "--output_path", type=str, help="Path to save the output Parquet file")
    args = parser.parse_args()

    csv_to_parquet(args.input_path, args.output_path)