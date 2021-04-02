import os
import argparse
import json
from tqdm import tqdm


def get_args():
    """[summary]

    Returns:
        [type] -- [description]
    """
    # def get_args():
    """set CLI arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default="../datasets/")
    parser.add_argument("--line-count", type=int, default=-1)
    parser.add_argument("--key", type=str, default="text")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    file_count = 0
    line_count = 0
    writer = open(f"{file_count+1}.txt", "w", encoding="utf-8")

    for row in tqdm(open(args.data_file, "r", encoding="utf-8")):
        line = json.loads(row)
        # print(line["_source"]["content"])
        writer.write(line["_source"]["content"])
        line_count = line_count + 1
        if (line_count + 1) % args.line_count == 0:
            writer.close()
            file_count = file_count + 1
            writer = open(f"{file_count+1}.txt", "w", encoding="utf-8")


if __name__ == "__main__":
    main()
