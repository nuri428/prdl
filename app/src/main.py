# test

import os
import sys
import argparse
from subsample.subsampling import gen_trainset
import multiprocessing
from tqdm import tqdm
from itertools import product, chain, combinations
import traceback
from multiprocessing.pool import ThreadPool
from util import file_reader, jsonp_reader

from khaiii import KhaiiiApi, KhaiiiWord
from konlpy.tag import Mecab

from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.models import BPE, WordPiece
from tokenizers.pre_tokenizers import Whitespace, BertPreTokenizer, CharDelimiterSplit

TRAIN_DIR = os.path.join(os.path.dirname(os.getcwd()), "trainset")

n_cores = 12
debug_mode = False


def generate_trainset(dataset_dir: str, sample_rate: float):
    """[summary]

    Arguments:
        dataset_dir {str} -- [데이터셋 디렉토리]

    Returns:
        boolean -- [동작 실행 결과]
    """
    train_dir = TRAIN_DIR
    current_dir = os.getcwd()
    # p = multiprocessing.Pool(n_cores)
    pool = ThreadPool(n_cores)
    data_path = os.path.join(current_dir, dataset_dir)

    try:
        file_list = os.listdir(data_path)
        print(f"file_list count : {len(file_list)}")
        target_path = os.path.join(current_dir, train_dir)

        results = [
            pool.apply_async(
                gen_trainset, (data_path, file_name, target_path, sample_rate)
            )
            for file_name in file_list
        ]

        pool.close()
        pool.join()

        if debug_mode:
            for result in results:
                out = result.get()
                print(f"out:{out}")
    except OSError as e:
        traceback.print_exception(*sys.exc_info())
        print(e)
        return False
    except Exception as e:
        traceback.print_exception(*sys.exc_info())
        print(e)
        return False
    return True


def list_to_iterables(data_list):
    for sub_gen in data_list:
        for element in sub_gen:
            # print((element))
            yield element


def generate_multi_generator(**kwargs):
    # train_dir
    # current_dir = os.getcwd()
    # base_dir = os.path.join(current_dir, train_dir)
    fr = file_reader
    if "file_type" in kwargs:
        file_type = kwargs["file_type"]
    else:
        file_type = "txt"

    if file_type == "json":
        if "key" not in kwargs:
            raise ValueError("must have key parameter")
        else:
            fr = jsonp_reader

    base_dir = TRAIN_DIR

    readers = [
        fr(file_name=os.path.join(base_dir, file_name), **kwargs)
        for file_name in os.listdir(base_dir)
    ]

    return list_to_iterables(readers)


def get_args():
    """[summary]

    Returns:
        [type] -- [description]
    """
    # def get_args():
    """set CLI arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="../datasets/")
    parser.add_argument("--sample-rate", type=float, default=0.01)
    parser.add_argument(
        "--pretokenizer-type", type=str, choices=["khaiii", "mecab"], default="khaiii"
    )
    parser.add_argument(
        "--tokenizer-type", type=str, choices=["bbpe", "cbpe", "wp"], default="bbpe"
    )
    parser.add_argument("--vocab-size", default=10000)
    parser.add_argument("--min-frequency", default=2),
    parser.add_argument("--special-tokens", nargs="+", default=[])
    parser.add_argument("--seed", type=float, default=42)
    args = parser.parse_args()
    return args


def get_pretokenize_generator(morpheme_func):
    #  = None
    readers = generate_multi_generator(file_type="json", key="text")
    count = 0

    for line in readers:
        # count = count + 1
        # if count > 10:
        #     break

        if morpheme_func:
            line = morpheme_func(line)
            line_str = " ".join(
                [
                    element.lex if isinstance(element, KhaiiiWord) else element[0]
                    for element in line
                ]
            )
            # print(line_str)
            yield line_str
        else:
            yield line


def train_tokenizer(args):
    """[summary]

    Arguments:
        args {[dictionary]} -- [arguments객체]
    """
    # Tokenizer train
    if args.pretokenizer_type == "khaiii":
        api = KhaiiiApi()
        morpheme_func = api.analyze
    elif args.pretokenizer_type == "mecab":
        mecab = Mecab()
        morpheme_func = mecab.morphs

    # tokenizer-type", type=str, choices=["bbpe", "cbpe", "wp"], default="bbpe"
    if args.tokenizer_type == "bbpe":
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = BertPreTokenizer()
        trainer = BpeTrainer(
            special_tokens=args.special_tokens,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
    elif args.tokenizer_type == "cbpe":
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = CharDelimiterSplit
        trainer = BpeTrainer(
            special_tokens=args.special_tokens,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
    elif agrs.tokenizer_type == "wp":
        tokenizer = Tokenizer(WordPiece())
        tokenizer.pre_tokenizer = Whitespace
        trainer = WordPieceTrainer(
            special_tokens=args.special_tokens,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )

    # tokenizer.train()
    tokenizer.train_from_iterator(
        get_pretokenize_generator(morpheme_func), trainer=trainer
    )
    tokenizer.save("./vocab.bin")
    test_string = "안녕하세요 이것은 테스트입니다"
    output = tokenizer.encode(test_string)
    print(output)
    print(output.tokens)
    print(output.type_ids)
    print(output.offsets)


def main():
    """[summary]
    메인 함수
    """
    args = get_args()
    print(args)

    # trainset 생성
    generate_trainset(args.data_path, args.sample_rate)

    print("sampling complete")

    train_tokenizer(args)


if __name__ == "__main__":
    main()
