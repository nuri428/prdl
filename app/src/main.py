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
from util import file_reader

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


def generate_multi_generator():
    # train_dir
    # current_dir = os.getcwd()
    # base_dir = os.path.join(current_dir, train_dir)
    base_dir = TRAIN_DIR

    readers = [
        file_reader(os.path.join(base_dir, file_name))
        for file_name in os.listdir(base_dir)
    ]

    return list_to_iterables(readers)


def get_args():
    """[summary]

    Returns:
        [type] -- [description]
    """
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data-path"", type=str, default="data")
    # parser.add_argument("train_dir", type=str)
    # parser.add_argument("n_cores", type=int, default=4)
    # parser.add_argument("-ratio", type=float, default=0.2)
    # parser.add_argument("-debug", default=False, action="store_true")
    # args = parser.parse_args()
    # return args

    # def get_args():
    """set CLI arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data")
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
    readers = generate_multi_generator()

    index = 0
    for line in readers:
        index = index + 1
        print(f"line extract test : {line}")
        if index > 5:
            break

    for line in readers:
        if morpheme_func:
            yield morpheme_func(line)
        else:
            yield line


def main():
    """[summary]
    메인 함수
    """
    args = get_args()
    print(args)

    print(TRAIN_DIR)

    print()

    # trainset 생성
    generate_trainset(args.data_path, args.sample_rate)

    print("sampling complete")

    if args.pretokenizer_type == "khaiii":
        from khaiii import KhaiiiApi

        api = KhaiiiApi()
        morpheme_func = api.analyze

    elif args.pretokenizer_type == "mecab":
        from konlpy.tag import Mecab

        mecab = Mecab()
        morpheme_func = mecab.morphs

    # pretokenizer 호출
    # morpheme_func = None
    count = 0
    for line in get_pretokenize_generator(morpheme_func):
        print(" ".join([element.lex for element in line]))
        # # print(type(line))
        # # print([element for element in line])
        # for element in line:
        #     print(element)
        count = count + 1
        if count > 5:
            break
    #     # print(get_pretokenize_generator(morpheme_func))
    # print(morpheme_func("이것은 테스트입니다"))
    # print()

    # for element in api.analyze("이것은 테스트입니다"):
    #     print(element)
    #     print(type(element))


if __name__ == "__main__":
    main()
