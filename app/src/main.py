# test

import os
import sys
import argparse
from subsample.subsampling import gen_trainset
import multiprocessing
from tqdm import tqdm
from itertools import product
import traceback
from multiprocessing.pool import ThreadPool


def generate_trainset(
    dataset_dir: str, train_dir: str, n_cores: int, ratio: float, debug_mode
):
    """[summary]

    Arguments:
        dataset_dir {str} -- [데이터셋 디렉토리]

    Returns:
        boolean -- [동작 실행 결과]
    """
    current_dir = os.getcwd()
    # p = multiprocessing.Pool(n_cores)
    pool = ThreadPool(n_cores)
    data_path = os.path.join(current_dir, dataset_dir)

    try:
        file_list = os.listdir(data_path)
        print(f"file_list count : {len(file_list)}")
        target_path = os.path.join(current_dir, train_dir)

        results = [
            pool.apply_async(gen_trainset, (data_path, file_name, target_path, ratio))
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


def get_args():
    """[summary]

    Returns:
        [type] -- [description]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("train_dir", type=str)
    parser.add_argument("n_cores", type=int, default=4)
    parser.add_argument("-ratio", type=float, default=0.2)
    parser.add_argument("-debug", default=False, action="store_true")
    args = parser.parse_args()
    return args


def main():
    """[summary]
    메인 함수
    """
    args = get_args()
    # print(args)
    generate_trainset(
        args.dataset_dir, args.train_dir, args.n_cores, args.ratio, args.debug
    )
    print("complete")


if __name__ == "__main__":
    main()
