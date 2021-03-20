# -*- coding: utf-8

"""[summary]
subsampling, txt 파일을 입력 받아서 랜덤으로 추출, 
ratio 비율에 맞게 추출하여 trainerset에 복사함
"""

import subprocess
import os

call_count = 0


def gen_trainset(
    source_path: str, dataset_file: str, target_path: str, sample_ratio: float = 0.1
):
    """[summary]
    입력 받은 텍스트 파일에서 줄 단위 랜덤으로 순서를 바꾸고, 
    입력 받은 비율 만큼 추출하여 학습을 위한 경로로 복사를 수행
    Arguments:
        source_path {str} -- [소스 파일을 읽어 오는 디렉토리]
        dataset_file {str} -- [소스 파일]
        target_path {str} -- [타겟 경로n]
        ratio {float} -- [description]
    """

    # print(source_path, dataset_file, target_path, sample_ratio)
    sample_ratio = float(sample_ratio)
    # check raito
    if sample_ratio > 1.0 or sample_ratio < 0.0:
        raise Exception("ration must be 0<= ratio <= 1")

    source = os.path.join(source_path, dataset_file)
    target = os.path.join(target_path, dataset_file)

    # print(f"{source}->{target}")

    # checo source file exists
    if not os.path.isfile(source):
        raise Exception(f"file {source} not exists")

    # check target dir exists
    if not os.path.isdir(target_path):
        raise Exception(f"target dir {target_path} not exists")

    # extract line count
    file_count = int(subprocess.check_output(["wc", "-l", source]).split()[0])
    # calc sample count
    ratio_count = int(file_count * ((sample_ratio * 100) / 100))

    # print(f"file count :{file_count}")
    # print(f"ratio count :{ratio_count}")

    awk_cmd = (
        "awk 'BEGIN{srand()} {print rand(), $0}' "
        + source
        + " | sort -n -k 1 "
        + """ | awk 'sub(/^[^ ]* /, "")' """
        + " | head -n "
        + str(ratio_count)
        + " > "
        + target
    )

    # print(awk_sub_cmd)

    # sampling_output = subprocess.Popen(awk_cmd)

    sampling_output = subprocess.Popen(
        awk_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True
    )

    # print(f"sampleing outut={sampling_output}")
    # return True
    return sampling_output


if __name__ == "__main__":
    gen_trainset(
        "/home/nuri/sources/git/prdl/app/datasets/",
        # "part-19715-b96bfb62-0215-41ea-84ec-cd7ef44fd4ba-c000.txt",
        "all.txt",
        "/home/nuri/sources/git/prdl/app/trainset/",
        sample_ratio=1,
    )
