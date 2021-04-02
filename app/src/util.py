# -*- coding: utf-8

from defines import TRAIN_DIR
import os
import re


def file_reader(file_name: str, **kwargs):
    """[summary]

    Arguments:
        file_name {str} -- [파일명]

    Returns:
        [str] -- [file에서 한줄 읽은 라인을 넘김]
    """

    # print(f"file name : {file_name}")
    for line in open(file_name, "r", encoding="utf-8"):
        line = re.sub(r"[\u4e00-\u9fff]+", "", line)
        yield line


def jsonp_reader(file_name: str, **kwargs):
    """[summary]

    Arguments:
        file_name {str} -- [description]
        key {str} -- [description]
    """
    import json

    if "key" not in kwargs:
        raise ValueError("must have key parameter")

    for row in open(file_name, "r", encoding="utf-8"):
        line = json.loads(row)
        line = line[kwargs["key"]]
        # remove hanja
        line = re.sub(r"[\u4e00-\u9fff]+", "", line)
        yield line


def get_list_file(dir_path: str) -> list:
    file_list = os.listdir(dir_path)
    return file_list


def generate_multi_generator(**kwargs):
    """[summary]

    Raises:
        ValueError: [description]

    Returns:
        [type] -- [description]
    """
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

    file_list = []
    for root, dirs, files in os.walk(base_dir):
        for file_name in files:
            file_list.append(os.path.join(root, file_name))
        for dir_name in dirs:
            sub_file_names = get_list_file(os.path.join(root, dir_name))
            for file_name in sub_file_names:
                file_list.append(os.path.join(root, dir_name, file_name))

    print(f"file list count : {len(file_list)}")

    readers = [
        jsonp_reader(file_name=os.path.join(base_dir, file_name), key="text")
        if file_name.endswith(".json")
        else file_reader(file_name=os.path.join(base_dir, file_name), **kwargs)
        for file_name in file_list
    ]

    return list_to_iterables(readers)


def list_to_iterables(data_list):
    for sub_gen in data_list:
        for element in sub_gen:
            # print((element))
            yield element
