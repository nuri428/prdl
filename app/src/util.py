# -*- coding: utf-8

from defines import TRAIN_DIR
import os


def file_reader(file_name: str, **kwargs):
    """[summary]

    Arguments:
        file_name {str} -- [파일명]

    Returns:
        [str] -- [file에서 한줄 읽은 라인을 넘김]
    """

    # print(f"file name : {file_name}")
    for row in open(file_name, "r", encoding="utf-8"):
        yield row


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
        yield line[kwargs["key"]]


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

    readers = [
        fr(file_name=os.path.join(base_dir, file_name), **kwargs)
        for file_name in os.listdir(base_dir)
    ]

    return list_to_iterables(readers)


def list_to_iterables(data_list):
    for sub_gen in data_list:
        for element in sub_gen:
            # print((element))
            yield element
