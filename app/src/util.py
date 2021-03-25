# -*- coding: utf-8


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

