# -*- coding: utf-8


def file_reader(file_name: str):
    """[summary]

    Arguments:
        file_name {str} -- [파일명]

    Returns:
        [str] -- [file에서 한줄 읽은 라인을 넘김]
    """
    # print(f"file name : {file_name}")
    for row in open(file_name, "r", encoding="utf-8"):
        yield row
