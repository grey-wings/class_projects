import pandas as pd


def getData(path: str) -> pd.DataFrame:
    """
    读取.data文件.
    :param path:要读取的文件的路径
    :return:
    """
    df = pd.read_csv(path, header=None)
    return df


if __name__ == '__main__':
    df1 = getData('bezdekIris.data')
    print(df1)
