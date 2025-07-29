import os
import sys

###### 数値の確認 ######

def isfloat(s: str) -> bool:
    '''
    文字列sがfloat型に変換可能かを判定する．
    具体的には実際にfloat(s)を実行し，変更可能かを検証する
    '''
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True

def isint(s: str):
    '''
    文字列sがfloat型に変換可能かを判定する．
    具体的には実際にfloat(s)を実行し，変更可能かを検証する
    '''
    try:
        int(s, 10)
    except ValueError:
        return False
    else:
        return True

def isNumber(s: str):
    return isfloat(s) or isint(s)

###### ファイル操作 ######

def resolve_conflict_filename(fname: str, path: str) -> str:
    '''
        ファイル名の重複解決

        pathにfnameがすでに存在すれば，fanme0など，後ろに数字をつけることで解決する
        なければfnameを返却する
    '''
    files_and_folders = [ff for ff in os.listdir()]

    # 名前が重複しなければそのまま
    fpath = os.path.join(path, fname)
    if not os.path.exists(fpath):
        return fname

    id = 0
    while True:
        candidate_fpath = fpath + '_' + str(id)
        if not os.path.exists(candidate_fpath):
            return fname + '_' + str(id) 
        id += 1

##### list計算 #####
def get_moveaverage(lst: list, span: int) -> list:
    ma = [0] * len(lst)
    for idx, val in enumerate(range(len(lst))):
        ma[idx] = sum(lst[max(0, idx - span - 1):idx]) / span
    
    return ma