import glob
from tqdm import tqdm
import imagehash
from PIL import Image, ImageChops

import pandas as pd

def check_hash(pre_hash, re_hash):
    for pre, re in zip(pre_hash, re_hash):
        hash_value = pre - re
        if hash_value != 0:
            return False
    
    return True

def solve_deplicate():
    data_list = pd.read_csv('./img/untreated_labels.csv')

    files = glob.glob('./img/*.jpg')
    hashes = []
    drop_index = []
    # 各画像のhash値を導出
    print('calcurate hash...')
    for index, file in tqdm(enumerate(files)):
        # 破損している画像はここで排除
        try:
            img = Image.open(file)
        except:
            drop_index.append(index)
        ave_hash = imagehash.average_hash(img)
        p_hash = imagehash.phash(img)
        d_hash = imagehash.dhash(img)
        w_hash = imagehash.whash(img)
        hashes.append([ave_hash, p_hash, d_hash, w_hash])
    print('Broken images num: {}'.format(len(drop_index)))
    
    print('check same image...')
    for pre_index, pre_hash in tqdm(enumerate(hashes[:-1])):
        if pre_index in drop_index:
            continue
        for re_index, re_hash in enumerate(hashes[pre_index+1:], pre_index+1):
            if re_index in drop_index:
                continue
            if check_hash(pre_hash, re_hash):
                drop_index.append(re_index)
            
    # 念の為listを一度uniqueに
    print('Drop image num: {}'.format(len(drop_index)))
    drop_index = sorted(list(set(drop_index)))
    data_list = data_list.drop(data_list.index[drop_index])
    data_list.to_csv('./img/labels.csv', index=False)

if __name__=='__main__':
    solve_deplicate()