import os
import shutil

def deldir(dir_path):
    try:
        shutil.rmtree(dir_path)
    except FileNotFoundError:
        pass

def mkdir(dir_path, delete=False):
    if delete:
        deldir(dir_path)
        os.makedirs(dir_path)
    else:
        os.makedirs(exist_ok=True)