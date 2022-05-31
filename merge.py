import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from tkinter.tix import Tree
import numpy as np
from tqdm import tqdm
# exceutor = ProcessPoolExecutor(max_workers=128)
j = 0
for i in range(20):
    files = []
    if i > 0:
        for f in tqdm(os.listdir(f'/local/s3092593/games{i + 1}')):
            a = np.load(f'/local/s3092593/games{i + 1}/{f}', allow_pickle=True)
            files.append(a)
            # exceutor.submit(shutil.copyfile, f'/local/s3092593/games{i + 1}/{f}', f'/local/s3092593/merges_games/game{j}.npy')
            j += 1
            # print(f)
    else:
        for f in tqdm(os.listdir(f'/local/s3092593/games')):
            a = np.load(f'/local/s3092593/games/{f}', allow_pickle=True)
            files.append(a)
            j += 1
            if j > 10000:
                break
            # print(f)
    np.savez(f'/local/s3092593/games{i + 1}.npz', *files)
    print('******************************done')