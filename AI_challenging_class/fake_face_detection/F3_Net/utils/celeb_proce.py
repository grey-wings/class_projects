import os
import random
import shutil


if __name__ == '__main__':
    l = os.listdir(r'G:\AI\celeb-df-v2\Celeb-synthesis')
    random.shuffle(l)
    l = l[:1000]
    for i in l:
        k = os.path.join(r'G:\AI\celeb-df-v2\Celeb-synthesis', i)
        shutil.move(k, os.path.join(r'G:\AI\celeb-df-v2\celeb-syn', i))
