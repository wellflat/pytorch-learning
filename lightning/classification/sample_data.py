#!/usr/bin/env python

import glob
import random
import shutil
import math


def random_sample_file(input_dir: str, output_dir: str, sample_ratio: float=0.05):
    print(f'copy {input_dir} to {output_dir}')
    files = glob.glob(input_dir + '/*.jpg')
    random_sample_file = random.sample(files,math.ceil(len(files)*sample_ratio))
    for file in random_sample_file:
        shutil.copy2(file, output_dir)
    print(f'complete {len(random_sample_file)} files')

if __name__ == '__main__':
    base_dir = '../../data/cat_or_dog'
    sample_ratio = 0.05
    random_sample_file(
        base_dir + '/train/cat',
        base_dir + '/train2/cat',
        sample_ratio
    )
    random_sample_file(
        base_dir + '/train/dog',
        base_dir + '/train2/dog',
        sample_ratio
    )