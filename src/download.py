import os
import pandas as pd
from utils import download_images

DATASET_FOLDER = '../dataset1/'
IMAGE_FOLDER = '../images/'

def main():
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))

    # Download images for the entire dataset
    download_images(test['image_link'], os.path.join(IMAGE_FOLDER, 'test'))

if __name__ == '__main__':
    main()