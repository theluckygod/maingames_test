import os
from PIL import Image
import utils

GROUND_TRUTH_PATH = 'test_data/test.txt'
TEST_DATA_PATH = 'test_data/test_images'
OUTPUT_PATH = 'test_data/test_images_processed'


if __name__ == '__main__':
    with open(GROUND_TRUTH_PATH, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    ground_truth = list(map(lambda x: x.split('\t'), lines))
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    for filename, label in ground_truth:
        print(filename, label)
        img = Image.open(os.path.join(TEST_DATA_PATH, filename))
        img = utils.process_test_image(img)
        img.save(os.path.join(OUTPUT_PATH, label + "_" + filename))