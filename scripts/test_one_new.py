from sys import argv, exit
from PIL import Image
import numpy as np
import matplotlib.pyplot
import skimage

import sys

sys.path.insert(0, '../caffe/python/')

import caffe

PREFIX_DIR = ''

IN_DIR = PREFIX_DIR + 'maps/'
OUT_DIR = PREFIX_DIR + 'data/'
TESTS_DIR = PREFIX_DIR + 'tests/'

IMG_EXTENSION = '.png'

TRAIN_DB_FILENAME = OUT_DIR + 'train.txt'
MODEL_FILE = PREFIX_DIR + 'snapshots/_iter_1774.caffemodel'

INPUT_IMAGE_SIZE = 820

WINDOW_SIZE = 20
WINDOWS_COUNT = 2

CAFFE_NUM_IMAGES = 103 * 7

FINAL_WINDOW_SIZE = WINDOW_SIZE * (WINDOWS_COUNT * 2 + 1)
NUM_FINAL_WINDOWS = (INPUT_IMAGE_SIZE - 2 * WINDOWS_COUNT * WINDOW_SIZE) / WINDOW_SIZE
NUM_PIXELS = (INPUT_IMAGE_SIZE - FINAL_WINDOW_SIZE) + 1


caffe.set_mode_cpu()
net = caffe.Net('model/test.prototxt', MODEL_FILE, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))

net.blobs['data'].reshape(CAFFE_NUM_IMAGES, 3, FINAL_WINDOW_SIZE, FINAL_WINDOW_SIZE)


class WindowCalculator:

    def __init__(self):
        self.vec_prob = None
        self.net_prob = []

    def add_vec_prob(self, p):
        self.vec_prob = p

    def add_net_prob(self, p):
        self.net_prob.append(p)

    def get_net_prob(self):
        return sum(self.net_prob) / float(len(self.net_prob))

    def get_error(self):
        return abs(self.get_net_prob() - self.vec_prob)


def get_classess_by_xy(x, y):
    classess = []

    classess.append((y / WINDOW_SIZE, x / WINDOW_SIZE))

    t = ((y + WINDOW_SIZE - 1) / WINDOW_SIZE, (x + WINDOW_SIZE - 1) / WINDOW_SIZE)
    if t not in classess: classess.append(t)

    t = (y / WINDOW_SIZE, (x + WINDOW_SIZE - 1) / WINDOW_SIZE)
    if t not in classess: classess.append(t)

    t = ((y + WINDOW_SIZE - 1) / WINDOW_SIZE, x / WINDOW_SIZE)
    if t not in classess: classess.append(t)

    return classess


def calculate_vec(vec_img, x, y):
    left = x + WINDOWS_COUNT * WINDOW_SIZE
    upper = y + WINDOWS_COUNT * WINDOW_SIZE
    right = left + WINDOW_SIZE
    lower = upper + WINDOW_SIZE

    box = (left, upper, right, lower)
    cropped = vec_img.crop(box)

    n = 0

    for ny in range(WINDOW_SIZE):
        for nx in range(WINDOW_SIZE):
            pixel = cropped.getpixel((nx, ny))
            if not (pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255):
                n += 1

    return n / float(WINDOW_SIZE * WINDOW_SIZE)


def get_net_img_from_map(map_img, x, y):
    left = x
    upper = y
    right = left + FINAL_WINDOW_SIZE
    lower = upper + FINAL_WINDOW_SIZE

    box = (left, upper, right, lower)

    return map_img.crop(box)


def main():
    if len(argv) != 2:
        exit(1)

    id = argv[1]

    orig_map_img = Image.open(str.format('{0}{1:0>4}__orig_map{2}', OUT_DIR, id, IMG_EXTENSION))
    orig_vec_img = Image.open(str.format('{0}{1:0>4}__orig_vec{2}', OUT_DIR, id, IMG_EXTENSION))

    win_calcs = [[WindowCalculator() for x in range(NUM_FINAL_WINDOWS)] for y in range(NUM_FINAL_WINDOWS)]

    for y in range(NUM_PIXELS):
        for x in range(NUM_PIXELS):
            print(str.format('Calculating {0} x {1}...', x, y))

            # Przeliczanie vec
            if (x % WINDOW_SIZE == 0) and (y % WINDOW_SIZE == 0):
                p = calculate_vec(orig_vec_img, x, y)
                c = get_classess_by_xy(x, y)[0]
                win_calcs[c[0]][c[1]].add_vec_prob(p)

            # Liczenie bledu
            net_img = get_net_img_from_map(orig_map_img, x, y)
            n = y * NUM_PIXELS + x

            in_data = skimage.img_as_float(np.asarray(net_img, dtype=np.uint8)[:, :, :3]).astype(np.float32)

            #img = caffe.io.load_image(row[j][4], color=True)
            net.blobs['data'].data[n % CAFFE_NUM_IMAGES] = transformer.preprocess('data', in_data)

            if (n + 1) % CAFFE_NUM_IMAGES == 0:
                print('Calculating error with Caffe...')

                out = net.forward()

                for i in range(CAFFE_NUM_IMAGES):
                    nn = n - (n % CAFFE_NUM_IMAGES) + i
                    new_x = nn % NUM_PIXELS
                    new_y = nn / NUM_PIXELS
                    p = out['prob'][i].argmax()
                    classess = get_classess_by_xy(new_x, new_y)

                    for c in classess:
                        win_calcs[c[0]][c[1]].add_net_prob(p)

    err = 0

    for wcr in win_calcs:
        for wc in wcr:
            print(wc.vec_prob, len(wc.net_prob), wc.get_error())
            err += wc.get_error()

    print(str.format('Accuracy: {0}% ({1} / {2})', 100.0 - (err * 100.0) / NUM_FINAL_WINDOWS**2 , err, NUM_FINAL_WINDOWS**2))


if __name__ == '__main__':
    main()
