
from sys import argv, exit
from PIL import Image

import sys

sys.path.insert(0, '../caffe/python/')

import caffe

PREFIX_DIR = ''

IN_DIR = PREFIX_DIR + 'maps/'
OUT_DIR = PREFIX_DIR + 'data/'
TESTS_DIR = PREFIX_DIR + 'tests/'

TRAIN_DB_FILENAME = OUT_DIR + 'train.txt'
MODEL_FILE = PREFIX_DIR + 'snapshots/_iter_66.caffemodel'

INPUT_IMAGE_SIZE = 820

WINDOW_SIZE = 20
WINDOWS_COUNT = 2

def set_window_color(pixels, i, j, val):
    for x in range(WINDOW_SIZE):
        for y in range(WINDOW_SIZE):
            pixels[j * WINDOW_SIZE + x, i * WINDOW_SIZE + y] = (val * 255, val * 255, val * 255)

def set_window_from_img(image, i, j, val):
    pos = WINDOW_SIZE * WINDOWS_COUNT
    posn = pos + WINDOW_SIZE
    part = val.crop((pos, pos, posn, posn))
    image.paste(part, (j * WINDOW_SIZE, i * WINDOW_SIZE, (j + 1) * WINDOW_SIZE, (i + 1) * WINDOW_SIZE))

def main():
    if len(argv) != 2:
        exit(1)

    id = argv[1]

    with open(TRAIN_DB_FILENAME, 'r') as f:
        data = [
            (line.split()[0].strip(), line.split()[1].strip())
            for line in f.readlines()
            if len(line) > 3 and str.format('{0}{1:0>4}__', OUT_DIR, id) in line.strip()
        ]

        parts = []

        for part in data:
            nums = part[0].split('__')[1].split('.')[0].split('_')
            parts.append((id, int(nums[0]), int(nums[1]), part[1], part[0]))

        rows = {}

        for part in parts:
            try:
                r = rows[part[1]]
            except KeyError:
                r = []
                rows[part[1]] = r

            r.append(part)


        caffe.set_mode_cpu()
        net = caffe.Net('model/test.prototxt', MODEL_FILE, caffe.TEST)

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))

        net.blobs['data'].reshape(37, 3, 100, 100) # TODO

        n = 0
        ok = 0

        size = INPUT_IMAGE_SIZE - 2 * WINDOW_SIZE * WINDOWS_COUNT

        image_photo = Image.new('RGB', (size, size), 'red')
        image_original = Image.new('RGB', (size, size), 'red')
        image_net = Image.new('RGB', (size, size), 'red')

        pixels_photo = image_photo.load()
        pixels_original = image_original.load()
        pixels_net = image_net.load()

        for i in rows:
            row = rows[i]
            print("Processing row " + str(i))

            for j in range(len(row)):
                img = caffe.io.load_image(row[j][4], color=True)
                net.blobs['data'].data[j] = transformer.preprocess('data', img)

            out = net.forward()

            for j in range(len(row)):
                if out['prob'][j].argmax() == int(row[j][3]):
                    ok += 1

                set_window_color(pixels_original, i, j, int(row[j][3]))
                set_window_color(pixels_net, i, j, out['prob'][j].argmax())
                set_window_from_img(image_photo, i, j, Image.open(row[j][4]))

                n += 1

        print(str.format('Accuracy: {0}% ({1} / {2})', (ok * 100.0) / n , ok, n))

        image_original.save(str.format('{0}{1:0>4}_orig.png', TESTS_DIR, id))
        image_net.save(str.format('{0}{1:0>4}_net.png', TESTS_DIR, id))
        image_photo.save(str.format('{0}{1:0>4}_map.png', TESTS_DIR, id))

if __name__ == '__main__':
    main()