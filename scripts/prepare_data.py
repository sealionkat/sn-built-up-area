
from os import listdir
from PIL import Image


PREFIX_DIR = '../'

IN_DIR = PREFIX_DIR + 'maps/'
OUT_DIR = PREFIX_DIR + 'data/'

MAP_PREFIX = 'map_'
VEC_PREFIX = 'vec_'

IMG_EXTENSION = '.png'

WINDOW_SIZE = 20
WINDOWS_COUNT = 10


def get_input_data():
    in_data = []

    map_files = []
    vec_files = []

    for f in listdir(IN_DIR):
        if f.endswith('.png'):
            if f.startswith(MAP_PREFIX):
                map_files.append(f)
            if f.startswith(VEC_PREFIX):
                vec_files.append(f)

    for mf in map_files:
        i = int(mf.replace(MAP_PREFIX, '').replace(IMG_EXTENSION, ''))
        vf = str.format('{}{}{}', VEC_PREFIX, i, IMG_EXTENSION)
        if vf in vec_files:
            in_data.append((mf, vf))

    return in_data


def transpose_images(images, direction):
    new_images = [(x[0].transpose(direction), x[1].transpose(direction)) for x in images]
    new_images.extend(images)
    return new_images


def generate_more_images(map_image, vec_image):
    images = [(map_image, vec_image)]
    images = transpose_images(images, Image.ROTATE_90)
    images = transpose_images(images, Image.FLIP_LEFT_RIGHT)
    images = transpose_images(images, Image.FLIP_TOP_BOTTOM)

    return images


def main():
    input_data = get_input_data()

    for d in input_data:
        map_image = Image.open(IN_DIR + d[0])
        vec_image = Image.open(IN_DIR + d[1])

        images = generate_more_images(map_image, vec_image)

        for i in images:
            # TODO
            #i[0].show()
            #i[1].show()
            pass


if __name__ == '__main__':
    main()
