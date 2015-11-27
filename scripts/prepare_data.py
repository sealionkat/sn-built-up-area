
from os import listdir
from PIL import Image
from random import choice


PREFIX_DIR = '../'

IN_DIR = PREFIX_DIR + 'maps/'
OUT_DIR = PREFIX_DIR + 'data/'

MAP_PREFIX = 'map_'
VEC_PREFIX = 'vec_'

IMG_EXTENSION = '.png'
FINAL_IMG_EXTENSION = '.jpg'

TRAIN_DB_FILENAME = OUT_DIR + 'train.txt'
PROCESSED_FILENAME = OUT_DIR + 'processed.txt'
#TEST_DB_FILENAME = OUT_DIR + 'test.txt'

WINDOW_SIZE = 20
WINDOWS_COUNT = 10
#TESTS_COUNT = 0


def get_input_data():
    in_data = []

    try:
        with open(PROCESSED_FILENAME, 'r') as f:
            existing_data = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    except IOError:
        existing_data = []

    map_files = []
    vec_files = []

    for f in listdir(IN_DIR):
        if f.endswith('.png'):
            if f.startswith(MAP_PREFIX):
                map_files.append(f)
            if f.startswith(VEC_PREFIX):
                vec_files.append(f)

    for mf in map_files:
        i = mf.replace(MAP_PREFIX, '').replace(IMG_EXTENSION, '')
        vf = str.format('{}{}{}', VEC_PREFIX, i, IMG_EXTENSION)
        if vf in vec_files and i not in existing_data:
            in_data.append((mf, vf, i))

    return in_data, len(existing_data) * 8


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


def get_label(vec_img):
    pos_left = WINDOW_SIZE * WINDOWS_COUNT
    pos_right = pos_left + WINDOW_SIZE
    pos_up = WINDOW_SIZE * WINDOWS_COUNT
    pos_down = pos_up + WINDOW_SIZE
    window_img = vec_img.crop((pos_left, pos_up, pos_right, pos_down))

    n = 0

    for i in range(WINDOW_SIZE):
        for j in range(WINDOW_SIZE):
            pixel = window_img.getpixel((i, j))
            if not (pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255):
                n += 1

    return 0 if n / (WINDOW_SIZE * WINDOW_SIZE) < 0.5 else 1


def create_final_images(main_images):
    final_images = []

    num_windows = WINDOWS_COUNT * 2 + 1

    for i in range(num_windows):
        for j in range(num_windows):
            pos_left = j * WINDOW_SIZE
            pos_right = pos_left + WINDOW_SIZE * num_windows
            pos_up = i * WINDOW_SIZE
            pos_down = pos_up + WINDOW_SIZE * num_windows
            map_img = main_images[0].crop((pos_left, pos_up, pos_right, pos_down))
            vec_img = main_images[1].crop((pos_left, pos_up, pos_right, pos_down))

            final_images.append((map_img, i, j, get_label(vec_img)))

    return final_images


def main():
    input_data, next_id = get_input_data()
    all_count = len(input_data) * 8
    n = 0

    for d in input_data:
        map_image = Image.open(IN_DIR + d[0])
        vec_image = Image.open(IN_DIR + d[1])

        images = generate_more_images(map_image, vec_image)

        for i in images:
            print(str.format('Processing image {} / {}... [{}]', n + 1, all_count, d[2]))
            final_images = create_final_images(i)

            with open(TRAIN_DB_FILENAME, 'a') as f:
                for entry in final_images:
                    filename = str.format('{0}{1:0>4}__{2:0>3}_{3:0>3}{4}', OUT_DIR, n + next_id, entry[1], entry[2], FINAL_IMG_EXTENSION)
                    entry[0].save(filename)
                    f.write(str.format('{0} {1}\n', filename, entry[3]))

            n += 1

        with open(PROCESSED_FILENAME, 'a') as f:
            f.write(str.format('{0}\n', d[2]))

    #if TESTS_COUNT > 0:
    #    with open(TEST_DB_FILENAME, 'w') as f:
    #        for i in range(TESTS_COUNT):
    #            entries = choice(image_entries)
    #            image_entries.remove(entries)

    #            for entry in entries:
    #                f.write(str.format('{0} {1}\n', entry[0], entry[1]))

    #with open(TRAIN_DB_FILENAME, 'a') as f:
    #    for entries in image_entries:
    #        for entry in entries:
    #            f.write(str.format('{0} {1}\n', entry[0], entry[1]))

    print('\nDone!')


if __name__ == '__main__':
    main()
