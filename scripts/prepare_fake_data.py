
from sys import argv, exit
from PIL import Image
from random import choice


def usage():
    print('Usage: %s <window_size> <num_windows_w> <num_windows_h> <num_images> <num_test_images>' % argv[0])
    exit(1)


def set_rand_window(pixels, win_size, i, j):
    col = choice([0, 255])

    for x in range(win_size):
        for y in range(win_size):
            pixels[i * win_size + x, j * win_size + y] = (col, col, col)


def create_main_image(w, h, wc_w, wc_h, win_size):
    main_img = Image.new('RGB', (w, h), 'black')
    pixels = main_img.load()

    for i in range(wc_h):
        for j in range(wc_w):
            set_rand_window(pixels, win_size, j, i)

    return main_img


def get_label(img, w, h):
    return 0 if (img.load()[w / 2, h / 2][0] == 0) else 1


def create_final_images(main_img, w, h, wc_w, wc_h, win_size):
    final_images = []

    for i in range(wc_h):
        for j in range(wc_w):
            pos_left = j * win_size
            pos_right = pos_left + w
            pos_up = i * win_size
            pos_down = pos_up + h

            new_img = main_img.crop((pos_left, pos_up, pos_right, pos_down))
            final_images.append((new_img, i, j, get_label(new_img, w, h)))

    return final_images


def run():
    if len(argv) != 6:
        usage()

    window_size = int(argv[1])
    num_w = int(argv[2])
    num_h = int(argv[3])
    num_images = int(argv[4])
    num_test = int(argv[5])

    final_windows_count_w = (2 * num_w + 1)
    final_windows_count_h = (2 * num_h + 1)

    main_windows_count_w = final_windows_count_w + 2 * num_w
    main_windows_count_h = final_windows_count_h + 2 * num_h

    final_images_w = window_size * final_windows_count_w
    final_images_h = window_size * final_windows_count_h

    main_image_w = window_size * main_windows_count_w
    main_image_h = window_size * main_windows_count_h

    with open('fake_data/train.txt', 'wb') as f:
        for n in range(num_images):
            print('Generating image %d...' % n)
            main_image = create_main_image(main_image_w, main_image_h, main_windows_count_w, main_windows_count_h, window_size)
            final_images = create_final_images(main_image, final_images_w, final_images_h, final_windows_count_w, final_windows_count_h, window_size)

            for img in final_images:
                filename = 'fake_data/%04d__%03d_%03d.png' % (n, img[1], img[2])
                img[0].save(filename)
                f.write(('%s %d\n' % (filename, img[3])).encode())

    print('')

    with open('fake_data/test.txt', 'wb') as f:
        for n in range(num_test):
            print('Generating test image %d...' % n)
            main_image = create_main_image(main_image_w, main_image_h, main_windows_count_w, main_windows_count_h, window_size)
            final_images = create_final_images(main_image, final_images_w, final_images_h, final_windows_count_w, final_windows_count_h, window_size)

            for img in final_images:
                filename = 'fake_data/test_%04d__%03d_%03d.png' % (n, img[1], img[2])
                img[0].save(filename)
                f.write(('%s %d\n' % (filename, img[3])).encode())

if __name__ == '__main__':
    run()
