
import sys

sys.path.insert(0, '../caffe/python/')

import caffe

def run():
    caffe.set_mode_cpu()
    net = caffe.Net('fake_model/deploy.prototxt', 'snapshots/fake.caffemodel', caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    #transformer.set_raw_scale('data', 255)

    net.blobs['data'].reshape(49, 1, 140, 140)

    n = 0
    ok = 0

    with open('fake_data/test.txt', 'r') as f:
        lines = f.readlines()

        for line in lines:
            l = line.split()
            if len(l) == 2:
                print('Processing image %s' % l[0].strip())
                n += 1

                img = caffe.io.load_image(l[0].strip(), color=False)
                img = img[:,:,[0]]

                net.blobs['data'].data[...] = transformer.preprocess('data', img)

                out = net.forward()
                if(out['prob'][0].argmax() == int(l[1].strip())):
                    ok += 1

    print('')
    print('Accuracy: %d%%' % ((ok * 100.0) / n))



if __name__ == '__main__':
    run()