import os
from six.moves.urllib.request import urlretrieve
import gzip, binascii, struct, numpy
import matplotlib.pyplot as plt

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = "/tmp/mnist-data"

%matplotlib inline

def maybe_download(filename):
    """A helper to download the data files if not present."""
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not os.path.exists(WORK_DIRECTORY):
        os.mkdir(WORK_DIRECTORY)
    if not os.path.exists(filepath):
        filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    else:
        print('Already downloaded', filename)
    return filepath

train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

#get train and test imgae data
IMAGE_SIZE = 28
PIXEL_DEPTH = 255
TRAIN_IMAGE_NUM = 60000
TEST_IMAGE_NUM = 10000

# read train and test file into memory
def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].

    For MNIST data, the number of channels is always 1.

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and dimensions; we know these values.
        bytestream.read(16)

        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        return data

train_data = extract_data(train_data_filename, TRAIN_IMAGE_NUM)
test_data = extract_data(test_data_filename, TEST_IMAGE_NUM)

# print image in the data file
def print_image(row, colum, start_pos, datas):
    print("printed data image shape", datas.shape)
    print("print imager number %d; first image pos %d"%(row * colum, start_pos))

    if (row * colum + start_pos) > len(datas):
        print("Error:invalid input, Out of bounds of datas")
        return 

    fig = plt.figure()
    for i in range (row):
        for j in range (colum):
            p = fig.add_subplot(row, colum, i*colum+j+1)
            p.imshow(datas[start_pos + i*colum+j].reshape(IMAGE_SIZE,IMAGE_SIZE), cmap=plt.cm.Greys);

print_image(10, 10, 100, train_data)

