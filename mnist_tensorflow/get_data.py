import os
from six.moves.urllib.request import urlretrieve
import gzip, binascii, struct, numpy
import matplotlib.pyplot as plt

%matplotlib inline

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = "/tmp/mnist-data"

class PrepareMnistData:
    source_url ='http://yann.lecun.com/exdb/mnist/'
    data_dir = "/tmp/mnist-data"

    TrainDataFileNameGz = 'train-images-idx3-ubyte.gz'
    TrainLabelsFileNameGz = 'train-labels-idx1-ubyte.gz'
    TestDataFileNameGz = 't10k-images-idx3-ubyte.gz' 
    TestLabelsFileNameGz = 't10k-labels-idx1-ubyte.gz'

    #get train and test imgae data
    IMAGE_SIZE = 28
    PIXEL_DEPTH = 255
    TRAIN_IMAGE_NUM = 60000
    TEST_IMAGE_NUM = 10000
    NUM_LABELS = 10
    VALIDATION_SIZE = 5000
        
    __train_data_filename = ''
    __train_labels_filename = ''
    __test_data_filename = ''
    __test_labels_filename = '' 

    __train_data = None
    __train_labels = None
    __test_data = None
    __test_labels = None
    __validation_data = None
    __validation_labels = None
   
    @classmethod
    def __maybe_download(cls, filename):
        """A helper to download the data files if not present."""
        filepath = os.path.join(cls.data_dir, filename)
        if not os.path.exists(cls.data_dir):
            os.mkdir(cls.data_dir)
        if not os.path.exists(filepath):
            filepath, _ = urlretrieve(cls.source_url + filename, filepath)
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        else:
            print('Already downloaded', filename)
        print('data directory', cls.data_dir)
        return filepath

    @classmethod
    def prepare_data_file(cls):
        cls.__train_data_filename = cls.__maybe_download(cls.TrainDataFileNameGz)
        cls.__train_labels_filename = cls.__maybe_download(cls.TrainLabelsFileNameGz)
        cls.__test_data_filename = cls.__maybe_download(cls.TestDataFileNameGz)
        cls.__test_labels_filename = cls.__maybe_download(cls.TestLabelsFileNameGz)

    # read train and test file into memory
    @classmethod
    def __extract_data(cls, filename, num_images):
        """Extract the images into a 4D tensor [image index, y, x, channels].

        For MNIST data, the number of channels is always 1.

        Values are rescaled from [0, 255] down to [-0.5, 0.5].
        """
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            # Skip the magic number and dimensions; we know these values.
            bytestream.read(16)

            buf = bytestream.read(cls.IMAGE_SIZE * cls.IMAGE_SIZE * num_images)
            data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
            data = (data - (cls.PIXEL_DEPTH / 2.0)) / cls.PIXEL_DEPTH
            data = data.reshape(num_images, cls.IMAGE_SIZE, cls.IMAGE_SIZE, 1)
            return data

    @classmethod
    def do_train_test_data(cls):
        cls.__train_data = cls.__extract_data(cls.__train_data_filename, cls.TRAIN_IMAGE_NUM)
        cls.__test_data = cls.__extract_data(cls.__test_data_filename, cls.TEST_IMAGE_NUM)

    # print image in the data file
    @classmethod
    def print_image(cls, row, colum, start_pos, datas):
        print("printed data image shape", datas.shape)
        print("print imager number %d; first image pos %d"%(row * colum, start_pos))

        if (row * colum + start_pos) > len(datas):
            print("Error:invalid input, Out of bounds of datas")
            return 

        fig = plt.figure()
        for i in range (row):
            for j in range (colum):
                p = fig.add_subplot(row, colum, i*colum+j+1)
                p.imshow(datas[start_pos + i*colum+j].reshape(cls.IMAGE_SIZE, cls.IMAGE_SIZE), cmap=plt.cm.Greys);

    #labels data
    @classmethod
    def __extract_labels(cls, filename, num_images):
        """Extract the labels into a 1-hot matrix [image index, label index]."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            # Skip the magic number and count; we know these values.
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = numpy.frombuffer(buf, dtype=numpy.uint8)
            # Convert to dense 1-hot representation.
            labels = (numpy.arange(cls.NUM_LABELS) == labels[:, None]).astype(numpy.float32)
            print('Training labels shape', labels.shape)
            return labels

    @classmethod
    def do_train_test_label(cls):
        cls.__train_labels = cls.__extract_labels(cls.__train_labels_filename, cls.TRAIN_IMAGE_NUM)
        cls.__test_labels = cls.__extract_labels(cls.__test_labels_filename, cls.TEST_IMAGE_NUM)

    #validation data
    @classmethod
    def do_validation_data_label(cls):
        cls.__validation_data = cls.__train_data[:cls.VALIDATION_SIZE, :, :, :]
        cls.__validation_labels = cls.__train_labels[:cls.VALIDATION_SIZE]
        cls.__train_data = cls.__train_data[cls.VALIDATION_SIZE:, :, :, :]
        cls.__train_labels = cls.__train_labels[cls.VALIDATION_SIZE:]
        
        train_size = cls.__train_labels.shape[0]
        print('Validation shape', cls.__validation_data.shape)
        print('Train size', train_size)

    @classmethod
    def run(cls):
        cls.prepare_data_file()
        cls.do_train_test_data()
        cls.do_train_test_label()
        cls.do_validation_data_label()
        print("train,test,validation data and lablels all ready!!!")

    @classmethod
    def get_train_data_labels(cls):
        return cls.__train_data, cls.__train_labels

    @classmethod
    def get_test_data_labels(cls):
        return cls.__test_data, cls.__test_labels

    @classmethod
    def get_validation_data_labels(cls):
        return cls.__validation_data, cls.__validation_labels

if __name__ == '__main__':
    PrepareMnistData.run()

    train_data, _ = PrepareMnistData.get_train_data_labels()
    print("first 100 elements in train data, shape 10 x 10")
    PrepareMnistData.print_image(10, 10, 0, train_data)

    test_data, _ = PrepareMnistData.get_test_data_labels()
    print("first 100 elements in test data, shape 10 x 10")
    PrepareMnistData.print_image(10, 10, 0, test_data)

    validation_data, _ = PrepareMnistData.get_validation_data_labels()
    print("first 100 elements in validation data, shape 10 x 10")
    PrepareMnistData.print_image(10, 10, 0, validation_data)
