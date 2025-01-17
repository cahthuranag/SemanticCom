import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from skimage.transform import resize
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Lambda, GaussianNoise
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from keras import backend as K
import tensorflow_compression as tfc
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
import math
from PIL import Image
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.utils import ebnodb2no
from sionna.channel import AWGN, FlatFadingChannel
from tqdm.auto import tqdm
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(3)
np.random.seed(3)

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

def real_awgn(x, stddev):
    """Implements the real additive white gaussian noise channel."""
    awgn = tf.random.normal(tf.shape(x), 0, stddev, dtype=tf.float32)
    y = x + awgn
    return y
class BPGEncoder():
    def __init__(self, working_directory='./analysis/temp'):
        '''
        working_directory: directory to save temp files
                           do not include '/' in the end
        '''
        self.working_directory = working_directory
    
    def run_bpgenc(self, qp, input_dir, output_dir='temp.bpg'):
        if os.path.exists(output_dir):
            os.remove(output_dir)
        os.system(f'bpgenc {input_dir} -q {qp} -o {output_dir} -f 444')

        if os.path.exists(output_dir):
            return os.path.getsize(output_dir)
        else:
            return -1
    
    def get_qp(self, input_dir, byte_threshold, output_dir='temp.bpg'):
        '''
        iteratively finds quality parameter that maximizes quality given the byte_threshold constraint
        '''
        # rate-match algorithm
        quality_max = 51
        quality_min = 0
        quality = (quality_max - quality_min) // 2
        
        while True:
            qp = 51 - quality
            bytes = self.run_bpgenc(qp, input_dir, output_dir)
            if quality == 0 or quality == quality_min or quality == quality_max:
                break
            elif bytes > byte_threshold and quality_min != quality - 1:
                quality_max = quality
                quality -= (quality - quality_min) // 2
            elif bytes > byte_threshold and quality_min == quality - 1:
                quality_max = quality
                quality -= 1
            elif bytes < byte_threshold and quality_max > quality:
                quality_min = quality
                quality += (quality_max - quality) // 2
            else:
                break
        
        return qp
    
    def encode(self, image_array, max_bytes, header_bytes=22):
        '''
        image_array: uint8 numpy array with shape (b, h, w, c)
        max_bytes: int, maximum bytes of the encoded image file (exlcuding header bytes)
        header_bytes: the size of BPG header bytes (to be excluded in image file size calculation)
        '''

        input_dir = f'{self.working_directory}/temp_enc.png'
        output_dir = f'{self.working_directory}/temp_enc.bpg'

        im = Image.fromarray(image_array, 'RGB')
        im.save(input_dir)

        qp = self.get_qp(input_dir, max_bytes + header_bytes, output_dir)
        
        if self.run_bpgenc(qp, input_dir, output_dir) < 0:
            raise RuntimeError("BPG encoding failed")

        # read binary and convert it to numpy binary array with float dtype
        return np.unpackbits(np.fromfile(output_dir, dtype=np.uint8)).astype(np.float32)

class LDPCTransmitter():
    '''
    Transmits given bits (float array of '0' and '1') with LDPC.
    '''
    def __init__(self, k, n, m, esno_db, channel='AWGN'):
        '''
        k: data bits per codeword (in LDPC)
        n: total codeword bits (in LDPC)
        m: modulation order (in m-QAM)
        esno_db: channel SNR
        channel: 'AWGN' or 'Rayleigh'
        '''
        self.k = k
        self.n = n
        self.num_bits_per_symbol = round(math.log2(m))

        constellation_type = 'qam' if m != 2 else 'pam'
        self.constellation = Constellation(constellation_type, num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper('app', constellation=self.constellation)
        self.channel = AWGN() if channel == 'AWGN' else FlatFadingChannel
        self.encoder = LDPC5GEncoder(k=self.k, n=self.n)
        self.decoder = LDPC5GDecoder(self.encoder, num_iter=20)
        self.esno_db = esno_db
    

    def send(self, source_bits):
        '''
        source_bits: float np array of '0' and '1', whose total # of bits is divisible with k
        '''
        lcm = np.lcm(self.k, self.num_bits_per_symbol)
        source_bits_pad = tf.pad(source_bits, [[0, math.ceil(len(source_bits)/lcm)*lcm - len(source_bits)]])
        u = np.reshape(source_bits_pad, (-1, self.k))

        no = ebnodb2no(self.esno_db, num_bits_per_symbol=1, coderate=1)
        c = self.encoder(u)
        x = self.mapper(c)
        y = self.channel([x, no])
        llr_ch = self.demapper([y, no])
        u_hat = self.decoder(llr_ch)

        return tf.reshape(u_hat, (-1))[:len(source_bits)]

class BPGDecoder():
    def __init__(self, working_directory='./analysis/temp'):
        '''
        working_directory: directory to save temp files
                           do not include '/' in the end
        '''
        self.working_directory = working_directory
    
    def run_bpgdec(self, input_dir, output_dir='temp.png'):
        if os.path.exists(output_dir):
            os.remove(output_dir)
        os.system(f'bpgdec {input_dir} -o {output_dir}')

        if os.path.exists(output_dir):
            return os.path.getsize(output_dir)
        else:
            return -1

    def decode(self, bit_array, image_shape):
        '''
        returns decoded result of given bit_array.
        if bit_array is not decodable, then returns the mean CIFAR-10 pixel values.

        byte_array: float array of '0' and '1'
        image_shape: used to generate image with mean pixel values if the given byte_array is not decodable
        '''
        input_dir = f'{self.working_directory}/temp_dec.bpg'
        output_dir = f'{self.working_directory}/temp_dec.png'

        byte_array = np.packbits(bit_array.astype(np.uint8))
        with open(input_dir, "wb") as binary_file:
            binary_file.write(byte_array.tobytes())

        cifar_mean = np.array([0.4913997551666284, 0.48215855929893703, 0.4465309133731618]) * 255
        cifar_mean = np.reshape(cifar_mean, [1] * (len(image_shape) - 1) + [3]).astype(np.uint8)

        if self.run_bpgdec(input_dir, output_dir) < 0:
            # print('warning: Decode failed. Returning mean pixel value')
            return 0 * np.ones(image_shape) + cifar_mean
        else:
            x = np.array(Image.open(output_dir).convert('RGB'))
            if x.shape != image_shape:
                return 0 * np.ones(image_shape) + cifar_mean
            return x
def build_djscc_model(snrdb, blocksize):
    input_img = Input(shape=(32, 32, 3))  # Adjust the input shape based on your image size
    num_filters = 64
    conv_depth = blocksize
    # Encoder layers
    encoded = tfc.SignalConv2D(
                    num_filters,
                    (9, 9),
                    name="layer_0",
                    corr=True,
                    strides_down=2,
                    padding="same_zeros",
                    use_bias=True,
                    activation=tfc.GDN(name="gdn_0"),
                )(input_img)
    encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
    encoded = tfc.SignalConv2D(
                    num_filters,
                    (5, 5),
                    name="layer_1",
                    corr=True,
                    strides_down=2,
                    padding="same_zeros",
                    use_bias=True,
                    activation=tfc.GDN(name="gdn_1"),
                )(encoded)
    encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
    encoded = tfc.SignalConv2D(
                    num_filters,
                    (5, 5),
                    name="layer_2",
                    corr=True,
                    strides_down=1,
                    padding="same_zeros",
                    use_bias=True,
                    activation=tfc.GDN(name="gdn_2"),
                )(encoded)
    encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
    encoded = tfc.SignalConv2D(
                    num_filters,
                    (5, 5),
                    name="layer_3",
                    corr=True,
                    strides_down=1,
                    padding="same_zeros",
                    use_bias=True,
                    activation=tfc.GDN(name="gdn_3"),
                )(encoded)
    encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
    encoded = tfc.SignalConv2D(
                    conv_depth,
                    (5, 5),
                    name="layer_out",
                    corr=True,
                    strides_down=1,
                    padding="same_zeros",
                    use_bias=True,
                    activation=None,
                )(encoded)

    # Define the SNR value in dB (adjust this as needed)
    snr_value_db = snrdb
    inter_shape = tf.shape(encoded)
    # reshape array to [-1, dim_z]
    z = layers.Flatten()(encoded)
    # convert from snr to std
    print("channel_snr: {}".format(snr_value_db))
    noise_stddev = np.sqrt(10 ** (-snr_value_db / 10))

    dim_z = tf.shape(z)[1]
    # normalize latent vector so that the average power is 1
    z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.nn.l2_normalize(
        z, axis=1
    )
    z_out = real_awgn(z_in, noise_stddev)

    # convert signal back to intermediate shape
    z_out = tf.reshape(z_out, inter_shape)

    # Encoder model
    encoder = Model(inputs=input_img, outputs=z_out)  # Use the output with AWGN for the encoder

    # Classifier model
    # Use the encoder output (classifier_input) directly as the input to the classifier
    classifier_input = encoder.output  # Use encoder output as the input to the classifier
    flatten = Flatten()(classifier_input) # Flatten the output
    classifier_output = Dense(64, activation='relu')(flatten)
    classifier_output = BatchNormalization()(classifier_output)
    classifier_output = Dropout(0.5)(classifier_output)
    classifier_output = Dense(10, activation='softmax')(classifier_output)  
    classifier_model = Model(inputs=input_img, outputs=classifier_output)  # Use encoder.input here
    classifier_model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return classifier_model


def train_djscc(train_snrdb, x_train, y_train, x_val, y_val, blocksize):
    model = build_djscc_model(train_snrdb, blocksize)
    early_stopping = EarlyStopping(monitor='accuracy', mode='max', patience=50, restore_best_weights=True, verbose=1)
    history = model.fit(x_train, y_train, epochs=200, batch_size=128, validation_data=(x_val, y_val), callbacks=[early_stopping], verbose=1)
    if os.path.exists('model_weights.h5'):
        os.remove('model_weights.h5')
    model.save_weights('model_weights.h5')

    # Check if early stopping occurred
    if early_stopping.stopped_epoch != 0:
        print(f"Early stopping occurred at epoch {early_stopping.stopped_epoch}")
    else:
        print("Training completed without early stopping.")

def test_djscc(snrdb, x_test, y_test, blocksize):
    model = build_djscc_model(snrdb, blocksize)
    model.load_weights('model_weights.h5')
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return accuracy

def plot_accuracy_vs_snr_djscc(snr_values,blocksize):
    train_snrdb = 10 # Fixed SNR value for training
    # Train the model
    train_djscc(train_snrdb, x_train, y_train, x_val, y_val, blocksize)

    accuracy_values_djscc = []

    for snr in snr_values:
        accuracy = test_djscc(snr, x_test, y_test, blocksize)
        accuracy_values_djscc.append(accuracy)
        print(f"SNR: {snr} dB, Accuracy (DJSCC): {accuracy:.4f}")

    return accuracy_values_djscc

def build_classifier_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adamax',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def imBatchtoImage(batch_images):
    '''
    turns b, 32, 32, 3 images into single sqrt(b) * 32, sqrt(b) * 32, 3 image.
    '''
    batch, h, w, c = batch_images.shape
    b = int(batch ** 0.5)

    divisor = b
    while batch % divisor != 0:
        divisor -= 1
    
    image = tf.reshape(batch_images, (-1, batch//divisor, h, w, c))
    image = tf.transpose(image, [0, 2, 1, 3, 4])
    image = tf.reshape(image, (-1, batch//divisor*w, c))
    return image

def calculate_accuracy_ldpc(bw_ratio, k, n, m, snrs, num_images=100):
    bpgencoder = BPGEncoder()
    bpgdecoder = BPGDecoder()
    acc_values = []

    if isinstance(snrs, (int, float)):
        snrs = [snrs]  # Convert individual value to a list

    # Load the classification model and train it on the original CIFAR-10 dataset
    classifier_model = build_classifier_model()
    (x_train, y_train), _ = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='max', patience=50, restore_best_weights=True)
    classifier_model.fit(x_train, y_train, batch_size=128, epochs=50, validation_split=0.1, verbose=1, callbacks=[EarlyStopping])
    
    # Save the trained model weights
    classifier_model.save_weights('classifier_model_weights_ldpc.h5')

    dataset = tfds.load('cifar10', split='test', shuffle_files=False)
    for esno_db in snrs:
        ldpctransmitter = LDPCTransmitter(k, n, m, esno_db, 'AWGN')
        decoded_images = []
        original_labels = []
        for example in tqdm(dataset.take(num_images)):
            image = example['image'].numpy()
            label = example['label'].numpy()  # Get the label for the image
            image = image[np.newaxis, ...]
            b, _, _, _ = image.shape
            image = tf.cast(imBatchtoImage(image), tf.uint8)
            max_bytes = b * 32 * 32 * 3 * bw_ratio * math.log2(m) * k / n / 8
            src_bits = bpgencoder.encode(image.numpy(), max_bytes)
            rcv_bits = ldpctransmitter.send(src_bits)
            decoded_image = bpgdecoder.decode(rcv_bits.numpy(), image.shape)
            decoded_images.append(decoded_image)
            original_labels.extend([label])  # Convert label to a list before extending

        # Load the trained model weights and evaluate classification accuracy on the decoded images
        classifier_model = build_classifier_model()
        classifier_model.load_weights('classifier_model_weights_ldpc.h5')
        decoded_images = np.array(decoded_images)
        original_labels = np.array(original_labels)
        predictions = classifier_model.predict(decoded_images / 255.0)
        predicted_labels = np.argmax(predictions, axis=1)
        acc = np.mean(predicted_labels == original_labels)
        acc_values.append(acc)

        print(f'SNR={esno_db},bw={bw_ratio},k={k},n={n},m={m},Accuracy={acc:.4f}')

    return acc_values

def plot_accuracy_vs_snr_ldpc(bw_ratio, k, n, m, snr_db):
    acc_values = calculate_accuracy_ldpc(bw_ratio, k, n, m, snr_db)
    print(f"SNR values: {snr_db}")
    print(f"LDPC Accuracy values: {acc_values}")
    return acc_values

