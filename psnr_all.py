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

def real_awgn(x, stddev):
    """Implements the real additive white gaussian noise channel."""
    awgn = tf.random.normal(tf.shape(x), 0, stddev, dtype=tf.float32)
    y = x + awgn
    return y

def build_model(snrdb, blocksize):
    input_img = Input(shape=(32, 32, 3))  # CIFAR-10 images have a size of 32x32x3
    num_filters = 16
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

    snr_value_db = snrdb
    inter_shape = tf.shape(encoded)
    z = layers.Flatten()(encoded)
    print("channel_snr: {}".format(snr_value_db))
    noise_stddev = np.sqrt(10 ** (-snr_value_db / 10))

    dim_z = tf.shape(z)[1]
    z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.nn.l2_normalize(
        z, axis=1
    )
    z_out = real_awgn(z_in, noise_stddev)
    z_out = tf.reshape(z_out, inter_shape)

    decoded = tfc.SignalConv2D(
        num_filters,
        (5, 5),
        corr=False,
        strides_up=1,
        padding="same_zeros",
        use_bias=True,
        activation=tfc.GDN(name="igdn_out", inverse=True),
    )(z_out)
    decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
    decoded = tfc.SignalConv2D(
        num_filters,
        (5, 5),
        corr=False,
        strides_up=1,
        padding="same_zeros",
        use_bias=True,
        activation=tfc.GDN(name="igdn_0", inverse=True),
    )(decoded)
    decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
    decoded = tfc.SignalConv2D(
        num_filters,
        (5, 5),
        corr=False,
        strides_up=1,
        padding="same_zeros",
        use_bias=True,
        activation=tfc.GDN(name="igdn_1", inverse=True),
    )(decoded)
    decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
    decoded = tfc.SignalConv2D(
        num_filters,
        (5, 5),
        corr=False,
        strides_up=2,
        padding="same_zeros",
        use_bias=True,
        activation=tfc.GDN(name="igdn_2", inverse=True),
    )(decoded)
    decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
    n_channels = 3
    decoded = tfc.SignalConv2D(
        n_channels,
        (9, 9),
        corr=False,
        strides_up=2,
        padding="same_zeros",
        use_bias=True,
        activation=tf.nn.sigmoid,
    )(decoded)

    def psnr_metric(x_in, x_out):
        if type(x_in) is list:
            img_in = x_in[0]
        else:
            img_in = x_in
        return tf.image.psnr(img_in, x_out, max_val=1.0)

    recovery_model = Model(inputs=input_img, outputs=decoded)
    model_metrics = [
        tf.keras.metrics.MeanSquaredError(),
        psnr_metric,
        PSNRsVar(),
    ]
    recovery_model.compile(optimizer='adam', loss='mse', metrics=model_metrics)
    return recovery_model

  # Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='psnr_metric', # Monitor validation loss
                               mode='max',         # Stop training when the quantity monitored has stopped increasing
                               patience=20,       # Number of epochs with no improvement after which training will be stopped
                               verbose=1,        # Print messages about the early stopping process
                               restore_best_weights=True)  # Restore model weights from the epoch with the best value of the monitored quantity

# Add the EarlyStopping callback to your list of callbacks when training the model
callbacks = [early_stopping]


def get_dataset(num_images=200):
    # Download and load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train, x_val, _, _ = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    
    # Select a subset of test images
    x_test = x_test[:num_images]
    y_test = y_test[:num_images]
    
    return x_train, x_val, x_test

def train(train_snrdb, x_train, x_val, x_test, block_size):
    rec_model = build_model(train_snrdb, block_size)
    rec_model.fit(x_train, x_train, batch_size=128, epochs=10, validation_data=(x_val, x_val), verbose=1, callbacks=callbacks)
    if os.path.exists('classifier_model_weights_rec_train.h5'):
        os.remove('classifier_model_weights_rec_train.h5')
    rec_model.save_weights('classifier_model_weights_rec_train.h5')
    _, psnr_train, _, _ = rec_model.evaluate(x_test, x_test, verbose=0)
    return psnr_train

def test(snrdb, x_test, block_size):
    rec_model_test = build_model(snrdb, block_size)
    rec_model_test.load_weights('classifier_model_weights_rec_train.h5')
    _, _, psnr_test, _ = rec_model_test.evaluate(x_test, x_test, verbose=0)
    return psnr_test

class PSNRsVar(tf.keras.metrics.Metric):
    """Calculate the variance of a distribution of PSNRs across batches"""

    def __init__(self, name="variance", **kwargs):
        super(PSNRsVar, self).__init__(name=name, **kwargs)
        self.count = self.add_weight(name="count", shape=(), initializer="zeros")
        self.mean = self.add_weight(name="mean", shape=(), initializer="zeros")
        self.var = self.add_weight(name="M2", shape=(), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        psnrs = tf.image.psnr(y_true, y_pred, max_val=1.0)
        samples = tf.cast(psnrs, self.dtype)
        batch_count = tf.size(samples)
        batch_count = tf.cast(batch_count, self.dtype)
        batch_mean = tf.math.reduce_mean(samples)
        batch_var = tf.math.reduce_variance(samples)

        new_count = self.count + batch_count
        new_mean = (self.count * self.mean + batch_count * batch_mean) / (
            self.count + batch_count
        )
        new_var = (
            (self.count * (self.var + tf.square(self.mean - new_mean)))
            + (batch_count * (batch_var + tf.square(batch_mean - new_mean)))
        ) / (self.count + batch_count)

        self.count.assign(new_count)
        self.mean.assign(new_mean)
        self.var.assign(new_var)

    def result(self):
        return self.var

    def reset_states(self):
        self.count.assign(np.zeros(self.count.shape))
        self.mean.assign(np.zeros(self.mean.shape))
        self.var.assign(np.zeros(self.var.shape))

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

def calculate_psnr(bw_ratio, k, n, m, snrs, num_images=10):
    bpgencoder = BPGEncoder()
    bpgdecoder = BPGDecoder()
    psnr_values = []
    
    if isinstance(snrs, (int, float)):
        snrs = [snrs]  # Convert individual value to a list
    
    dataset = tfds.load('cifar10', split='test', shuffle_files=False)
    for esno_db in snrs:
        i = 0
        psnr = 0
        ssim = 0
        total_images = 0
        ldpctransmitter = LDPCTransmitter(k, n, m, esno_db, 'AWGN')
        for example in tqdm(dataset.take(num_images)):
            image = example['image'].numpy()
            image = image[np.newaxis, ...]
            b, _, _, _ = image.shape
            image = tf.cast(imBatchtoImage(image), tf.uint8)
            max_bytes = b * 32 * 32 * 3 * bw_ratio * math.log2(m) * k / n / 8
            src_bits = bpgencoder.encode(image.numpy(), max_bytes)
            rcv_bits = ldpctransmitter.send(src_bits)
            decoded_image = bpgdecoder.decode(rcv_bits.numpy(), image.shape)
            total_images += b
            psnr = (total_images - b) / (total_images) * psnr + float(b * tf.image.psnr(decoded_image, image, max_val=255)) / (total_images)
            ssim = (total_images - b) / (total_images) * ssim + float(b * tf.image.ssim(tf.cast(decoded_image, dtype=tf.float32), tf.cast(image, dtype=tf.float32), max_val=255)) / (total_images)
        print(f'SNR={esno_db},bw={bw_ratio},k={k},n={n},m={m},PSNR={psnr:.2f},SSIM={ssim:.2f}')
        psnr_values.append(psnr)
    
    return psnr_values

if __name__ == "__main__":
    bw_ratio = 1/3
    k = 3072
    n = 4608
    m = 4
    snr_db = np.linspace(10, 20, num=2)
    psnr_values = calculate_psnr(bw_ratio, k, n, m, snr_db)
    
    x_train, x_val, x_test = get_dataset()
    train_snrdb = 10
    block_size = 16
    train_psnr = train(train_snrdb, x_train, x_val, x_test, block_size)
    
    djscc_psnr_values = []
    for snr_value_db in snr_db:
        test_psnr = test(snr_value_db, x_test, block_size)
        djscc_psnr_values.append(test_psnr)
        print(f"SNR: {snr_value_db}, PSNR: {test_psnr}")  # Print the SNR and PSNR values
    
    print(f"SNR values: {snr_db}")  # Print the SNR values
    print(f"DJSCC PSNR values: {djscc_psnr_values}")  # Print the DJSCC PSNR values
    print(f"LDPC PSNR values: {psnr_values}")  # Print the LDPC PSNR values
    
    plt.plot(snr_db, djscc_psnr_values, marker='o', linestyle='-', color='b', label='DJSCC')
    if psnr_values is not None:
        plt.plot(snr_db, psnr_values, marker='x', linestyle='--', color='r', label='LDPC')
    plt.xlabel('SNR (dB)')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR as a function of SNR')
    plt.grid(True)
    plt.legend()
    plt.show()
