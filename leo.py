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

class LEOChannel:
    def __init__(self, carrier_freq=20e9, orbit_height=600e3, elevation_angle=30, 
                 rain_rate=10, antenna_gain_tx=30, antenna_gain_rx=0, 
                 rician_k=10):
        """
        LEO Satellite Channel Model
        """
        self.carrier_freq = carrier_freq
        self.orbit_height = orbit_height
        self.elevation_angle = elevation_angle
        self.rain_rate = rain_rate
        self.antenna_gain_tx = antenna_gain_tx
        self.antenna_gain_rx = antenna_gain_rx
        self.rician_k_linear = 10**(rician_k/10)
        
        self.R_earth = 6371e3
        self.c = 3e8
        
        self.G_total = self.calculate_aggregate_gain()
        print(f"LEO Channel: G_total = {self.G_total:.2e}, K = {rician_k}dB")
        
    def calculate_slant_range(self):
        epsilon_rad = math.radians(self.elevation_angle)
        d = self.R_earth * (math.sqrt(((self.orbit_height + self.R_earth) / self.R_earth)**2 - 
                                    math.cos(epsilon_rad)**2) - math.sin(epsilon_rad))
        return d
    
    def calculate_free_space_path_loss(self):
        d = self.calculate_slant_range()
        fspl = (4 * math.pi * d * self.carrier_freq / self.c)**2
        return fspl
    
    def calculate_rain_attenuation(self):
        k = 0.075
        alpha = 1.099
        epsilon_rad = math.radians(self.elevation_angle)
        L = (0.00741 * self.rain_rate**0.776 + 
             (0.232 - 0.00018) * math.sin(epsilon_rad))**-1
        rain_att = k * self.rain_rate**alpha * L
        return rain_att
    
    def calculate_total_path_loss(self):
        fspl = self.calculate_free_space_path_loss()
        rain_att_linear = 10**(self.calculate_rain_attenuation() / 10)
        total_pl_linear = fspl * rain_att_linear
        return total_pl_linear
    
    def calculate_aggregate_gain(self):
        total_pl = self.calculate_total_path_loss()
        antenna_gain_tx_linear = 10**(self.antenna_gain_tx / 10)
        antenna_gain_rx_linear = 10**(self.antenna_gain_rx / 10)
        G_total = (antenna_gain_tx_linear * antenna_gain_rx_linear) / total_pl
        return G_total

def rician_fading_channel(x, snrdb, leo_channel):
    """
    Implements Rician fading channel with LEO satellite parameters
    """
    # Convert SNR to noise stddev
    noise_stddev = np.sqrt(10 ** (-snrdb / 10))
    
    # Get batch size and signal dimensions
    batch_size = tf.shape(x)[0]
    signal_length = tf.shape(x)[1]
    
    # Ensure even length for complex conversion
    signal_length_even = signal_length // 2 * 2
    
    # Process only the even part
    x_processed = x[:, :signal_length_even]
    
    # Convert to complex (first half real, second half imaginary)
    dim_z_half = signal_length_even // 2
    x_real = x_processed[:, :dim_z_half]
    x_imag = x_processed[:, dim_z_half:2*dim_z_half]
    
    # Normalize power
    power = tf.reduce_mean(x_real**2 + x_imag**2, axis=1, keepdims=True)
    scaling_factor = tf.sqrt(tf.cast(dim_z_half, tf.float32) / (power + 1e-8))
    x_real_normalized = x_real * scaling_factor
    x_imag_normalized = x_imag * scaling_factor
    
    x_complex = tf.complex(x_real_normalized, x_imag_normalized)
    
    # Generate Rician fading coefficients
    # LOS component
    h_los_real = tf.ones((batch_size, dim_z_half), dtype=tf.float32) * tf.sqrt(leo_channel.rician_k_linear / (leo_channel.rician_k_linear + 1))
    h_los_imag = tf.zeros((batch_size, dim_z_half), dtype=tf.float32)
    h_los = tf.complex(h_los_real, h_los_imag)
    
    # NLOS component (Rayleigh)
    h_nlos_real = tf.random.normal((batch_size, dim_z_half), 0, 1/tf.sqrt(2*(leo_channel.rician_k_linear + 1)))
    h_nlos_imag = tf.random.normal((batch_size, dim_z_half), 0, 1/tf.sqrt(2*(leo_channel.rician_k_linear + 1)))
    h_nlos = tf.complex(h_nlos_real, h_nlos_imag)
    
    # Combine LOS and NLOS
    h = h_los + h_nlos
    
    # Apply channel
    y_complex = tf.cast(tf.sqrt(leo_channel.G_total), tf.complex64) * h * x_complex
    
    # Add AWGN
    noise_real = tf.random.normal(tf.shape(y_complex), 0, noise_stddev/tf.sqrt(2.0))
    noise_imag = tf.random.normal(tf.shape(y_complex), 0, noise_stddev/tf.sqrt(2.0))
    noise = tf.complex(noise_real, noise_imag)
    
    y_complex_noisy = y_complex + noise
    
    # Convert back to real
    y_real = tf.math.real(y_complex_noisy)
    y_imag = tf.math.imag(y_complex_noisy)
    y_combined = tf.concat([y_real, y_imag], axis=1)
    
    # Pad if necessary using TensorFlow operations
    padding_needed = signal_length - signal_length_even
    y_out = tf.cond(
        padding_needed > 0,
        lambda: tf.concat([y_combined, tf.zeros((batch_size, padding_needed), dtype=tf.float32)], axis=1),
        lambda: y_combined
    )
    
    return y_out

def build_djscc_model_leo(snrdb, blocksize):
    """Build DJSCC model with LEO Rician fading channel"""
    # Create LEO channel instance
    leo_channel = LEOChannel(
        carrier_freq=20e9, orbit_height=600e3, elevation_angle=45,
        rain_rate=5, rician_k=15, antenna_gain_tx=35, antenna_gain_rx=5
    )
    
    input_img = Input(shape=(32, 32, 3))
    num_filters = 64
    conv_depth = blocksize
    
    # Encoder layers
    encoded = tfc.SignalConv2D(
        num_filters, (9, 9), name="layer_0", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True, activation=tfc.GDN(name="gdn_0")
    )(input_img)
    encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
    
    encoded = tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True, activation=tfc.GDN(name="gdn_1")
    )(encoded)
    encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
    
    encoded = tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=True, strides_down=1,
        padding="same_zeros", use_bias=True, activation=tfc.GDN(name="gdn_2")
    )(encoded)
    encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
    
    encoded = tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_3", corr=True, strides_down=1,
        padding="same_zeros", use_bias=True, activation=tfc.GDN(name="gdn_3")
    )(encoded)
    encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
    
    encoded = tfc.SignalConv2D(
        conv_depth, (5, 5), name="layer_out", corr=True, strides_down=1,
        padding="same_zeros", use_bias=True, activation=None
    )(encoded)

    inter_shape = tf.shape(encoded)
    
    # reshape array to [-1, dim_z]
    z = layers.Flatten()(encoded)
    
    dim_z = tf.shape(z)[1]
    # normalize latent vector so that the average power is 1
    z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.nn.l2_normalize(z, axis=1)
    
    # Apply LEO Rician fading channel
    z_out = Lambda(lambda x: rician_fading_channel(x, snrdb, leo_channel))(z_in)
    
    # convert signal back to intermediate shape
    z_out = tf.reshape(z_out, inter_shape)

    # Classifier model
    classifier_input = z_out
    flatten = Flatten()(classifier_input)
    classifier_output = Dense(64, activation='relu')(flatten)
    classifier_output = BatchNormalization()(classifier_output)
    classifier_output = Dropout(0.5)(classifier_output)
    classifier_output = Dense(10, activation='softmax')(classifier_output)  
    
    classifier_model = Model(inputs=input_img, outputs=classifier_output)
    classifier_model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return classifier_model

class LDPCTransmitterLEO:
    '''
    Transmits given bits with LDPC over LEO Rician fading channel.
    '''
    def __init__(self, k, n, m, esno_db):
        '''
        k: data bits per codeword (in LDPC)
        n: total codeword bits (in LDPC)
        m: modulation order (in m-QAM)
        esno_db: channel SNR
        '''
        self.k = k
        self.n = n
        self.num_bits_per_symbol = round(math.log2(m))
        self.esno_db = esno_db
        
        # Create LEO channel
        self.leo_channel = LEOChannel(
            carrier_freq=20e9, orbit_height=600e3, elevation_angle=45,
            rain_rate=5, rician_k=15, antenna_gain_tx=35, antenna_gain_rx=5
        )

        constellation_type = 'qam' if m != 2 else 'pam'
        self.constellation = Constellation(constellation_type, num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper('app', constellation=self.constellation)
        self.encoder = LDPC5GEncoder(k=self.k, n=self.n)
        self.decoder = LDPC5GDecoder(self.encoder, num_iter=20)
    
    def apply_leo_channel_to_symbols(self, symbols):
        """Apply LEO Rician fading channel to symbols"""
        batch_size, num_symbols = symbols.shape
        
        # Generate Rician fading
        h_los_real = tf.ones((batch_size, num_symbols), dtype=tf.float32) * tf.sqrt(self.leo_channel.rician_k_linear / (self.leo_channel.rician_k_linear + 1))
        h_los_imag = tf.zeros((batch_size, num_symbols), dtype=tf.float32)
        h_los = tf.complex(h_los_real, h_los_imag)
        
        h_nlos_real = tf.random.normal((batch_size, num_symbols), 0, 1/tf.sqrt(2*(self.leo_channel.rician_k_linear + 1)))
        h_nlos_imag = tf.random.normal((batch_size, num_symbols), 0, 1/tf.sqrt(2*(self.leo_channel.rician_k_linear + 1)))
        h_nlos = tf.complex(h_nlos_real, h_nlos_imag)
        
        h = h_los + h_nlos
        
        # Apply channel
        y = tf.cast(tf.sqrt(self.leo_channel.G_total), tf.complex64) * h * symbols
        
        # Add noise
        signal_power = tf.reduce_mean(tf.math.abs(y)**2)
        snr_linear = 10**(self.esno_db / 10)
        noise_power = signal_power / snr_linear
        
        noise_stddev = tf.sqrt(noise_power / 2)
        noise_real = tf.random.normal(tf.shape(y), 0, noise_stddev)
        noise_imag = tf.random.normal(tf.shape(y), 0, noise_stddev)
        noise = tf.complex(noise_real, noise_imag)
        
        return y + noise

    def send(self, source_bits):
        '''
        source_bits: float np array of '0' and '1', whose total # of bits is divisible with k
        '''
        lcm = np.lcm(self.k, self.num_bits_per_symbol)
        source_bits_pad = tf.pad(source_bits, [[0, math.ceil(len(source_bits)/lcm)*lcm - len(source_bits)]])
        u = np.reshape(source_bits_pad, (-1, self.k))

        c = self.encoder(u)
        x = self.mapper(c)
        
        # Apply LEO Rician fading channel
        y = self.apply_leo_channel_to_symbols(x)
        
        # For demodulation, use effective SNR
        effective_snr = self.esno_db
        no = ebnodb2no(effective_snr, num_bits_per_symbol=self.num_bits_per_symbol, coderate=self.k/self.n)
        
        llr_ch = self.demapper([y, no])
        u_hat = self.decoder(llr_ch)

        return tf.reshape(u_hat, (-1))[:len(source_bits)]

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

class BPGEncoder():
    def __init__(self, working_directory='./analysis/temp'):
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
        input_dir = f'{self.working_directory}/temp_enc.png'
        output_dir = f'{self.working_directory}/temp_enc.bpg'

        im = Image.fromarray(image_array, 'RGB')
        im.save(input_dir)

        qp = self.get_qp(input_dir, max_bytes + header_bytes, output_dir)
        
        if self.run_bpgenc(qp, input_dir, output_dir) < 0:
            raise RuntimeError("BPG encoding failed")

        return np.unpackbits(np.fromfile(output_dir, dtype=np.uint8)).astype(np.float32)

class BPGDecoder():
    def __init__(self, working_directory='./analysis/temp'):
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
        input_dir = f'{self.working_directory}/temp_dec.bpg'
        output_dir = f'{self.working_directory}/temp_dec.png'

        byte_array = np.packbits(bit_array.astype(np.uint8))
        with open(input_dir, "wb") as binary_file:
            binary_file.write(byte_array.tobytes())

        cifar_mean = np.array([0.4913997551666284, 0.48215855929893703, 0.4465309133731618]) * 255
        cifar_mean = np.reshape(cifar_mean, [1] * (len(image_shape) - 1) + [3]).astype(np.uint8)

        if self.run_bpgdec(input_dir, output_dir) < 0:
            return 0 * np.ones(image_shape) + cifar_mean
        else:
            x = np.array(Image.open(output_dir).convert('RGB'))
            if x.shape != image_shape:
                return 0 * np.ones(image_shape) + cifar_mean
            return x

# Replace the original functions with LEO versions
def train_djscc_leo(train_snrdb, x_train, y_train, x_val, y_val, blocksize):
    model = build_djscc_model_leo(train_snrdb, blocksize)
    early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True, verbose=1)
    history = model.fit(x_train, y_train, epochs=50, batch_size=128, 
                       validation_data=(x_val, y_val), callbacks=[early_stopping], verbose=1)
    model.save_weights('model_weights_leo.h5')
    
    if early_stopping.stopped_epoch != 0:
        print(f"Early stopping occurred at epoch {early_stopping.stopped_epoch}")
    else:
        print("Training completed without early stopping.")
    
    return model, history

def test_djscc_leo(snrdb, x_test, y_test, blocksize):
    model = build_djscc_model_leo(snrdb, blocksize)
    model.load_weights('model_weights_leo.h5')
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return accuracy

def calculate_accuracy_ldpc_leo(bw_ratio, k, n, m, snrs, num_images=50):
    bpgencoder = BPGEncoder()
    bpgdecoder = BPGDecoder()
    acc_values = []

    if isinstance(snrs, (int, float)):
        snrs = [snrs]

    # Load the classification model and train it on the original CIFAR-10 dataset
    classifier_model = build_classifier_model()
    (x_train, y_train), _ = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='max', patience=10, restore_best_weights=True)
    classifier_model.fit(x_train, y_train, batch_size=128, epochs=20, validation_split=0.1, verbose=0, callbacks=[EarlyStopping])
    
    classifier_model.save_weights('classifier_model_weights_ldpc_leo.h5')

    dataset = tfds.load('cifar10', split='test', shuffle_files=False)
    for esno_db in snrs:
        ldpctransmitter = LDPCTransmitterLEO(k, n, m, esno_db)
        decoded_images = []
        original_labels = []
        for example in tqdm(dataset.take(num_images), desc=f"SNR {esno_db}dB"):
            image = example['image'].numpy()
            label = example['label'].numpy()
            image = image[np.newaxis, ...]
            b, _, _, _ = image.shape
            image = tf.cast(imBatchtoImage(image), tf.uint8)
            max_bytes = b * 32 * 32 * 3 * bw_ratio * math.log2(m) * k / n / 8
            src_bits = bpgencoder.encode(image.numpy(), max_bytes)
            rcv_bits = ldpctransmitter.send(src_bits)
            decoded_image = bpgdecoder.decode(rcv_bits.numpy(), image.shape)
            decoded_images.append(decoded_image)
            original_labels.extend([label])

        classifier_model = build_classifier_model()
        classifier_model.load_weights('classifier_model_weights_ldpc_leo.h5')
        decoded_images = np.array(decoded_images)
        original_labels = np.array(original_labels)
        predictions = classifier_model.predict(decoded_images / 255.0, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        acc = np.mean(predicted_labels == original_labels)
        acc_values.append(acc)

        print(f'SNR={esno_db}, Accuracy={acc:.4f}')

    return acc_values

def compare_systems_leo():
    """Compare DJSCC vs LDPC+BPG over LEO channel"""
    print("=== Comparing DJSCC vs LDPC+BPG over LEO Satellite Channel ===\n")
    
    # Train DJSCC at a moderate SNR
    train_snrdb = 10
    blocksize = 64
    print("Training DJSCC with LEO channel...")
    model, history = train_djscc_leo(train_snrdb, x_train, y_train, x_val, y_val, blocksize)
    
    # Test at different SNR values
    snr_values = [0, 5, 10, 15, 20]
    djscc_accuracies = []
    ldpc_accuracies = []
    
    print("\nSNR\tDJSCC\tLDPC+BPG")
    print("-" * 30)
    
    for snr in snr_values:
        # Test DJSCC
        djscc_acc = test_djscc_leo(snr, x_test, y_test, blocksize)
        djscc_accuracies.append(djscc_acc)
        
        # Test LDPC+BPG
        ldpc_acc = calculate_accuracy_ldpc_leo(bw_ratio=0.1, k=1024, n=2048, m=16, snrs=snr, num_images=50)
        ldpc_accuracies.extend(ldpc_acc)
        
        print(f"{snr} dB\t{djscc_acc:.4f}\t{ldpc_acc[0]:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, djscc_accuracies, 'o-', linewidth=2, markersize=8, label='DJSCC with LEO')
    plt.plot(snr_values, ldpc_accuracies, 's-', linewidth=2, markersize=8, label='LDPC+BPG with LEO')
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('Classification Accuracy', fontsize=14)
    plt.title('DJSCC vs LDPC+BPG over LEO Satellite Channel (Rician Fading)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('djscc_vs_ldpc_leo_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary
    print("\n=== Summary ===")
    print(f"DJSCC Average Accuracy: {np.mean(djscc_accuracies):.4f}")
    print(f"LDPC+BPG Average Accuracy: {np.mean(ldpc_accuracies):.4f}")
    
    return djscc_accuracies, ldpc_accuracies

# Run the comparison
if __name__ == "__main__":
    djscc_acc, ldpc_acc = compare_systems_leo()