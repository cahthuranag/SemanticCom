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
import gc
import time
from datetime import datetime

# Set matplotlib to use LaTeX for publication-quality figures
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

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
    def __init__(self, user_id, carrier_freq=20e9, orbit_height=600e3, elevation_angle=30, 
                 rain_rate=10, antenna_gain_tx=50, antenna_gain_rx=40, 
                 rician_k=10, transmission_power_watts=100.0, bandwidth_hz=1e6,
                 noise_temperature_k=290, noise_figure_db=3):
        """
        LEO Satellite Channel Model for individual user with transmission power-based SNR calculation
        """
        self.user_id = user_id
        self.carrier_freq = carrier_freq
        self.orbit_height = orbit_height
        self.elevation_angle = elevation_angle
        self.rain_rate = rain_rate
        self.antenna_gain_tx = antenna_gain_tx
        self.antenna_gain_rx = antenna_gain_rx
        self.rician_k_linear = 10**(rician_k/10)
        self.transmission_power_watts = transmission_power_watts
        self.bandwidth_hz = bandwidth_hz
        self.noise_temperature_k = noise_temperature_k
        self.noise_figure_db = noise_figure_db
        
        self.R_earth = 6371e3
        self.c = 3e8
        self.k_boltzmann = 1.38064852e-23  # Boltzmann constant
        
        self.G_total = self.calculate_aggregate_gain()
        self.noise_power = self.calculate_noise_power()
        self.received_power = self.calculate_received_power()
        self.snr_linear = self.calculate_snr()
        self.snr_db = 10 * np.log10(self.snr_linear)
        
        print(f"User {user_id}: Elevation={elevation_angle:.1f}°, Rain={rain_rate:.1f}mm/h, SNR={self.snr_db:.2f}dB")
        
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
    
    def calculate_noise_power(self):
        """Calculate noise power in watts"""
        # Thermal noise
        thermal_noise = self.k_boltzmann * self.noise_temperature_k * self.bandwidth_hz
        
        # Account for receiver noise figure
        noise_figure_linear = 10**(self.noise_figure_db / 10)
        total_noise_power = thermal_noise * noise_figure_linear
        
        return total_noise_power
    
    def calculate_received_power(self):
        """Calculate received power in watts"""
        received_power = self.transmission_power_watts * self.G_total
        return received_power
    
    def calculate_snr(self):
        """Calculate SNR from transmission power and channel conditions"""
        return self.received_power / self.noise_power

class MultiUserLEOSystem:
    """
    Multi-user LEO satellite system with different channel conditions for each user
    """
    def __init__(self, num_users=5, transmission_power_watts=100.0):
        self.num_users = num_users
        self.transmission_power_watts = transmission_power_watts
        self.users = []
        self.setup_users()
        
    def setup_users(self):
        """Setup users with different elevation angles and rain rates"""
        # Generate random but realistic parameters for each user
        np.random.seed(42)  # For reproducible results
        
        for i in range(self.num_users):
            # Different elevation angles (20° to 80°)
            elevation_angle = np.random.uniform(20, 80)
            
            # Different rain rates (0.1mm/h to 20mm/h)
            rain_rate = np.random.uniform(0.1, 20)
            
            # Different Rician K factors (8dB to 15dB)
            rician_k = np.random.uniform(8, 15)
            
            # Create user channel
            user_channel = LEOChannel(
                user_id=i+1,
                elevation_angle=elevation_angle,
                rain_rate=rain_rate,
                rician_k=rician_k,
                transmission_power_watts=self.transmission_power_watts
            )
            
            self.users.append(user_channel)
    
    def get_user_snrs(self):
        """Get SNR values for all users"""
        return [user.snr_db for user in self.users]
    
    def get_user_conditions(self):
        """Get channel conditions for all users"""
        conditions = []
        for user in self.users:
            conditions.append({
                'user_id': user.user_id,
                'elevation_angle': user.elevation_angle,
                'rain_rate': user.rain_rate,
                'snr_db': user.snr_db
            })
        return conditions

def awgn_channel(x, snr_db):
    """
    AWGN channel for training with fixed SNR
    """
    snr_linear = 10**(snr_db / 10)
    noise_stddev = np.sqrt(1.0 / (2 * snr_linear))
    
    # Add AWGN
    noise = tf.random.normal(tf.shape(x), 0, noise_stddev)
    return x + noise

def rician_fading_channel(x, leo_channel):
    """
    Rician fading channel for testing with LEO satellite parameters
    Uses SNR calculated from transmission power
    """
    # Get SNR from LEO channel
    snr_linear = leo_channel.snr_linear
    noise_stddev = np.sqrt(1.0 / (2 * snr_linear))  # Assuming unit signal power
    
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

def build_djscc_model(blocksize, channel_type='awgn', leo_channel=None, snr_db_train=10.0):
    """Build DJSCC model - same for training and testing, only channel changes"""
    
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
    
    # Apply channel - different for training vs testing
    if channel_type == 'awgn':
        # Training: AWGN with fixed SNR
        z_out = Lambda(lambda x: awgn_channel(x, snr_db_train))(z_in)
    else:
        # Testing: LEO Rician fading with calculated SNR
        z_out = Lambda(lambda x: rician_fading_channel(x, leo_channel))(z_in)
    
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
    Uses transmission power to calculate SNR.
    '''
    def __init__(self, k, n, m, leo_channel):
        '''
        k: data bits per codeword (in LDPC)
        n: total codeword bits (in LDPC)
        m: modulation order (in m-QAM) - must be power of 2
        leo_channel: LEO channel instance for the user
        '''
        self.k = k
        self.n = n
        self.leo_channel = leo_channel
        
        # Ensure m is a power of 2 for Sionna
        if not (m & (m - 1) == 0) and m != 0:
            raise ValueError(f"Modulation order m must be a power of 2, got {m}")
        
        self.num_bits_per_symbol = int(math.log2(m))

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
        
        # Add noise based on calculated SNR
        signal_power = tf.reduce_mean(tf.math.abs(y)**2)
        noise_power = self.leo_channel.noise_power / (signal_power + 1e-8)  # Normalize
        
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
        
        # Use calculated SNR for demodulation
        effective_snr = self.leo_channel.snr_db
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
        os.makedirs(working_directory, exist_ok=True)
    
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
        os.makedirs(working_directory, exist_ok=True)
    
    def run_bpgdec(self, input_dir, output_dir='temp.png'):
        if os.path.exists(output_dir):
            os.remove(output_dir)
        result = os.system(f'bpgdec {input_dir} -o {output_dir} 2>/dev/null')
        return result == 0

    def decode(self, bit_array, image_shape):
        input_dir = f'{self.working_directory}/temp_dec.bpg'
        output_dir = f'{self.working_directory}/temp_dec.png'

        byte_array = np.packbits(bit_array.astype(np.uint8))
        
        # Ensure the BPG file has proper header
        if len(byte_array) < 4:  # Minimum BPG file size
            return self.get_default_image(image_shape)
            
        try:
            with open(input_dir, "wb") as binary_file:
                binary_file.write(byte_array.tobytes())

            if self.run_bpgdec(input_dir, output_dir) and os.path.exists(output_dir):
                x = np.array(Image.open(output_dir).convert('RGB'))
                if x.shape == image_shape:
                    return x
                else:
                    return self.get_default_image(image_shape)
            else:
                return self.get_default_image(image_shape)
                
        except Exception as e:
            return self.get_default_image(image_shape)
    
    def get_default_image(self, image_shape):
        """Return a default image when decoding fails"""
        return 128 * np.ones(image_shape, dtype=np.uint8)  # Gray image

class AgeOfInformationAnalyzer:
    """
    Age of Information (AoI) analyzer for multi-user LEO satellite communication systems
    """
    def __init__(self, lambda_I=1.0, gamma_th=5.0):
        self.lambda_I = lambda_I  # Information arrival rate
        self.gamma_th = gamma_th  # SNR threshold for adaptive method
        
        # Constants from the system model
        self.D_enc = 0.01  # Encoding delay (seconds)
        self.T_s = 1e-6    # Symbol duration (seconds)
        self.D_cls_djscc = 0.02  # Classification delay for DJSCC (seconds)
        self.D_cls_trad = 0.03   # Classification delay for traditional method (seconds)
        
        # Image dimensions
        self.I_H, self.I_W, self.I_C = 32, 32, 3  # CIFAR-10 image dimensions
        self.k_P = self.I_H * self.I_W * self.I_C  # Source bandwidth (pixels)
        self.n_con = 64                 # From encoder architecture
        self.n_T = (self.n_con * self.k_P) / (16 * self.I_C)  # Channel bandwidth
    
    def calculate_network_aomi(self, users_accuracy_results, method="Adaptive"):
        """
        Calculate Network AAoMI for multiple users using actual simulation results
        Equation (24): α_avg^net = (1/U) * Σ [1/(λ_I * ρ_k) + D_total^(k)/ρ_k + (λ_I * (D_total^(k))^2)/(λ_I * D_total^(k) + 1)]
        """
        U = len(users_accuracy_results)
        total_aomi = 0.0
        
        for user_result in users_accuracy_results:
            # Get classification accuracy for this user
            if method == "DJSCC":
                rho_k = user_result['djscc_accuracy']
            elif method == "LDPC+BPG":
                rho_k = user_result['ldpc_accuracy']
            else:  # Adaptive
                rho_k = user_result['adaptive_accuracy']
            
            # Get SNR to determine method for adaptive case
            snr_db = user_result['snr_db']
            
            # Calculate total delay for this user
            D_total_k = self.calculate_total_delay(method, snr_db)
            
            # Ensure classification accuracy is within valid range
            rho_k = max(0.01, min(0.99, rho_k))
            
            # Equation (23): α_avg^(k) = 1/(λ_I * ρ_k) + D_total^(k)/ρ_k + (λ_I * (D_total^(k))^2)/(λ_I * D_total^(k) + 1)
            term1 = 1 / (self.lambda_I * rho_k)
            term2 = D_total_k / rho_k
            term3 = (self.lambda_I * (D_total_k ** 2)) / (self.lambda_I * D_total_k + 1)
            
            user_aomi = term1 + term2 + term3
            total_aomi += user_aomi
        
        # Network AAoMI is the average over all users
        network_aomi = total_aomi / U
        return network_aomi
    
    def calculate_total_delay(self, method, snr_db):
        """Calculate total delay D_total^(k) for each method"""
        transmission_time = self.n_T * self.T_s
        
        if method == "DJSCC":
            return self.D_enc + transmission_time + self.D_cls_djscc
        elif method == "LDPC+BPG":
            return self.D_enc + transmission_time + self.D_cls_trad
        else:  # Adaptive
            if snr_db < self.gamma_th:
                return self.D_enc + transmission_time + self.D_cls_djscc
            else:
                return self.D_enc + transmission_time + self.D_cls_trad

def train_djscc_awgn(snr_db_train, x_train, y_train, x_val, y_val, blocksize):
    """Train DJSCC model with fixed AWGN SNR"""
    model = build_djscc_model(blocksize, channel_type='awgn', snr_db_train=snr_db_train)
    early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True, verbose=1)
    history = model.fit(x_train, y_train, epochs=3, batch_size=128, 
                       validation_data=(x_val, y_val), callbacks=[early_stopping], verbose=1)
    model.save_weights('model_weights_awgn.h5')
    
    if early_stopping.stopped_epoch != 0:
        print(f"Early stopping occurred at epoch {early_stopping.stopped_epoch}")
    else:
        print("Training completed without early stopping.")
    
    return model, history

def test_djscc_user(leo_channel, x_test, y_test, blocksize):
    """Test DJSCC model for a specific user"""
    # Use the same model architecture but with LEO channel
    model = build_djscc_model(blocksize, channel_type='leo', leo_channel=leo_channel)
    model.load_weights('model_weights_awgn.h5')  # Load weights trained with AWGN
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return accuracy

def calculate_accuracy_ldpc_user(bw_ratio, k, n, m, leo_channel, classifier_model, num_images=10):
    """Calculate LDPC+BPG accuracy for a specific user using pre-trained classifier"""
    bpgencoder = BPGEncoder()
    bpgdecoder = BPGDecoder()
    
    dataset = tfds.load('cifar10', split='test', shuffle_files=False)
    decoded_images = []
    original_labels = []
    
    # Create LDPC transmitter for this user
    try:
        ldpctransmitter = LDPCTransmitterLEO(k, n, m, leo_channel)
    except ValueError as e:
        print(f"Error creating LDPC transmitter: {e}")
        ldpctransmitter = LDPCTransmitterLEO(k, n, 4, leo_channel)  # Fallback to QPSK
    
    for example in tqdm(dataset.take(num_images), desc=f"User {leo_channel.user_id}", leave=False):
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

    decoded_images = np.array(decoded_images)
    original_labels = np.array(original_labels)
    predictions = classifier_model.predict(decoded_images / 255.0, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    acc = np.mean(predicted_labels == original_labels)

    return acc

def adaptive_method_user(leo_channel, x_test, y_test, blocksize, classifier_model, bw_ratio=0.05, k=512, n=1024, m=4, num_images=10):
    """
    Adaptive method for a specific user
    """
    snr_db = leo_channel.snr_db
    
    # Choose method based on SNR threshold
    if snr_db < 5.0:
        accuracy = test_djscc_user(leo_channel, x_test, y_test, blocksize)
        method_used = "DJSCC"
    else:
        accuracy = calculate_accuracy_ldpc_user(bw_ratio, k, n, m, leo_channel, classifier_model, num_images)
        method_used = "LDPC+BPG"
    
    return accuracy, method_used

def evaluate_multi_user_system(num_users=5, transmission_powers=[10.0, 50.0, 100.0, 200.0]):
    """
    Evaluate multi-user LEO satellite system with different transmission powers
    """
    print("=== Multi-User LEO Satellite System Evaluation ===\n")
    
    blocksize = 32
    snr_db_train = 10.0
    
    # Initialize AoI analyzer
    aoi_analyzer = AgeOfInformationAnalyzer(lambda_I=1.0, gamma_th=5.0)
    
    # Train models first (ONCE for all users)
    print("Training DJSCC model with fixed AWGN SNR...")
    train_djscc_awgn(snr_db_train, x_train, y_train, x_val, y_val, blocksize)
    
    print("Training classifier model for LDPC+BPG...")
    classifier_model = build_classifier_model()
    (x_train_full, y_train_full), _ = cifar10.load_data()
    x_train_full = x_train_full.astype('float32') / 255.0
    early_stopping = EarlyStopping(monitor='accuracy', mode='max', patience=5, restore_best_weights=True)
    classifier_model.fit(x_train_full, y_train_full, batch_size=128, epochs=10, 
                        validation_split=0.1, verbose=0, callbacks=[early_stopping])
    classifier_model.save_weights('classifier_model_weights_ldpc_leo.h5')
    
    # Load the trained classifier model for LDPC+BPG
    classifier_model_ldpc = build_classifier_model()
    classifier_model_ldpc.load_weights('classifier_model_weights_ldpc_leo.h5')
    
    # Results storage
    all_results = []
    
    for power in transmission_powers:
        print(f"\n--- Evaluating Transmission Power: {power}W ---")
        
        # Create multi-user system for this power
        multi_user_system = MultiUserLEOSystem(num_users=num_users, transmission_power_watts=power)
        
        # Evaluate each user
        user_results = []
        
        for user_channel in multi_user_system.users:
            print(f"  User {user_channel.user_id}: Elevation={user_channel.elevation_angle:.1f}°, "
                  f"Rain={user_channel.rain_rate:.1f}mm/h, SNR={user_channel.snr_db:.2f}dB")
            
            try:
                # Test DJSCC
                djscc_acc = test_djscc_user(user_channel, x_test[:200], y_test[:200], blocksize)
                
                # Test LDPC+BPG using the same pre-trained classifier
                ldpc_acc = calculate_accuracy_ldpc_user(0.05, 512, 1024, 4, user_channel, classifier_model_ldpc, 5)
                
                # Test Adaptive using the same pre-trained classifier
                adaptive_acc, method_used = adaptive_method_user(user_channel, x_test[:200], y_test[:200], 
                                                               blocksize, classifier_model_ldpc, 0.05, 512, 1024, 4, 5)
                
                user_result = {
                    'user_id': user_channel.user_id,
                    'transmission_power': power,
                    'elevation_angle': user_channel.elevation_angle,
                    'rain_rate': user_channel.rain_rate,
                    'snr_db': user_channel.snr_db,
                    'djscc_accuracy': djscc_acc,
                    'ldpc_accuracy': ldpc_acc,
                    'adaptive_accuracy': adaptive_acc,
                    'adaptive_method': method_used
                }
                
                user_results.append(user_result)
                print(f"    DJSCC: {djscc_acc:.4f}, LDPC: {ldpc_acc:.4f}, Adaptive: {adaptive_acc:.4f} ({method_used})")
                
            except Exception as e:
                print(f"    Error evaluating user {user_channel.user_id}: {e}")
                continue
        
        # Calculate network metrics
        if user_results:
            # Average accuracy across users
            avg_djscc = np.mean([r['djscc_accuracy'] for r in user_results])
            avg_ldpc = np.mean([r['ldpc_accuracy'] for r in user_results])
            avg_adaptive = np.mean([r['adaptive_accuracy'] for r in user_results])
            
            # Network AAoMI
            network_aomi_djscc = aoi_analyzer.calculate_network_aomi(user_results, "DJSCC")
            network_aomi_ldpc = aoi_analyzer.calculate_network_aomi(user_results, "LDPC+BPG")
            network_aomi_adaptive = aoi_analyzer.calculate_network_aomi(user_results, "Adaptive")
            
            power_result = {
                'transmission_power': power,
                'num_users': len(user_results),
                'avg_djscc_accuracy': avg_djscc,
                'avg_ldpc_accuracy': avg_ldpc,
                'avg_adaptive_accuracy': avg_adaptive,
                'network_aomi_djscc': network_aomi_djscc,
                'network_aomi_ldpc': network_aomi_ldpc,
                'network_aomi_adaptive': network_aomi_adaptive,
                'user_details': user_results
            }
            
            all_results.append(power_result)
            
            print(f"  Average Accuracy - DJSCC: {avg_djscc:.4f}, LDPC: {avg_ldpc:.4f}, Adaptive: {avg_adaptive:.4f}")
            print(f"  Network AAoMI - DJSCC: {network_aomi_djscc:.2f}, LDPC: {network_aomi_ldpc:.2f}, Adaptive: {network_aomi_adaptive:.2f}")
    
    return all_results

def plot_separate_results(all_results):
    """Plot separate figures for accuracy and AAoMI"""
    # Extract data for plotting
    powers = [r['transmission_power'] for r in all_results]
    djscc_acc = [r['avg_djscc_accuracy'] for r in all_results]
    ldpc_acc = [r['avg_ldpc_accuracy'] for r in all_results]
    adaptive_acc = [r['avg_adaptive_accuracy'] for r in all_results]
    
    djscc_aomi = [r['network_aomi_djscc'] for r in all_results]
    ldpc_aomi = [r['network_aomi_ldpc'] for r in all_results]
    adaptive_aomi = [r['network_aomi_adaptive'] for r in all_results]
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Plot 1: Average Classification Accuracy (Separate File)
    plt.figure(figsize=(8, 6))
    plt.plot(powers, djscc_acc, 'o-', linewidth=2, markersize=8, label='DJSCC')
    plt.plot(powers, ldpc_acc, 's-', linewidth=2, markersize=8, label='LDPC+BPG')
    plt.plot(powers, adaptive_acc, '^-', linewidth=2, markersize=8, label='Adaptive')
    
    plt.xlabel('Transmission Power $P_T$ (W)', fontsize=14)
    plt.ylabel('Average Classification Accuracy', fontsize=14)
    plt.title('Multi-User Average Classification Accuracy', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.tight_layout()
    
    # Save accuracy plots
    plt.savefig('results/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/accuracy_comparison.eps', format='eps', bbox_inches='tight')
    plt.close()
    
    # Plot 2: Network AAoMI (Separate File)
    plt.figure(figsize=(8, 6))
    plt.plot(powers, djscc_aomi, 'o-', linewidth=2, markersize=8, label='DJSCC')
    plt.plot(powers, ldpc_aomi, 's-', linewidth=2, markersize=8, label='LDPC+BPG')
    plt.plot(powers, adaptive_aomi, '^-', linewidth=2, markersize=8, label='Adaptive')
    
    plt.xlabel('Transmission Power $P_T$ (W)', fontsize=14)
    plt.ylabel('Network AAoMI $\\alpha_{\\text{avg}}^{\\text{net}}$', fontsize=14)
    plt.title('Multi-User Network AAoMI', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.tight_layout()
    
    # Save AAoMI plots
    plt.savefig('results/aomi_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/aomi_comparison.eps', format='eps', bbox_inches='tight')
    plt.close()
    
    # Also save combined plot for reference
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Combined plot - Accuracy
    ax1.plot(powers, djscc_acc, 'o-', linewidth=2, markersize=8, label='DJSCC')
    ax1.plot(powers, ldpc_acc, 's-', linewidth=2, markersize=8, label='LDPC+BPG')
    ax1.plot(powers, adaptive_acc, '^-', linewidth=2, markersize=8, label='Adaptive')
    ax1.set_xlabel('Transmission Power $P_T$ (W)', fontsize=14)
    ax1.set_ylabel('Average Classification Accuracy', fontsize=14)
    ax1.set_title('Multi-User Average Classification Accuracy', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.set_xscale('log')
    
    # Combined plot - AAoMI
    ax2.plot(powers, djscc_aomi, 'o-', linewidth=2, markersize=8, label='DJSCC')
    ax2.plot(powers, ldpc_aomi, 's-', linewidth=2, markersize=8, label='LDPC+BPG')
    ax2.plot(powers, adaptive_aomi, '^-', linewidth=2, markersize=8, label='Adaptive')
    ax2.set_xlabel('Transmission Power $P_T$ (W)', fontsize=14)
    ax2.set_ylabel('Network AAoMI $\\alpha_{\\text{avg}}^{\\text{net}}$', fontsize=14)
    ax2.set_title('Multi-User Network AAoMI', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('results/combined_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/combined_results.eps', format='eps', bbox_inches='tight')
    plt.close()

def save_separate_data_files(all_results):
    """Save separate CSV files for accuracy and AAoMI data"""
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Accuracy data
    accuracy_data = []
    for result in all_results:
        accuracy_data.append({
            'transmission_power_W': result['transmission_power'],
            'num_users': result['num_users'],
            'djscc_accuracy': result['avg_djscc_accuracy'],
            'ldpc_accuracy': result['avg_ldpc_accuracy'],
            'adaptive_accuracy': result['avg_adaptive_accuracy']
        })
    
    accuracy_df = pd.DataFrame(accuracy_data)
    accuracy_df.to_csv('results/accuracy_results.csv', index=False)
    
    # AAoMI data
    aomi_data = []
    for result in all_results:
        aomi_data.append({
            'transmission_power_W': result['transmission_power'],
            'num_users': result['num_users'],
            'djscc_aomi': result['network_aomi_djscc'],
            'ldpc_aomi': result['network_aomi_ldpc'],
            'adaptive_aomi': result['network_aomi_adaptive']
        })
    
    aomi_df = pd.DataFrame(aomi_data)
    aomi_df.to_csv('results/aomi_results.csv', index=False)
    
    print("Separate data files saved:")
    print("  - results/accuracy_results.csv")
    print("  - results/aomi_results.csv")

def print_summary_table(all_results):
    """Print comprehensive summary table"""
    print("\n" + "="*100)
    print("MULTI-USER SYSTEM SUMMARY")
    print("="*100)
    print(f"{'Power (W)':<12} {'Users':<8} {'DJSCC Acc':<12} {'LDPC Acc':<12} {'Adaptive Acc':<14} {'DJSCC AAoMI':<14} {'LDPC AAoMI':<14} {'Adaptive AAoMI':<16}")
    print("-"*100)
    
    for result in all_results:
        print(f"{result['transmission_power']:<12} {result['num_users']:<8} "
              f"{result['avg_djscc_accuracy']:.4f}      {result['avg_ldpc_accuracy']:.4f}      "
              f"{result['avg_adaptive_accuracy']:.4f}        {result['network_aomi_djscc']:.2f}        "
              f"{result['network_aomi_ldpc']:.2f}        {result['network_aomi_adaptive']:.2f}")
    
    # Calculate overall improvements
    if len(all_results) > 0:
        avg_adaptive_acc = np.mean([r['avg_adaptive_accuracy'] for r in all_results])
        avg_adaptive_aomi = np.mean([r['network_aomi_adaptive'] for r in all_results])
        avg_djscc_aomi = np.mean([r['network_aomi_djscc'] for r in all_results])
        avg_ldpc_aomi = np.mean([r['network_aomi_ldpc'] for r in all_results])
        
        print("\nOVERALL PERFORMANCE:")
        print(f"Average Adaptive Accuracy: {avg_adaptive_acc:.4f}")
        print(f"Average Adaptive AAoMI: {avg_adaptive_aomi:.2f}")
        if avg_djscc_aomi > 0:
            print(f"AAoMI Improvement vs DJSCC: {((avg_djscc_aomi - avg_adaptive_aomi) / avg_djscc_aomi * 100):.1f}%")
        if avg_ldpc_aomi > 0:
            print(f"AAoMI Improvement vs LDPC: {((avg_ldpc_aomi - avg_adaptive_aomi) / avg_ldpc_aomi * 100):.1f}%")

# Main execution
if __name__ == "__main__":
    # Run multi-user evaluation
    print("Starting Multi-User LEO Satellite System Evaluation...")
    
    # Define transmission powers to test (more realistic for satellites)
    transmission_powers = [10.0, 50.0, 100.0, 200.0]
    num_users = 5
    
    # Run evaluation
    all_results = evaluate_multi_user_system(num_users=num_users, transmission_powers=transmission_powers)
    
    if all_results:
        # Plot separate results
        plot_separate_results(all_results)
        
        # Save separate data files
        save_separate_data_files(all_results)
        
        # Save detailed results
        detailed_results = []
        for power_result in all_results:
            for user_result in power_result['user_details']:
                detailed_results.append(user_result)
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv('results/multi_user_detailed_results.csv', index=False)
        
        summary_df = pd.DataFrame([
            {
                'transmission_power': r['transmission_power'],
                'num_users': r['num_users'],
                'avg_djscc_accuracy': r['avg_djscc_accuracy'],
                'avg_ldpc_accuracy': r['avg_ldpc_accuracy'],
                'avg_adaptive_accuracy': r['avg_adaptive_accuracy'],
                'network_aomi_djscc': r['network_aomi_djscc'],
                'network_aomi_ldpc': r['network_aomi_ldpc'],
                'network_aomi_adaptive': r['network_aomi_adaptive']
            }
            for r in all_results
        ])
        summary_df.to_csv('results/multi_user_summary_results.csv', index=False)
        
        print("\n=== Multi-User Evaluation Completed Successfully ===")
        print("Results saved to:")
        print("  Figures:")
        print("    - results/accuracy_comparison.png/.eps")
        print("    - results/aomi_comparison.png/.eps") 
        print("    - results/combined_results.png/.eps")
        print("  Data files:")
        print("    - results/accuracy_results.csv")
        print("    - results/aomi_results.csv")
        print("    - results/multi_user_detailed_results.csv")
        print("    - results/multi_user_summary_results.csv")
        
        # Print summary
        print_summary_table(all_results)
    else:
        print("\n=== Evaluation Failed - No Results Generated ===")