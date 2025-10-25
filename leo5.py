import os
import cv2
import gc
from tensorflow.keras import backend as K
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

# Set matplotlib for publication-quality figures
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

tf.random.set_seed(3)
np.random.seed(3)

# Configure TensorFlow for better memory management
def configure_tensorflow():
    """Configure TensorFlow for optimal memory usage"""
    print("🔧 Configuring TensorFlow for memory optimization...")
    
    # Limit GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("   ✅ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"   ⚠️ GPU configuration warning: {e}")
    
    # Set TensorFlow to use less memory
    tf.config.optimizer.set_jit(True)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    print("   ✅ TensorFlow optimization settings applied")

configure_tensorflow()

def clear_tf_memory_aggressive():
    """More aggressive TensorFlow memory clearing"""
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Force TensorFlow to release GPU memory
    if tf.config.list_physical_devices('GPU'):
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

def load_and_preprocess_data():
    """Load and preprocess CIFAR-10 data using TFDS"""
    print("📊 Loading CIFAR-10 dataset...")
    # Load training data
    train_ds = tfds.load('cifar10', split='train', as_supervised=True)
    test_ds = tfds.load('cifar10', split='test', as_supervised=True)
    
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    # Preprocess and batch training data
    train_ds = train_ds.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
    
    print("   ✅ Dataset loaded and preprocessed")
    return train_ds, test_ds

def get_test_data_array(num_images=None):
    """Get test data as numpy arrays for LDPC+BPG evaluation"""
    test_ds = tfds.load('cifar10', split='test', as_supervised=True)
    
    test_images = []
    test_labels = []
    
    for image, label in test_ds:
        test_images.append(image.numpy())
        test_labels.append(label.numpy())
        
        if num_images and len(test_images) >= num_images:
            break
    
    test_images = np.array(test_images).astype('float32') / 255.0
    test_labels = np.array(test_labels)
    
    return test_images, test_labels

class LEOChannel:
    def __init__(self, user_id, carrier_freq=20e9, orbit_height=600e3, elevation_angle=30, 
                 rain_rate=10, antenna_gain_tx=30, antenna_gain_rx=25, 
                 rician_k=10, transmission_power_watts=10.0, bandwidth_hz=10e6,
                 noise_temperature_k=500, noise_figure_db=2.0):
        """
        LEO Satellite Channel Model with Rician fading
        """
        self.user_id = user_id
        self.carrier_freq = carrier_freq  # 20 GHz Ka-band (typical for LEO)
        self.orbit_height = orbit_height  # orbit height in meters
        self.elevation_angle = elevation_angle  # 30° (minimum practical elevation)
        self.rain_rate = rain_rate  # mm/h
        self.antenna_gain_tx = antenna_gain_tx  # 30 dBi (realistic for LEO satellite)
        self.antenna_gain_rx = antenna_gain_rx  # 25 dBi (realistic for ground station)
        self.rician_k_linear = 10**(rician_k/10)  # Rician K-factor in linear scale
        self.transmission_power_watts = transmission_power_watts  # 10W (realistic satellite power)
        self.bandwidth_hz = bandwidth_hz  # 10 MHz (typical bandwidth)
        self.noise_temperature_k = noise_temperature_k  # 500K (realistic system temperature)
        self.noise_figure_db = noise_figure_db  # 2 dB (good receiver)
        
        self.R_earth = 6371e3
        self.c = 3e8
        self.k_boltzmann = 1.38064852e-23  # Boltzmann constant
        
        # Calculate base SNR without fading
        self.G_total = self.calculate_aggregate_gain()
        self.noise_power = self.calculate_noise_power()
        self.received_power = self.calculate_received_power()
        self.base_snr_linear = self.calculate_base_snr()
        self.base_snr_db = 10 * np.log10(self.base_snr_linear)
        
        # Apply Rician fading to get current SNR
        self.current_fading_gain = self.generate_rician_fading_gain()
        self.snr_db = self.apply_fading_to_snr(self.base_snr_db)
        self.snr_linear = 10**(self.snr_db / 10)
        
        print(f"   👤 User {user_id}: Orbit={orbit_height/1000:.0f}km, Elevation={elevation_angle:.1f}°, Rain={rain_rate:.1f}mm/h, Base SNR={self.base_snr_db:.2f}dB, Current SNR={self.snr_db:.2f}dB")
        
    def generate_rician_fading_gain(self):
        """
        Generate Rician fading gain according to equation (4) in the paper
        h_i = √(K/(K+1)) * h_LOS + √(1/(K+1)) * h_NLOS
        """
        # Deterministic LOS component
        h_LOS = 1.0
        
        # Random NLOS component - complex Gaussian
        h_NLOS_real = np.random.randn() / np.sqrt(2)
        h_NLOS_imag = np.random.randn() / np.sqrt(2)
        h_NLOS = h_NLOS_real + 1j * h_NLOS_imag
        
        # Combine according to Rician fading formula
        h_i = (np.sqrt(self.rician_k_linear / (self.rician_k_linear + 1)) * h_LOS + 
               np.sqrt(1 / (self.rician_k_linear + 1)) * h_NLOS)
        
        # Return the magnitude (fading gain)
        return np.abs(h_i)
    
    def apply_fading_to_snr(self, snr_db):
        """
        Apply Rician fading effect to SNR
        Returns faded SNR in dB
        """
        # Convert SNR to linear scale, apply fading, then back to dB
        snr_linear = 10**(snr_db / 10)
        faded_snr_linear = snr_linear * (self.current_fading_gain ** 2)
        faded_snr_db = 10 * np.log10(faded_snr_linear)
        
        # Ensure realistic SNR range for satellite communications
        faded_snr_db = max(-5.0, min(20.0, faded_snr_db))
        
        return faded_snr_db
    
    def update_fading(self):
        """
        Update the fading gain with new random values
        Call this to simulate time-varying fading
        """
        self.current_fading_gain = self.generate_rician_fading_gain()
        self.snr_db = self.apply_fading_to_snr(self.base_snr_db)
        self.snr_linear = 10**(self.snr_db / 10)
    
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
        # Cap rain attenuation at realistic levels
        return min(20.0, rain_att)
    
    def calculate_total_path_loss(self):
        fspl = self.calculate_free_space_path_loss()
        rain_att_linear = 10**(self.calculate_rain_attenuation() / 10)
        # Add implementation losses (pointing, polarization, etc.)
        implementation_losses = 10**(2.0 / 10)  # 2 dB implementation margin
        total_pl_linear = fspl * rain_att_linear * implementation_losses
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
    
    def calculate_base_snr(self):
        """Calculate base SNR without fading"""
        return self.received_power / self.noise_power

class MultiUserLEOSystem:
    def __init__(self, num_users=5, transmission_power_watts=10.0, orbit_height=600e3):
        self.num_users = num_users
        self.transmission_power_watts = transmission_power_watts
        self.orbit_height = orbit_height
        self.users = []
        self.setup_users()
        
    def setup_users(self):
        """Setup users with realistic satellite communication parameters"""
        np.random.seed(42)
        
        for i in range(self.num_users):
            # Realistic elevation angles (20° to 60°)
            elevation_angle = np.random.uniform(20, 60)
            
            # Realistic rain rates (0.1mm/h to 25mm/h)
            rain_rate = np.random.uniform(0.1, 25)
            
            # Realistic Rician K factors (5dB to 15dB)
            rician_k = np.random.uniform(5, 15)
            
            # Realistic antenna gains with some variation
            antenna_gain_tx = 30 + np.random.uniform(-2, 2)  # 28-32 dBi
            antenna_gain_rx = 25 + np.random.uniform(-2, 2)  # 23-27 dBi
            
            user_channel = LEOChannel(
                user_id=i+1,
                orbit_height=self.orbit_height,
                elevation_angle=elevation_angle,
                rain_rate=rain_rate,
                rician_k=rician_k,
                antenna_gain_tx=antenna_gain_tx,
                antenna_gain_rx=antenna_gain_rx,
                transmission_power_watts=self.transmission_power_watts
            )
            
            self.users.append(user_channel)
    
    def update_all_fading(self):
        """Update fading for all users to simulate time variation"""
        for user in self.users:
            user.update_fading()
    
    def get_user_snrs(self):
        """Get current SNR values for all users with fading"""
        return [user.snr_db for user in self.users]
    
    def get_user_conditions(self):
        """Get channel conditions for all users"""
        conditions = []
        for user in self.users:
            conditions.append({
                'user_id': user.user_id,
                'orbit_height_km': user.orbit_height / 1000,
                'elevation_angle': user.elevation_angle,
                'rain_rate': user.rain_rate,
                'base_snr_db': user.base_snr_db,
                'current_snr_db': user.snr_db,
                'fading_gain': user.current_fading_gain,
                'rician_k_dB': 10 * np.log10(user.rician_k_linear)
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

def build_djscc_model(blocksize, channel_type='awgn', leo_channel=None, snr_db_train=10.0):
    """Build DJSCC model - same for training and testing, only channel changes"""
    
    input_img = Input(shape=(32, 32, 3))
    num_filters = 64
    conv_depth = blocksize
    
    # Encoder layers - YOUR ORIGINAL ARCHITECTURE
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
    
    # Apply channel - ONLY AWGN for both training and testing
    if channel_type == 'awgn':
        # Training: AWGN with fixed SNR
        z_out = Lambda(lambda x: awgn_channel(x, snr_db_train))(z_in)
    else:
        # Testing: AWGN with calculated SNR from LEO channel
        z_out = Lambda(lambda x: awgn_channel(x, leo_channel.snr_db))(z_in)
    
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
    classifier_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return classifier_model

def build_classifier_model():
    """Use FIRST CODE architecture for LDPC+BPG classification"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

class BPGEncoder():
    def __init__(self, working_directory='./analysis/temp'):
        self.working_directory = working_directory
        os.makedirs(working_directory, exist_ok=True)
    
    def run_bpgenc(self, qp, input_dir, output_dir='temp.bpg'):
        if os.path.exists(output_dir):
            os.remove(output_dir)
        os.system(f'bpgenc {input_dir} -q {qp} -o {output_dir} -f 444 > /dev/null 2>&1')

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
            if bytes == -1:
                return -1
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
        if qp == -1:
            raise RuntimeError("BPG encoding failed")
        
        final_bytes = self.run_bpgenc(qp, input_dir, output_dir)
        if final_bytes < 0:
            raise RuntimeError("BPG encoding failed")

        return np.unpackbits(np.fromfile(output_dir, dtype=np.uint8)).astype(np.float32)

class BPGDecoder():
    def __init__(self, working_directory='./analysis/temp'):
        self.working_directory = working_directory
        os.makedirs(working_directory, exist_ok=True)
    
    def run_bpgdec(self, input_dir, output_dir='temp.png'):
        if os.path.exists(output_dir):
            os.remove(output_dir)
        result = os.system(f'bpgdec {input_dir} -o {output_dir} > /dev/null 2>&1')
        return result == 0

    def decode(self, bit_array, image_shape):
        input_dir = f'{self.working_directory}/temp_dec.bpg'
        output_dir = f'{self.working_directory}/temp_dec.png'

        byte_array = np.packbits(bit_array.astype(np.uint8))
        
        if len(byte_array) < 4:
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
        # Use CIFAR-10 mean instead of uniform 128
        cifar_mean = np.array([0.4913997551666284, 0.48215855929893703, 0.4465309133731618]) * 255
        cifar_mean = np.reshape(cifar_mean, [1, 1, 3]).astype(np.uint8)
        return np.tile(cifar_mean, (image_shape[0], image_shape[1], 1))

class LDPCTransmitterLEO:
    '''
    Transmits given bits with LDPC over AWGN channel using calculated SNR from LEO model.
    '''
    def __init__(self, k, n, m, leo_channel):
        '''
        k: data bits per codeword (in LDPC)
        n: total codeword bits (in LDPC)
        m: modulation order (in m-QAM) - must be power of 2
        leo_channel: LEO channel instance for the user (to get SNR)
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
    
    def send(self, source_bits):
        '''
        source_bits: float np array of '0' and '1', whose total # of bits is divisible with k
        '''
        lcm = np.lcm(self.k, self.num_bits_per_symbol)
        source_bits_pad = tf.pad(source_bits, [[0, math.ceil(len(source_bits)/lcm)*lcm - len(source_bits)]])
        u = np.reshape(source_bits_pad, (-1, self.k))

        c = self.encoder(u)
        x = self.mapper(c)
        
        # Use calculated SNR from LEO channel
        effective_snr = self.leo_channel.snr_db
        no = ebnodb2no(effective_snr, num_bits_per_symbol=1, coderate=1)
        
        # Apply AWGN channel
        channel = AWGN()
        y = channel([x, no])
        
        llr_ch = self.demapper([y, no])
        u_hat = self.decoder(llr_ch)

        return tf.reshape(u_hat, (-1))[:len(source_bits)]

class AgeOfInformationAnalyzer:
    """
    Age of Information (AoI) analyzer for multi-user LEO satellite communication systems
    """
    def __init__(self, lambda_I=1.0, eta_aomi=2.0):
        self.lambda_I = lambda_I  # Information arrival rate
        self.eta_aomi = eta_aomi  # Maximum allowable AoMI threshold
        
        # Constants from the system model
        self.D_enc = 0.01  # Encoding delay (seconds)
        self.T_s = 125e-7    # Symbol duration (seconds)
        self.D_cls_djscc = 0.02  # Classification delay for DJSCC (seconds)
        self.D_cls_trad = 0.03   # Classification delay for traditional method (seconds)
        
        # Image dimensions
        self.I_H, self.I_W, self.I_C = 32, 32, 3  # CIFAR-10 image dimensions
        self.k_P = self.I_H * self.I_W * self.I_C  # Source bandwidth (pixels)
        self.n_con = 16                # From encoder architecture
        self.n_T = (self.n_con * self.k_P) / (16 * self.I_C)  # Channel bandwidth
    
    def calculate_network_aomi(self, users_accuracy_results, method="DJSCC"):
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
            else:  # LDPC+BPG
                rho_k = user_result['ldpc_accuracy']
            
            # Calculate total delay for this user
            D_total_k = self.calculate_total_delay(method)
            
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
    
    def calculate_total_delay(self, method):
        """Calculate total delay D_total^(k) for each method"""
        transmission_time = self.n_T * self.T_s
        
        if method == "DJSCC":
            return self.D_enc + transmission_time + self.D_cls_djscc
        else:  # LDPC+BPG
            return self.D_enc + transmission_time + self.D_cls_trad
    
    def calculate_threshold_compliance_ratio(self, users_accuracy_results, method="DJSCC"):
        """
        Calculate threshold compliance ratio Γ
        Γ = (1/U) * Σ I(α_avg^(k) ≤ η_aomi)
        """
        U = len(users_accuracy_results)
        compliant_users = 0
        
        for user_result in users_accuracy_results:
            # Get classification accuracy for this user
            if method == "DJSCC":
                rho_k = user_result['djscc_accuracy']
            else:  # LDPC+BPG
                rho_k = user_result['ldpc_accuracy']
            
            # Calculate total delay for this user
            D_total_k = self.calculate_total_delay(method)
            
            # Ensure classification accuracy is within valid range
            rho_k = max(0.01, min(0.99, rho_k))
            
            # Calculate individual AAoMI
            term1 = 1 / (self.lambda_I * rho_k)
            term2 = D_total_k / rho_k
            term3 = (self.lambda_I * (D_total_k ** 2)) / (self.lambda_I * D_total_k + 1)
            user_aomi = term1 + term2 + term3
            
            # Check threshold compliance
            if user_aomi <= self.eta_aomi:
                compliant_users += 1
        
        return compliant_users / U if U > 0 else 0.0

# Global variables to store pre-trained models
DJSCC_MODEL = None
CLASSIFIER_MODEL = None

def get_djscc_model(blocksize, channel_type='awgn', leo_channel=None, snr_db_train=10.0):
    """Get DJSCC model with proper memory management"""
    global DJSCC_MODEL
    
    # Clear any existing model
    if DJSCC_MODEL is not None:
        del DJSCC_MODEL
        clear_tf_memory_aggressive()
    
    # Build new model
    DJSCC_MODEL = build_djscc_model(blocksize, channel_type, leo_channel, snr_db_train)
    return DJSCC_MODEL

def get_classifier_model():
    """Get classifier model with proper memory management"""
    global CLASSIFIER_MODEL
    
    if CLASSIFIER_MODEL is None:
        print("   🔄 Loading pre-trained classifier model...")
        CLASSIFIER_MODEL = build_classifier_model()
        CLASSIFIER_MODEL.load_weights('classifier_model_weights_ldpc_tfds.h5')
        print("   ✅ Classifier model loaded")
    
    return CLASSIFIER_MODEL

def train_djscc_awgn(snr_db_train, blocksize):
    """Train DJSCC model with fixed AWGN SNR using TFDS"""
    print(f"🧠 Training DJSCC with fixed SNR: {snr_db_train} dB")
    
    # Load data using TFDS - get the full dataset first
    train_ds_full = tfds.load('cifar10', split='train', as_supervised=True)
    
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    # Preprocess the full dataset
    train_ds_full = train_ds_full.map(preprocess)
    
    # Convert to list and split manually to ensure proper splitting
    full_data = list(train_ds_full.batch(1).as_numpy_iterator())  # Batch size 1 to get individual samples
    total_samples = len(full_data)
    train_size = int(0.9 * total_samples)
    
    # Split the data
    train_data = full_data[:train_size]
    val_data = full_data[train_size:]
    
    print(f"   📈 Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("Empty dataset detected after splitting!")
    
    # Convert back to datasets with proper batching
    def create_dataset_from_list(data_list, batch_size=128):
        images = np.array([item[0][0] for item in data_list])  # Unbatch
        labels = np.array([item[1][0] for item in data_list])
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    train_ds = create_dataset_from_list(train_data)
    val_ds = create_dataset_from_list(val_data)
    
    model = build_djscc_model(blocksize, channel_type='awgn', snr_db_train=snr_db_train)
    early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=20, restore_best_weights=True, verbose=1)
    
    print("   🏋️ Starting DJSCC training...")
    history = model.fit(train_ds, epochs=60, validation_data=val_ds, callbacks=[early_stopping], verbose=1)
    model.save_weights('model_weights_awgn.h5')
    
    if early_stopping.stopped_epoch != 0:
        print(f"   ⏹️ Early stopping occurred at epoch {early_stopping.stopped_epoch}")
    else:
        print("   ✅ Training completed without early stopping.")
    
    return model, history

def test_djscc_user_optimized(leo_channel, test_ds, blocksize):
    """Optimized DJSCC testing with memory management"""
    try:
        # Get model with proper memory handling
        model = get_djscc_model(blocksize, channel_type='leo', leo_channel=leo_channel)
        model.load_weights('model_weights_awgn.h5')
        
        # Evaluate with minimal memory footprint
        _, accuracy = model.evaluate(test_ds, verbose=0, steps=len(test_ds))
        
        return accuracy
    except Exception as e:
        print(f"    ❌ DJSCC test error: {e}")
        return 0.0
    finally:
        # Always clear memory
        clear_tf_memory_aggressive()

def calculate_accuracy_ldpc_user_optimized(bw_ratio, k, n, m, leo_channel, num_images=5):
    """Optimized LDPC+BPG testing with memory management"""
    try:
        # Get classifier model once
        classifier_model = get_classifier_model()
        
        bpgencoder = BPGEncoder()
        bpgdecoder = BPGDecoder()
        
        # Load minimal test data
        test_ds = tfds.load('cifar10', split='test', as_supervised=True)
        test_ds = test_ds.take(num_images)
        
        # Create LDPC transmitter
        ldpctransmitter = LDPCTransmitterLEO(k, n, m, leo_channel)
        
        decoded_images = []
        original_labels = []
        
        successful_images = 0
        for image, label in test_ds:
            image_np = image.numpy()
            label_np = label.numpy()
            
            # Process single image
            image_uint8 = image_np.astype(np.uint8)
            
            # Calculate max bytes
            max_bytes = 32 * 32 * 3 * bw_ratio * math.log2(m) * k / n / 8
            
            try:
                src_bits = bpgencoder.encode(image_uint8, max_bytes)
                rcv_bits = ldpctransmitter.send(src_bits)
                decoded_image = bpgdecoder.decode(rcv_bits.numpy(), image_uint8.shape)
                
                # Check if it's the fallback image
                cifar_mean_uint8 = np.array([125, 123, 114])
                is_fallback = np.allclose(decoded_image, cifar_mean_uint8, atol=10)
                
                if not is_fallback:
                    decoded_images.append(decoded_image)
                    original_labels.append(label_np)
                    successful_images += 1
                    
            except Exception as e:
                continue

        if not decoded_images:
            print(f"  ❌ LDPC: No successful decodes for user {leo_channel.user_id}")
            return 0.0
            
        decoded_images = np.array(decoded_images)
        original_labels = np.array(original_labels)
        
        # Normalize and predict in batches to save memory
        decoded_images_normalized = decoded_images.astype('float32') / 255.0
        
        # Predict in smaller batches
        batch_size = min(4, len(decoded_images_normalized))
        predictions = []
        
        for i in range(0, len(decoded_images_normalized), batch_size):
            batch = decoded_images_normalized[i:i+batch_size]
            batch_pred = classifier_model.predict(batch, verbose=0)
            predictions.append(batch_pred)
            
        predictions = np.vstack(predictions)
        predicted_labels = np.argmax(predictions, axis=1)
        acc = np.mean(predicted_labels == original_labels)
        
        print(f"  ✅ LDPC: {successful_images}/{num_images} successful decodes, accuracy: {acc:.4f}")
        return acc
        
    except Exception as e:
        print(f"    ❌ LDPC test error: {e}")
        return 0.0
    finally:
        # Clear memory after LDPC testing
        clear_tf_memory_aggressive()

def train_classifier_model_tfds():
    """Train classifier model for LDPC+BPG using TFDS (FIRST CODE architecture)"""
    print("🧠 Training classifier model for LDPC+BPG using TFDS...")
    
    # Load data using TFDS
    train_ds, _ = load_and_preprocess_data()
    
    # Build and train model using FIRST CODE architecture
    classifier_model = build_classifier_model()
    
    # Add early stopping with minimal changes
    early_stopping = EarlyStopping(
        monitor='accuracy',  # Monitor training accuracy instead of val
        mode='max', 
        patience=20, 
        restore_best_weights=True,
        verbose=1
    )
    
    # Train with early stopping - keep 50 epochs but can stop early
    print("   🏋️ Starting classifier training...")
    classifier_model.fit(train_ds, epochs=150, callbacks=[early_stopping], verbose=1)
    classifier_model.save_weights('classifier_model_weights_ldpc_tfds.h5')
    print("   ✅ Classifier model trained and saved")
    
    return classifier_model

def evaluate_multi_user_system_optimized(num_users=5, transmission_powers=[10.0, 50.0, 100.0, 200.0], orbit_heights=[400e3, 600e3, 1000e3]):
    """
    Optimized multi-user LEO satellite system evaluation with memory management
    """
    print("🚀 === Optimized Multi-User LEO Satellite System Evaluation ===\n")
    
    blocksize = 16
    snr_db_train = 10.0
    
    # Proven parameters
    bw_ratio = 1/3
    k = 3072
    n = 4608
    m = 4
    
    print(f"📊 Using proven parameters: k={k}, n={n}, Code Rate={k/n:.3f}, BW Ratio={bw_ratio}")
    
    aoi_analyzer = AgeOfInformationAnalyzer(lambda_I=1.0, eta_aomi=2.0)
    
    # Train models ONCE at the beginning
    print("\n🎯 PHASE 1: Model Training")
    print("─" * 50)
    train_djscc_awgn(snr_db_train, blocksize)
    clear_tf_memory_aggressive()
    
    train_classifier_model_tfds()
    clear_tf_memory_aggressive()
    
    # Load test data ONCE
    print("\n🎯 PHASE 2: Data Preparation")
    print("─" * 50)
    _, test_ds_full = load_and_preprocess_data()
    test_ds_eval = test_ds_full.take(50)  # Use smaller subset
    
    # Results storage
    all_results = []
    
    # Reduced number of test images for LDPC/BPG evaluation
    num_images = 5

    print("\n🎯 PHASE 3: Multi-User System Evaluation")
    print("─" * 50)
    
    for orbit_height in orbit_heights:
        print(f"\n🛰️ === Evaluating Orbit Height: {orbit_height/1000:.0f} km ===")
        
        orbit_results = []
        
        for power in transmission_powers:
            print(f"\n⚡ --- Transmission Power: {power}W ---")
            
            # Create multi-user system
            multi_user_system = MultiUserLEOSystem(num_users=num_users, 
                                                 transmission_power_watts=power,
                                                 orbit_height=orbit_height)
            
            # Evaluate each user
            user_results = []
            
            for user_channel in multi_user_system.users:
                print(f"  👤 User {user_channel.user_id}: Elevation={user_channel.elevation_angle:.1f}°, "
                      f"Rain={user_channel.rain_rate:.1f}mm/h, Base SNR={user_channel.base_snr_db:.2f}dB, Current SNR={user_channel.snr_db:.2f}dB")
                
                try:
                    # Test DJSCC with optimized memory
                    djscc_acc = test_djscc_user_optimized(user_channel, test_ds_eval, blocksize)
                    
                    # Test LDPC+BPG with optimized memory
                    ldpc_acc = calculate_accuracy_ldpc_user_optimized(
                        bw_ratio, k, n, m, user_channel, num_images
                    )
                    
                    user_result = {
                        'user_id': user_channel.user_id,
                        'orbit_height_km': orbit_height / 1000,
                        'transmission_power': power,
                        'elevation_angle': user_channel.elevation_angle,
                        'rain_rate': user_channel.rain_rate,
                        'base_snr_db': user_channel.base_snr_db,
                        'snr_db': user_channel.snr_db,
                        'fading_gain': user_channel.current_fading_gain,
                        'djscc_accuracy': djscc_acc,
                        'ldpc_accuracy': ldpc_acc
                    }
                    
                    user_results.append(user_result)
                    print(f"    ✅ DJSCC: {djscc_acc:.4f}, LDPC: {ldpc_acc:.4f}")
                    
                except Exception as e:
                    print(f"    ❌ Error evaluating user {user_channel.user_id}: {e}")
                    clear_tf_memory_aggressive()
                    continue
            
            # Calculate network metrics
            if user_results:
                avg_djscc = np.mean([r['djscc_accuracy'] for r in user_results])
                avg_ldpc = np.mean([r['ldpc_accuracy'] for r in user_results])
                
                network_aomi_djscc = aoi_analyzer.calculate_network_aomi(user_results, "DJSCC")
                network_aomi_ldpc = aoi_analyzer.calculate_network_aomi(user_results, "LDPC+BPG")
                
                compliance_ratio_djscc = aoi_analyzer.calculate_threshold_compliance_ratio(user_results, "DJSCC")
                compliance_ratio_ldpc = aoi_analyzer.calculate_threshold_compliance_ratio(user_results, "LDPC+BPG")
                
                power_result = {
                    'orbit_height_km': orbit_height / 1000,
                    'transmission_power': power,
                    'num_users': len(user_results),
                    'avg_djscc_accuracy': avg_djscc,
                    'avg_ldpc_accuracy': avg_ldpc,
                    'network_aomi_djscc': network_aomi_djscc,
                    'network_aomi_ldpc': network_aomi_ldpc,
                    'compliance_ratio_djscc': compliance_ratio_djscc,
                    'compliance_ratio_ldpc': compliance_ratio_ldpc,
                    'user_details': user_results
                }
                
                orbit_results.append(power_result)
                
                print(f"  📊 Average Accuracy - DJSCC: {avg_djscc:.4f}, LDPC: {avg_ldpc:.4f}")
                print(f"  ⏱️ Network AAoMI - DJSCC: {network_aomi_djscc:.2f}, LDPC: {network_aomi_ldpc:.2f}")
                print(f"  ✅ Compliance Ratio - DJSCC: {compliance_ratio_djscc:.2f}, LDPC: {compliance_ratio_ldpc:.2f}")
            
            # Clear memory after each power iteration
            print(f"  🧹 Cleaning memory after power {power}W evaluation...")
            clear_tf_memory_aggressive()
        
        all_results.extend(orbit_results)
        
        # Clear memory after each orbit height iteration
        print(f"🧹 Cleaning memory after orbit {orbit_height/1000:.0f}km evaluation...")
        clear_tf_memory_aggressive()
    
    # Final cleanup
    print("\n🎯 PHASE 4: Final Cleanup")
    print("─" * 50)
    global DJSCC_MODEL, CLASSIFIER_MODEL
    DJSCC_MODEL = None
    CLASSIFIER_MODEL = None
    clear_tf_memory_aggressive()
    print("✅ All TensorFlow models cleared from memory")
    print("✅ All GPU memory released")
    print("✅ Garbage collection completed")
    
    return all_results

def plot_separate_results(all_results):
    """Plot separate figures for accuracy and AAoMI with different orbit heights"""
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Get unique orbit heights
    orbit_heights = sorted(set([r['orbit_height_km'] for r in all_results]))
    
    # Define markers and colors for different orbit heights
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Plot 1: Average Classification Accuracy
    plt.figure(figsize=(10, 6))
    
    for i, orbit_height in enumerate(orbit_heights):
        orbit_data = [r for r in all_results if r['orbit_height_km'] == orbit_height]
        powers = [r['transmission_power'] for r in orbit_data]
        djscc_acc = [r['avg_djscc_accuracy'] for r in orbit_data]
        ldpc_acc = [r['avg_ldpc_accuracy'] for r in orbit_data]
        
        # DJSCC lines
        plt.plot(powers, djscc_acc, marker=markers[i], color=colors[i], 
                 linewidth=2, markersize=8, 
                 label=f'DJSCC ({orbit_height:.0f} km)')
        
        # LDPC+BPG lines (dashed)
        plt.plot(powers, ldpc_acc, marker=markers[i], color=colors[i], 
                 linewidth=2, markersize=8, linestyle='--',
                 label=f'LDPC+BPG ({orbit_height:.0f} km)')
    
    plt.xlabel('Transmission Power $P_T$ [W]', fontsize=16)
    plt.ylabel('Classification Accuracy $\\rho$', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.tight_layout()
    
    # Save accuracy plots
    plt.savefig('results/accuracy_comparison.png', dpi=500, bbox_inches='tight')
    plt.savefig('results/accuracy_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    # Plot 2: Network AAoMI
    plt.figure(figsize=(10, 6))
    
    for i, orbit_height in enumerate(orbit_heights):
        orbit_data = [r for r in all_results if r['orbit_height_km'] == orbit_height]
        powers = [r['transmission_power'] for r in orbit_data]
        djscc_aomi = [r['network_aomi_djscc'] for r in orbit_data]
        ldpc_aomi = [r['network_aomi_ldpc'] for r in orbit_data]
        
        # DJSCC lines
        plt.plot(powers, djscc_aomi, marker=markers[i], color=colors[i], 
                 linewidth=2, markersize=8, 
                 label=f'DJSCC ({orbit_height:.0f} km)')
        
        # LDPC+BPG lines (dashed)
        plt.plot(powers, ldpc_aomi, marker=markers[i], color=colors[i], 
                 linewidth=2, markersize=8, linestyle='--',
                 label=f'LDPC+BPG ({orbit_height:.0f} km)')
    
    plt.xlabel('Transmission Power $P_T$ [W]', fontsize=16)
    plt.ylabel('Network AAoMI $\\alpha_{\\text{avg}}^{\\text{net}}$ [s]', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 10)
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.tight_layout()
    
    # Save AAoMI plots
    plt.savefig('results/aomi_comparison.png', dpi=500, bbox_inches='tight')
    plt.savefig('results/aomi_comparison.pdf', bbox_inches='tight')
    plt.close()

def plot_threshold_compliance_analysis(all_results):
    """Plot threshold compliance ratio for different orbit heights"""
    # Get unique orbit heights
    orbit_heights = sorted(set([r['orbit_height_km'] for r in all_results]))
    
    # Define markers and colors for different orbit heights
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    for i, orbit_height in enumerate(orbit_heights):
        orbit_data = [r for r in all_results if r['orbit_height_km'] == orbit_height]
        powers = [r['transmission_power'] for r in orbit_data]
        
        # Compliance ratio data
        compliance_djscc = [r['compliance_ratio_djscc'] for r in orbit_data]
        compliance_ldpc = [r['compliance_ratio_ldpc'] for r in orbit_data]
        
        # DJSCC compliance ratio
        plt.plot(powers, compliance_djscc, marker=markers[i], color=colors[i], 
                 linewidth=3, markersize=8, 
                 label=f'DJSCC ({orbit_height:.0f} km)')
        
        # LDPC compliance ratio (dashed)
        plt.plot(powers, compliance_ldpc, marker=markers[i], color=colors[i], 
                 linewidth=3, markersize=8, linestyle='--',
                 label=f'LDPC+BPG ({orbit_height:.0f} km)')
    
    plt.xlabel('Transmission Power $P_T$ [W]', fontsize=16)
    plt.ylabel('Threshold Compliance Ratio $\\Gamma$', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/threshold_compliance_analysis.png', dpi=500, bbox_inches='tight')
    plt.savefig('results/threshold_compliance_analysis.pdf', bbox_inches='tight')
    plt.close()

def save_separate_data_files(all_results):
    """Save separate CSV files for accuracy and AAoMI data"""
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Accuracy data
    accuracy_data = []
    for result in all_results:
        accuracy_data.append({
            'orbit_height_km': result['orbit_height_km'],
            'transmission_power_W': result['transmission_power'],
            'num_users': result['num_users'],
            'djscc_accuracy': result['avg_djscc_accuracy'],
            'ldpc_accuracy': result['avg_ldpc_accuracy']
        })
    
    accuracy_df = pd.DataFrame(accuracy_data)
    accuracy_df.to_csv('results/accuracy_results.csv', index=False)
    
    # AAoMI data
    aomi_data = []
    for result in all_results:
        aomi_data.append({
            'orbit_height_km': result['orbit_height_km'],
            'transmission_power_W': result['transmission_power'],
            'num_users': result['num_users'],
            'djscc_aomi': result['network_aomi_djscc'],
            'ldpc_aomi': result['network_aomi_ldpc']
        })
    
    aomi_df = pd.DataFrame(aomi_data)
    aomi_df.to_csv('results/aomi_results.csv', index=False)
    
    # Threshold compliance data
    threshold_data = []
    for result in all_results:
        threshold_data.append({
            'orbit_height_km': result['orbit_height_km'],
            'transmission_power_W': result['transmission_power'],
            'num_users': result['num_users'],
            'compliance_ratio_djscc': result['compliance_ratio_djscc'],
            'compliance_ratio_ldpc': result['compliance_ratio_ldpc']
        })
    
    threshold_df = pd.DataFrame(threshold_data)
    threshold_df.to_csv('results/threshold_compliance_results.csv', index=False)
    
    print("💾 Separate data files saved:")
    print("   📄 results/accuracy_results.csv")
    print("   📄 results/aomi_results.csv")
    print("   📄 results/threshold_compliance_results.csv")

def print_summary_table(all_results):
    """Print comprehensive summary table"""
    print("\n" + "="*140)
    print("📋 MULTI-USER SYSTEM SUMMARY")
    print("="*140)
    print(f"{'Orbit (km)':<12} {'Power (W)':<12} {'Users':<8} {'DJSCC Acc':<12} {'LDPC Acc':<12} {'DJSCC AAoMI':<14} {'LDPC AAoMI':<14} {'DJSCC Γ':<10} {'LDPC Γ':<10}")
    print("-"*140)
    
    # Group by orbit height for better readability
    orbit_heights = sorted(set([r['orbit_height_km'] for r in all_results]))
    
    for orbit_height in orbit_heights:
        orbit_data = [r for r in all_results if r['orbit_height_km'] == orbit_height]
        print(f"🛰️ ORBIT {orbit_height:.0f} km:")
        
        for result in orbit_data:
            print(f"{'':<12} {result['transmission_power']:<12} {result['num_users']:<8} "
                  f"{result['avg_djscc_accuracy']:.4f}      {result['avg_ldpc_accuracy']:.4f}      "
                  f"{result['network_aomi_djscc']:.2f}        {result['network_aomi_ldpc']:.2f}        "
                  f"{result['compliance_ratio_djscc']:.2f}      {result['compliance_ratio_ldpc']:.2f}")

# Main execution
if __name__ == "__main__":
    start_time = time.time()
    print("🚀 Starting Optimized Multi-User LEO Satellite System Evaluation...")
    print("=" * 70)
    
    # Use smaller test parameters initially
    transmission_powers = [0.1, 0.5, 1, 3, 5, 7, 10, 15, 20, 50, 75, 100]  # Reduced set
    orbit_heights = [400e3, 600e3, 1000e3]  # Reduced set
    num_users = 8 # Reduced number of users
    
    print(f"📊 Evaluation Parameters:")
    print(f"   👥 Number of users: {num_users}")
    print(f"   ⚡ Transmission powers: {transmission_powers}")
    print(f"   🛰️ Orbit heights: {[h/1000 for h in orbit_heights]} km")
    print("=" * 70)
    
    try:
        # Run optimized evaluation
        all_results = evaluate_multi_user_system_optimized(
            num_users=num_users, 
            transmission_powers=transmission_powers,
            orbit_heights=orbit_heights
        )
        
        if all_results:
            print("\n🎯 PHASE 5: Results Processing and Visualization")
            print("─" * 50)
            
            # Plot results and save data
            print("📊 Generating plots...")
            plot_separate_results(all_results)
            plot_threshold_compliance_analysis(all_results)
            
            print("💾 Saving data files...")
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
                    'orbit_height_km': r['orbit_height_km'],
                    'transmission_power': r['transmission_power'],
                    'num_users': r['num_users'],
                    'avg_djscc_accuracy': r['avg_djscc_accuracy'],
                    'avg_ldpc_accuracy': r['avg_ldpc_accuracy'],
                    'network_aomi_djscc': r['network_aomi_djscc'],
                    'network_aomi_ldpc': r['network_aomi_ldpc'],
                    'compliance_ratio_djscc': r['compliance_ratio_djscc'],
                    'compliance_ratio_ldpc': r['compliance_ratio_ldpc']
                }
                for r in all_results
            ])
            summary_df.to_csv('results/multi_user_summary_results.csv', index=False)
            
            # Print summary
            print_summary_table(all_results)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            print("\n🎉 === OPTIMIZED EVALUATION COMPLETED SUCCESSFULLY ===")
            print("=" * 70)
            print("📁 Results saved to:")
            print("   📊 Figures:")
            print("      🖼️ results/accuracy_comparison.png/.pdf")
            print("      🖼️ results/aomi_comparison.png/.pdf") 
            print("      🖼️ results/threshold_compliance_analysis.png/.pdf")
            print("   📄 Data files:")
            print("      📊 results/accuracy_results.csv")
            print("      📊 results/aomi_results.csv")
            print("      📊 results/threshold_compliance_results.csv")
            print("      📊 results/multi_user_detailed_results.csv")
            print("      📊 results/multi_user_summary_results.csv")
            print(f"\n⏱️ Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
            print("✅ All TensorFlow models cleared from memory")
            print("✅ All GPU memory released")
            print("✅ Garbage collection completed")
            print("🎯 Evaluation finished successfully!")
            print("=" * 70)
            
        else:
            print("\n❌ === EVALUATION FAILED - NO RESULTS GENERATED ===")
            
    except Exception as e:
        print(f"\n💥 === CRITICAL ERROR OCCURRED ===")
        print(f"Error: {e}")
        print("Performing emergency cleanup...")
        clear_tf_memory_aggressive()
        print("✅ Emergency cleanup completed")
        raise e