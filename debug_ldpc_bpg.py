import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow_datasets as tfds
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.utils import ebnodb2no
from sionna.channel import AWGN
import math
from PIL import Image

# Import your BPG classes (use the ones from bpgtest.py since they work)
class BPGEncoder():
    def __init__(self, working_directory='./temp'):
        self.working_directory = working_directory
        os.makedirs(working_directory, exist_ok=True)
    
    def run_bpgenc(self, qp, input_dir, output_dir='temp.bpg'):
        if os.path.exists(output_dir):
            os.remove(output_dir)
        cmd = f'bpgenc {input_dir} -q {qp} -o {output_dir} -f 444'
        result = os.system(cmd)
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

        with open(output_dir, 'rb') as f:
            binary_data = f.read()
        
        bit_array = np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8)).astype(np.float32)
        
        # Clean up
        for f in [input_dir, output_dir]:
            if os.path.exists(f):
                os.remove(f)
                
        return bit_array

class BPGDecoder():
    def __init__(self, working_directory='./temp'):
        self.working_directory = working_directory
        os.makedirs(working_directory, exist_ok=True)
    
    def run_bpgdec(self, input_dir, output_dir='temp.png'):
        if os.path.exists(output_dir):
            os.remove(output_dir)
        cmd = f'bpgdec {input_dir} -o {output_dir}'
        result = os.system(cmd)
        return os.path.exists(output_dir)

    def decode(self, bit_array, image_shape):
        input_dir = f'{self.working_directory}/temp_dec.bpg'
        output_dir = f'{self.working_directory}/temp_dec.png'

        byte_array = np.packbits(bit_array.astype(np.uint8))
        
        if len(byte_array) < 11:
            return self.get_default_image(image_shape)
        
        try:
            with open(input_dir, "wb") as binary_file:
                binary_file.write(byte_array.tobytes())

            if self.run_bpgdec(input_dir, output_dir) and os.path.exists(output_dir):
                decoded_img = Image.open(output_dir).convert('RGB')
                decoded_array = np.array(decoded_img)
                
                # Clean up
                for f in [input_dir, output_dir]:
                    if os.path.exists(f):
                        os.remove(f)
                
                if decoded_array.shape == image_shape:
                    return decoded_array
                else:
                    return self.get_default_image(image_shape)
            else:
                return self.get_default_image(image_shape)
                
        except Exception as e:
            return self.get_default_image(image_shape)
    
    def get_default_image(self, image_shape):
        return 128 * np.ones(image_shape, dtype=np.uint8)

def imBatchtoImage(batch_images):
    batch, h, w, c = batch_images.shape
    divisor = batch
    while batch % divisor != 0:
        divisor -= 1
    
    image = tf.reshape(batch_images, (-1, batch//divisor, h, w, c))
    image = tf.transpose(image, [0, 2, 1, 3, 4])
    image = tf.reshape(image, (-1, batch//divisor*w, c))
    return image

def debug_ldpc_bpg_pipeline():
    """Debug the complete LDPC+BPG pipeline step by step"""
    print("=== Debugging LDPC+BPG Pipeline ===")
    
    # Parameters from your code
    bw_ratio = 0.05
    k = 512
    n = 1024
    m = 4
    snr_db = 10  # Use a reasonable SNR
    
    print(f"Parameters: bw_ratio={bw_ratio}, k={k}, n={n}, m={m}, SNR={snr_db}dB")
    
    # Initialize components
    bpgencoder = BPGEncoder()
    bpgdecoder = BPGDecoder()
    
    # Use simple AWGN LDPC transmitter for testing
    class SimpleLDPCTransmitter:
        def __init__(self, k, n, m, esno_db):
            self.k = k
            self.n = n
            self.num_bits_per_symbol = int(math.log2(m))
            self.constellation = Constellation('qam', num_bits_per_symbol=self.num_bits_per_symbol)
            self.mapper = Mapper(constellation=self.constellation)
            self.demapper = Demapper('app', constellation=self.constellation)
            self.channel = AWGN()
            self.encoder = LDPC5GEncoder(k=self.k, n=self.n)
            self.decoder = LDPC5GDecoder(self.encoder, num_iter=20)
            self.esno_db = esno_db
        
        def send(self, source_bits):
            # Pad to multiple of k
            padding = (self.k - len(source_bits) % self.k) % self.k
            source_bits_pad = np.pad(source_bits, (0, padding))
            u = np.reshape(source_bits_pad, (-1, self.k))
            
            # Encode and transmit
            no = ebnodb2no(self.esno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=self.k/self.n)
            c = self.encoder(u)
            x = self.mapper(c)
            y = self.channel([x, no])
            llr_ch = self.demapper([y, no])
            u_hat = self.decoder(llr_ch)
            
            # Remove padding
            return tf.reshape(u_hat, (-1))[:len(source_bits)]
    
    ldpctransmitter = SimpleLDPCTransmitter(k, n, m, snr_db)
    
    # Load test dataset
    test_dataset = tfds.load('cifar10', split='test', shuffle_files=False)
    
    # Test with a few images
    bit_error_rates = []
    successful_decodes = 0
    total_images = 3
    
    for i, example in enumerate(test_dataset.take(total_images)):
        print(f"\n--- Processing Image {i+1} ---")
        
        # Original image
        image = example['image'].numpy()
        label = example['label'].numpy()
        print(f"Original image - shape: {image.shape}, label: {label}")
        
        # Prepare for BPG
        image_batch = image[np.newaxis, ...]  # Add batch dimension
        b, h, w, c = image_batch.shape
        
        # Convert to format for BPG
        image_for_bpg = tf.cast(imBatchtoImage(image_batch), tf.uint8)
        print(f"BPG input - shape: {image_for_bpg.shape}")
        
        # Calculate max bytes
        max_bytes = b * 32 * 32 * 3 * bw_ratio * math.log2(m) * k / n / 8
        print(f"Max bytes allowed: {max_bytes:.2f}")
        
        try:
            # Step 1: BPG Encoding
            print("1. BPG Encoding...")
            src_bits = bpgencoder.encode(image_for_bpg.numpy(), max_bytes)
            print(f"   Encoded {len(src_bits)} bits")
            print(f"   Bit distribution - 0s: {np.sum(src_bits == 0)}, 1s: {np.sum(src_bits == 1)}")
            
            # Step 2: LDPC Transmission
            print("2. LDPC Transmission...")
            rcv_bits = ldpctransmitter.send(src_bits)
            print(f"   Received {len(rcv_bits)} bits")
            
            # Calculate bit error rate
            bit_errors = np.sum(src_bits != rcv_bits.numpy())
            ber = bit_errors / len(src_bits)
            bit_error_rates.append(ber)
            print(f"   Bit errors: {bit_errors}/{len(src_bits)} (BER: {ber:.6f})")
            
            # Step 3: BPG Decoding
            print("3. BPG Decoding...")
            decoded_image = bpgdecoder.decode(rcv_bits.numpy(), image_for_bpg.shape)
            print(f"   Decoded image shape: {decoded_image.shape}")
            
            # Check if decoding was successful
            if np.all(decoded_image == 128):  # Fallback image
                print("   ⚠️ BPG decoding failed - using fallback")
            else:
                print("   ✅ BPG decoding successful")
                successful_decodes += 1
                
                # Calculate image quality
                psnr = float(tf.image.psnr(image_for_bpg, decoded_image, max_val=255))
                print(f"   PSNR: {psnr:.2f} dB")
                
        except Exception as e:
            print(f"   ❌ Error in pipeline: {e}")
            bit_error_rates.append(1.0)  # Maximum error
    
    # Summary
    print(f"\n=== Pipeline Summary ===")
    print(f"Successful BPG decodes: {successful_decodes}/{total_images}")
    if bit_error_rates:
        avg_ber = np.mean(bit_error_rates)
        print(f"Average Bit Error Rate: {avg_ber:.6f}")
        print(f"Transmission Success Rate: {(1-avg_ber)*100:.2f}%")
    
    return successful_decodes > 0

def test_classifier_on_bpg_images():
    """Test if the classifier can recognize BPG-decoded images"""
    print("\n=== Testing Classifier on BPG Images ===")
    
    # Simple classifier for testing
    def build_simple_classifier():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    # Load and preprocess CIFAR-10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    # Train a simple classifier
    print("Training classifier...")
    classifier = build_simple_classifier()
    classifier.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=1)
    
    # Test on original images
    original_accuracy = classifier.evaluate(x_test[:100], y_test[:100], verbose=0)[1]
    print(f"Classifier accuracy on original images: {original_accuracy:.4f}")
    
    # Test on BPG compressed-decompressed images (no transmission)
    bpgencoder = BPGEncoder()
    bpgdecoder = BPGDecoder()
    
    bpg_images = []
    bpg_labels = []
    
    for i in range(10):  # Test on 10 images
        original_image = (x_test[i] * 255).astype(np.uint8)
        
        # BPG encode and decode (no transmission errors)
        encoded_bits = bpgencoder.encode(original_image, 2000)  # Generous byte limit
        decoded_image = bpgdecoder.decode(encoded_bits, original_image.shape)
        
        if not np.all(decoded_image == 128):  # Skip fallback images
            bpg_images.append(decoded_image / 255.0)
            bpg_labels.append(y_test[i])
    
    if bpg_images:
        bpg_images = np.array(bpg_images)
        bpg_labels = np.array(bpg_labels)
        bpg_accuracy = classifier.evaluate(bpg_images, bpg_labels, verbose=0)[1]
        print(f"Classifier accuracy on BPG images (no transmission): {bpg_accuracy:.4f}")
    else:
        print("No successful BPG decodes for classifier test")

def main():
    """Main debug function"""
    print("LDPC+BPG Pipeline Debug")
    print("=" * 50)
    
    # Test 1: LDPC+BPG pipeline
    pipeline_works = debug_ldpc_bpg_pipeline()
    
    # Test 2: Classifier on BPG images
    test_classifier_on_bpg_images()
    
    print("\n" + "=" * 50)
    if pipeline_works:
        print("✅ LDPC+BPG pipeline is working")
        print("The issue might be in your SNR values or channel model")
    else:
        print("❌ LDPC+BPG pipeline has issues")
        print("Check the bit error rates and BPG decoding")

if __name__ == "__main__":
    main()