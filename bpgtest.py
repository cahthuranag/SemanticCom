import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

def check_bpg_installation():
    """Check if BPG tools are properly installed"""
    print("=== Checking BPG Installation ===")
    
    # Check if bpgenc exists
    bpgenc_check = os.system('which bpgenc > /dev/null 2>&1')
    if bpgenc_check != 0:
        print("❌ BPG encoder (bpgenc) not found!")
        print("Install with: sudo apt-get install bpgenc")
        return False
    else:
        print("✅ BPG encoder (bpgenc) found")
    
    # Check if bpgdec exists
    bpgdec_check = os.system('which bpgdec > /dev/null 2>&1')
    if bpgdec_check != 0:
        print("❌ BPG decoder (bpgdec) not found!")
        print("Install with: sudo apt-get install bpgdec")
        return False
    else:
        print("✅ BPG decoder (bpgdec) found")
    
    # Test basic BPG functionality
    print("\n=== Testing Basic BPG Functionality ===")
    
    # Create a simple test image
    test_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    test_pil = Image.fromarray(test_image, 'RGB')
    test_pil.save('test_input.png')
    
    # Try to encode with BPG
    encode_result = os.system('bpgenc test_input.png -q 30 -o test_output.bpg -f 444 > /dev/null 2>&1')
    if encode_result != 0:
        print("❌ BPG encoding failed")
        return False
    else:
        print("✅ BPG encoding successful")
    
    # Try to decode with BPG
    decode_result = os.system('bpgdec test_output.bpg -o test_decoded.png > /dev/null 2>&1')
    if decode_result != 0:
        print("❌ BPG decoding failed")
        return False
    else:
        print("✅ BPG decoding successful")
    
    # Clean up test files
    for f in ['test_input.png', 'test_output.bpg', 'test_decoded.png']:
        if os.path.exists(f):
            os.remove(f)
    
    return True

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
                print(f"  Encoding failed at quality {quality}, qp {qp}")
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

        # Save image
        im = Image.fromarray(image_array, 'RGB')
        im.save(input_dir)

        # Get quality parameter
        qp = self.get_qp(input_dir, max_bytes + header_bytes, output_dir)
        if qp == -1:
            raise RuntimeError("BPG encoding failed - could not find valid QP")
        
        # Final encode
        final_bytes = self.run_bpgenc(qp, input_dir, output_dir)
        if final_bytes < 0:
            raise RuntimeError("BPG encoding failed")
        
        print(f"  Encoded to {final_bytes} bytes (max: {max_bytes}), QP: {qp}")

        # Read binary and convert to bits
        with open(output_dir, 'rb') as f:
            binary_data = f.read()
        
        bit_array = np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8)).astype(np.float32)
        
        # Clean up
        if os.path.exists(input_dir):
            os.remove(input_dir)
        if os.path.exists(output_dir):
            os.remove(output_dir)
            
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

        # Convert bits back to bytes
        byte_array = np.packbits(bit_array.astype(np.uint8))
        
        # Ensure we have enough data for a valid BPG file
        if len(byte_array) < 11:  # Minimum BPG header size
            print(f"  Warning: Too few bytes ({len(byte_array)}) for BPG decoding")
            return self.get_default_image(image_shape)
        
        try:
            # Write BPG file
            with open(input_dir, "wb") as binary_file:
                binary_file.write(byte_array.tobytes())

            # Decode
            if self.run_bpgdec(input_dir, output_dir) and os.path.exists(output_dir):
                decoded_img = Image.open(output_dir).convert('RGB')
                decoded_array = np.array(decoded_img)
                
                # Clean up
                if os.path.exists(input_dir):
                    os.remove(input_dir)
                if os.path.exists(output_dir):
                    os.remove(output_dir)
                
                if decoded_array.shape == image_shape:
                    print(f"  Successfully decoded to shape {decoded_array.shape}")
                    return decoded_array
                else:
                    print(f"  Shape mismatch: expected {image_shape}, got {decoded_array.shape}")
                    return self.get_default_image(image_shape)
            else:
                print("  BPG decoding command failed")
                return self.get_default_image(image_shape)
                
        except Exception as e:
            print(f"  BPG decoding error: {e}")
            return self.get_default_image(image_shape)
    
    def get_default_image(self, image_shape):
        """Return a default image when decoding fails"""
        print("  Using fallback image")
        return 128 * np.ones(image_shape, dtype=np.uint8)  # Gray image

def test_bpg_encoder_decoder():
    """Test the BPG encoder and decoder classes"""
    print("\n=== Testing BPG Encoder/Decoder Classes ===")
    
    # Create test image
    test_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    print(f"Test image shape: {test_image.shape}")
    print(f"Test image range: {test_image.min()} - {test_image.max()}")
    
    # Initialize encoder and decoder
    encoder = BPGEncoder()
    decoder = BPGDecoder()
    
    # Test different byte limits
    byte_limits = [500, 1000, 2000, 5000]
    
    for max_bytes in byte_limits:
        print(f"\n--- Testing with max_bytes = {max_bytes} ---")
        
        try:
            # Encode
            encoded_bits = encoder.encode(test_image, max_bytes)
            print(f"Encoded bits: {len(encoded_bits)}")
            print(f"Bit distribution - 0s: {np.sum(encoded_bits == 0)}, 1s: {np.sum(encoded_bits == 1)}")
            
            # Decode
            decoded_image = decoder.decode(encoded_bits, test_image.shape)
            print(f"Decoded image shape: {decoded_image.shape}")
            print(f"Decoded image range: {decoded_image.min()} - {decoded_image.max()}")
            
            # Calculate PSNR
            if decoded_image.shape == test_image.shape:
                psnr = tf.image.psnr(test_image, decoded_image, max_val=255)
                print(f"PSNR: {psnr:.2f} dB")
            else:
                print("Shape mismatch - cannot calculate PSNR")
                
        except Exception as e:
            print(f"Error: {e}")

def test_cifar10_bpg():
    """Test BPG with actual CIFAR-10 images"""
    print("\n=== Testing BPG with CIFAR-10 Images ===")
    
    # Load CIFAR-10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Convert to uint8
    x_test_uint8 = (x_test * 255).astype(np.uint8)
    
    print(f"CIFAR-10 test set: {x_test_uint8.shape}")
    
    # Initialize encoder and decoder
    encoder = BPGEncoder()
    decoder = BPGDecoder()
    
    # Test with a few images
    num_test_images = 3
    max_bytes = 1000  # Conservative limit
    
    for i in range(num_test_images):
        print(f"\n--- CIFAR-10 Image {i+1} ---")
        
        original_image = x_test_uint8[i]
        print(f"Original shape: {original_image.shape}")
        print(f"Label: {y_test[i][0]}")
        
        try:
            # Encode
            encoded_bits = encoder.encode(original_image, max_bytes)
            print(f"Encoded to {len(encoded_bits)} bits")
            
            # Decode
            decoded_image = decoder.decode(encoded_bits, original_image.shape)
            
            # Calculate metrics
            if decoded_image.shape == original_image.shape:
                psnr = float(tf.image.psnr(original_image, decoded_image, max_val=255))
                ssim = float(tf.image.ssim(
                    tf.cast(original_image, tf.float32), 
                    tf.cast(decoded_image, tf.float32), 
                    max_val=255
                ))
                print(f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
                
                # Check if we got the fallback image
                if np.all(decoded_image == 128):
                    print("⚠️  Using fallback (gray) image - BPG decoding may have failed")
                else:
                    print("✅ Successfully encoded and decoded")
            else:
                print(f"Shape mismatch: expected {original_image.shape}, got {decoded_image.shape}")
                
        except Exception as e:
            print(f"Error processing image: {e}")

def main():
    """Main test function"""
    print("BPG Installation and Functionality Test")
    print("=" * 50)
    
    # Check installation
    if not check_bpg_installation():
        print("\n❌ BPG tools are not properly installed. Please install them first.")
        return
    
    # Test basic functionality
    test_bpg_encoder_decoder()
    
    # Test with CIFAR-10
    test_cifar10_bpg()
    
    print("\n" + "=" * 50)
    print("✅ BPG Testing Completed")

if __name__ == "__main__":
    main()