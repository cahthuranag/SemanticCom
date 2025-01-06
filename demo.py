import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
import cv2
import tensorflow_compression as tfc
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import os

class DJSCCDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("DJSCC System Demo")
        self.block_size = 16
        self.model = None
        
        # Configure the window
        self.root.geometry("1200x800")
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add title
        title = ttk.Label(main_frame, text="Deep Joint Source-Channel Coding Demo", 
                         font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Create image display area
        self.create_image_frames(main_frame)
        
        # Create control panel
        self.create_controls(main_frame)
        
        # Load the model
        self.load_model()
        
        # Load initial demo image
        self.load_demo_image()

    def load_model(self):
        try:
            self.model = self.build_model(self.snr_var.get(), self.block_size)
            if os.path.exists('classifier_model_weights_rec_train.h5'):
                self.model.load_weights('classifier_model_weights_rec_train.h5')
                print("Model loaded successfully")
            else:
                print("No saved weights found. Using untrained model.")
        except Exception as e:
            print(f"Error loading model: {e}")
        
    def build_model(self, snrdb, blocksize):
        input_img = Input(shape=(32, 32, 3))
        num_filters = 16
        conv_depth = blocksize

        # Enhanced Encoder layers
        encoded = tfc.SignalConv2D(
            num_filters, (9, 9), name="layer_0",
            corr=True, strides_down=2, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(name="gdn_0")
        )(input_img)
        encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
        
        encoded = tfc.SignalConv2D(
            num_filters, (5, 5), name="layer_1",
            corr=True, strides_down=2, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(name="gdn_1")
        )(encoded)
        encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
        
        encoded = tfc.SignalConv2D(
            num_filters, (5, 5), name="layer_2",
            corr=True, strides_down=1, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(name="gdn_2")
        )(encoded)
        encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
        
        encoded = tfc.SignalConv2D(
            num_filters, (5, 5), name="layer_3",
            corr=True, strides_down=1, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(name="gdn_3")
        )(encoded)
        encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
        
        encoded = tfc.SignalConv2D(
            conv_depth, (5, 5), name="layer_out",
            corr=True, strides_down=1, padding="same_zeros",
            use_bias=True, activation=None
        )(encoded)

        # Channel simulation
        snr_value_db = snrdb
        inter_shape = tf.shape(encoded)
        z = layers.Flatten()(encoded)
        noise_stddev = np.sqrt(10 ** (-snr_value_db / 10))
        
        dim_z = tf.shape(z)[1]
        z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.nn.l2_normalize(z, axis=1)
        z_out = self.real_awgn(z_in, noise_stddev)
        z_out = tf.reshape(z_out, inter_shape)

        # Enhanced Decoder layers
        decoded = tfc.SignalConv2D(
            num_filters, (5, 5), corr=False,
            strides_up=1, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(name="igdn_out", inverse=True)
        )(z_out)
        decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
        
        decoded = tfc.SignalConv2D(
            num_filters, (5, 5), corr=False,
            strides_up=1, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(name="igdn_0", inverse=True)
        )(decoded)
        decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
        
        decoded = tfc.SignalConv2D(
            num_filters, (5, 5), corr=False,
            strides_up=1, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(name="igdn_1", inverse=True)
        )(decoded)
        decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
        
        decoded = tfc.SignalConv2D(
            num_filters, (5, 5), corr=False,
            strides_up=2, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(name="igdn_2", inverse=True)
        )(decoded)
        decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
        
        decoded = tfc.SignalConv2D(
            3, (9, 9), corr=False,
            strides_up=2, padding="same_zeros",
            use_bias=True, activation=tf.nn.sigmoid
        )(decoded)

        model = Model(inputs=input_img, outputs=decoded)
        
        def psnr_metric(x_in, x_out):
            return tf.image.psnr(x_in, x_out, max_val=1.0)

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=[
                tf.keras.metrics.MeanSquaredError(),
                psnr_metric
            ]
        )
        return model

    def real_awgn(self, x, stddev):
        """Implements the real additive white gaussian noise channel."""
        awgn = tf.random.normal(tf.shape(x), 0, stddev, dtype=tf.float32)
        return x + awgn
        
    def create_image_frames(self, parent):
        # Frame for all images
        images_frame = ttk.Frame(parent)
        images_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Input Image
        input_frame = ttk.LabelFrame(images_frame, text="Input Image")
        input_frame.grid(row=0, column=0, padx=10, pady=5)
        self.input_label = ttk.Label(input_frame)
        self.input_label.pack(padx=5, pady=5)
        
        # Encoded Signal
        encoded_frame = ttk.LabelFrame(images_frame, text="Encoded Signal")
        encoded_frame.grid(row=0, column=1, padx=10, pady=5)
        self.encoded_label = ttk.Label(encoded_frame)
        self.encoded_label.pack(padx=5, pady=5)
        
        # Add MSE and PSNR Display
        metrics_frame = ttk.Frame(images_frame)
        metrics_frame.grid(row=1, column=1, pady=5)
        
        self.psnr_var = tk.StringVar(value="PSNR: --")
        self.mse_var = tk.StringVar(value="MSE: --")
        
        ttk.Label(metrics_frame, textvariable=self.psnr_var).pack()
        ttk.Label(metrics_frame, textvariable=self.mse_var).pack()
        
        # Reconstructed Image
        output_frame = ttk.LabelFrame(images_frame, text="Reconstructed Image")
        output_frame.grid(row=0, column=2, padx=10, pady=5)
        self.output_label = ttk.Label(output_frame)
        self.output_label.pack(padx=5, pady=5)

    def create_controls(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=10)
        
        # SNR Control
        ttk.Label(control_frame, text="Channel SNR (dB):").pack(side=tk.LEFT, padx=5)
        self.snr_var = tk.DoubleVar(value=10.0)
        snr_scale = ttk.Scale(
            control_frame, 
            from_=0, to=20,
            variable=self.snr_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.on_snr_change
        )
        snr_scale.pack(side=tk.LEFT, padx=5)
        
        # SNR Value Display
        self.snr_display = ttk.Label(control_frame, text="10.0 dB")
        self.snr_display.pack(side=tk.LEFT, padx=5)
        
        # Load Image Button
        self.load_button = ttk.Button(
            control_frame,
            text="Load Image",
            command=self.load_custom_image
        )
        self.load_button.pack(side=tk.LEFT, padx=20)
        
        # Start Button
        self.start_button = ttk.Button(
            control_frame,
            text="Start Transmission",
            command=self.process_image
        )
        self.start_button.pack(side=tk.LEFT, padx=20)

    def on_snr_change(self, value):
        self.snr_display.configure(text=f"{float(value):.1f} dB")
        
    def load_custom_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if file_path:
            try:
                # Load and preprocess image
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (32, 32))
                img = img.astype('float32') / 255.0
                
                self.current_image = img
                self.display_image(self.input_label, (img * 255).astype(np.uint8))
            except Exception as e:
                print(f"Error loading image: {e}")

    def load_demo_image(self):
        # Load CIFAR-10 dataset for demo
        (_, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
        test_image = x_test[0].astype('float32') / 255.0
        
        # Display the image
        self.current_image = test_image
        self.display_image(self.input_label, (test_image * 255).astype(np.uint8))

    def process_image(self):
        if self.model is None or self.current_image is None:
            print("Model or image not loaded")
            return
            
        self.start_button.state(['disabled'])
        self.load_button.state(['disabled'])
        
        try:
            # Prepare input
            input_image = np.expand_dims(self.current_image, 0)
            
            # Process through model
            output = self.model.predict(input_image)
            
            # Calculate metrics
            psnr = tf.image.psnr(input_image, output, max_val=1.0)
            mse = tf.keras.losses.MSE(input_image, output)
            
            self.psnr_var.set(f"PSNR: {psnr[0]:.2f} dB")
            self.mse_var.set(f"MSE: {mse:.4f}")
            
            # Display output
            output_image = (output[0] * 255).astype(np.uint8)
            self.display_image(self.output_label, output_image)
            
            # Display encoded representation
            encoded_layer_model = Model(
                inputs=self.model.input,
                outputs=self.model.layers[9].output  # Adjusted for new architecture
            )
            encoded = encoded_layer_model.predict(input_image)
            self.display_encoded_signal(encoded[0])
            
        except Exception as e:
            print(f"Error processing image: {e}")
        
        finally:
            self.start_button.state(['!disabled'])
            self.load_button.state(['!disabled'])

    def display_image(self, label, img):
        # Resize for display
        img = cv2.resize(img, (200, 200))
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        label.configure(image=img)
        label.image = img

    def display_encoded_signal(self, encoded):
        # Normalize and display encoded signal
        encoded = np.mean(encoded, axis=-1)
        encoded = ((encoded - encoded.min()) * 255 / 
                  (encoded.max() - encoded.min())).astype(np.uint8)
        encoded = cv2.resize(encoded, (200, 200))
        encoded = cv2.cvtColor(encoded, cv2.COLOR_GRAY2RGB)
        self.display_image(self.encoded_label, encoded)

def main():
    # Enable memory growth for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU(s) found and configured")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("No GPU found. Using CPU.")
    
    root = tk.Tk()
    app = DJSCCDemo(root)
    root.mainloop()

if __name__ == "__main__":
    main()