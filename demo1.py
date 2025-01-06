import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageTk
import cv2
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow_compression as tfc
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import os

def real_awgn(x, stddev):
    """Implements the real additive white gaussian noise channel."""
    awgn = tf.random.normal(tf.shape(x), 0, stddev, dtype=tf.float32)
    y = x + awgn
    return y

def build_model(snrdb, blocksize):
    input_img = Input(shape=(32, 32, 3))
    num_filters = 16
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

    # Channel simulation
    inter_shape = tf.shape(encoded)
    z = layers.Flatten()(encoded)
    noise_stddev = np.sqrt(10 ** (-snrdb / 10))
    dim_z = tf.shape(z)[1]
    z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.nn.l2_normalize(z, axis=1)
    z_out = real_awgn(z_in, noise_stddev)
    z_out = tf.reshape(z_out, inter_shape)

    # Decoder layers
    decoded = tfc.SignalConv2D(
        num_filters, (5, 5), corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_out", inverse=True)
    )(z_out)
    decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
    
    decoded = tfc.SignalConv2D(
        num_filters, (5, 5), corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_0", inverse=True)
    )(decoded)
    decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
    
    decoded = tfc.SignalConv2D(
        num_filters, (5, 5), corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_1", inverse=True)
    )(decoded)
    decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
    
    decoded = tfc.SignalConv2D(
        num_filters, (5, 5), corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_2", inverse=True)
    )(decoded)
    decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
    
    decoded = tfc.SignalConv2D(
        3, (9, 9), corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tf.nn.sigmoid
    )(decoded)

    model = Model(inputs=input_img, outputs=decoded)
    return model

class DJSCCGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DJSCC System Visualization")
        
        # Initialize status variable first
        self.status_var = tk.StringVar(value="Initializing...")
        
        # Set up the main container
        self.mainframe = ttk.Frame(root, padding="10")
        self.mainframe.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Variables
        self.snr_var = tk.DoubleVar(value=10.0)
        self.block_size_var = tk.IntVar(value=16)
        
        # Create GUI elements
        self.create_control_panel()
        self.create_visualization_panel()
        
        # Load the model
        self.load_model()
        
        # Load sample data
        self.load_sample_data()
        
        # Initial update
        self.update_visualization()

    def load_model(self):
        try:
            self.model = build_model(self.snr_var.get(), self.block_size_var.get())
            weights_path = 'classifier_model_weights_rec_train.h5'
            
            if not os.path.exists(weights_path):
                messagebox.showwarning("Warning", 
                    "Model weights file not found. Please train the model first or ensure the weights file is in the correct location.")
                self.status_var.set("Model weights not found - Running in demo mode")
            else:
                self.model.load_weights(weights_path)
                self.status_var.set("Model loaded successfully")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
            self.status_var.set("Error loading model - Running in demo mode")

    def create_control_panel(self):
        control_frame = ttk.LabelFrame(self.mainframe, text="Control Panel", padding="5")
        control_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # SNR Slider
        ttk.Label(control_frame, text="SNR (dB):").grid(row=0, column=0, padx=5, pady=5)
        snr_slider = ttk.Scale(control_frame, from_=1, to=20, orient=tk.HORIZONTAL, 
                             variable=self.snr_var, command=self.update_visualization)
        snr_slider.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Block Size Slider
        ttk.Label(control_frame, text="Block Size:").grid(row=1, column=0, padx=5, pady=5)
        block_size_slider = ttk.Scale(control_frame, from_=8, to=32, orient=tk.HORIZONTAL,
                                    variable=self.block_size_var, command=self.update_visualization)
        block_size_slider.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Process Button
        ttk.Button(control_frame, text="Process Image", 
                  command=self.process_image).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Training Status
        ttk.Label(control_frame, textvariable=self.status_var).grid(
            row=3, column=0, columnspan=2, pady=5)

    def create_visualization_panel(self):
        viz_frame = ttk.LabelFrame(self.mainframe, text="System Visualization", padding="5")
        viz_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)

    def load_sample_data(self):
        (_, _), (self.x_test, _) = cifar10.load_data()
        self.x_test = self.x_test.astype('float32') / 255.0
        self.current_image = self.x_test[0]

    def process_image(self):
        try:
            self.status_var.set("Processing...")
            self.root.update()
            
            # Prepare input
            input_image = np.expand_dims(self.current_image, 0)
            
            # Get model prediction
            decoded_image = self.model.predict(input_image, verbose=0)
            
            # Update visualization
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            
            self.ax1.imshow(self.current_image)
            self.ax1.set_title("Input Image")
            self.ax1.axis('off')
            
            # Get intermediate layer output
            intermediate_layer_model = Model(inputs=self.model.input,
                                          outputs=self.model.get_layer('layer_out').output)
            encoded = intermediate_layer_model.predict(input_image, verbose=0)
            
            self.ax2.imshow(encoded[0, :, :, 0], cmap='viridis')
            self.ax2.set_title("Encoded Signal")
            self.ax2.axis('off')
            
            self.ax3.imshow(decoded_image[0])
            self.ax3.set_title("Decoded Image")
            self.ax3.axis('off')
            
            self.canvas.draw()
            
            # Calculate and display PSNR
            psnr = tf.image.psnr(self.current_image, decoded_image[0], max_val=1.0)
            self.status_var.set(f"PSNR: {psnr.numpy():.2f} dB")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
            self.status_var.set("Error processing image")

    def update_visualization(self, *args):
        try:
            # Rebuild model with new parameters
            self.model = build_model(self.snr_var.get(), self.block_size_var.get())
            weights_path = 'classifier_model_weights_rec_train.h5'
            
            if os.path.exists(weights_path):
                self.model.load_weights(weights_path)
                self.process_image()
            else:
                self.status_var.set("Running without trained weights")
                self.process_image()
                
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")

def main():
    root = tk.Tk()
    root.title("DJSCC System")
    
    # Configure style
    style = ttk.Style()
    style.theme_use('clam')
    
    try:
        app = DJSCCGUI(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"Application error: {str(e)}")

if __name__ == "__main__":
    main()