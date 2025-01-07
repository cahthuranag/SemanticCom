import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import tensorflow as tf
from PIL import Image, ImageTk
import os

# Import from original PSNR code
from psnr_all import (build_model, get_dataset, train, test)

class DJSCCAnalysisFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.init_variables()
        self.create_widgets()
        self.load_data()
        
    def init_variables(self):
        # DJSCC variables
        self.train_snr_var = tk.DoubleVar(value=10.0)  # SNR for training
        self.channel_snr_var = tk.DoubleVar(value=10.0)  # SNR for channel transmission
        self.block_size_var = tk.IntVar(value=16)
        
        # General variables
        self.status_var = tk.StringVar(value="Ready")
        self.is_training = False
        self.model = None
        self.current_image_idx = tk.IntVar(value=0)

    def create_widgets(self):
        # Main container
        main_container = ttk.Frame(self)
        main_container.grid(row=0, column=0, sticky="nsew")
        
        # Configure weights
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Create frames
        self.create_djscc_frame(main_container)
        self.create_control_frame(main_container)

    def create_djscc_frame(self, parent):
        djscc_frame = ttk.LabelFrame(parent, text="DJSCC System")
        djscc_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Parameters Frame
        param_frame = ttk.Frame(djscc_frame)
        param_frame.pack(fill="x", padx=5, pady=5)
        
        # Training SNR
        ttk.Label(param_frame, text="Training SNR (dB):").grid(row=0, column=0, pady=5, padx=5)
        ttk.Entry(param_frame, textvariable=self.train_snr_var, width=10).grid(row=0, column=1)
        
        # Channel SNR
        ttk.Label(param_frame, text="Channel SNR (dB):").grid(row=1, column=0, pady=5, padx=5)
        ttk.Entry(param_frame, textvariable=self.channel_snr_var, width=10).grid(row=1, column=1)
        
        # Block Size
        ttk.Label(param_frame, text="Block Size:").grid(row=2, column=0, pady=5, padx=5)
        ttk.Entry(param_frame, textvariable=self.block_size_var, width=10).grid(row=2, column=1)
        
        # Buttons
        btn_frame = ttk.Frame(djscc_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Train New Model",
                  command=self.start_training).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Load & Process",
                  command=self.process_djscc).pack(side="left", padx=5)
        
        # Display Frame
        display_frame = ttk.Frame(djscc_frame)
        display_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create image labels
        self.input_label = ttk.Label(display_frame)
        self.input_label.grid(row=0, column=0, padx=5)
        
        ttk.Label(display_frame, text="→").grid(row=0, column=1)
        
        self.encoded_label = ttk.Label(display_frame)
        self.encoded_label.grid(row=0, column=2, padx=5)
        
        ttk.Label(display_frame, text="→").grid(row=0, column=3)
        
        self.output_label = ttk.Label(display_frame)
        self.output_label.grid(row=0, column=4, padx=5)
        
        # Labels
        ttk.Label(display_frame, text="Input").grid(row=1, column=0)
        ttk.Label(display_frame, text="Encoded").grid(row=1, column=2)
        ttk.Label(display_frame, text="Output").grid(row=1, column=4)

    def create_control_frame(self, parent):
        control_frame = ttk.LabelFrame(parent, text="Image Control")
        control_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        ttk.Label(control_frame, text="Image Index:").pack(side="left", padx=5)
        ttk.Entry(control_frame, textvariable=self.current_image_idx, width=5).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Load Image", 
                  command=self.load_selected_image).pack(side="left", padx=5)
        
        ttk.Label(control_frame, textvariable=self.status_var).pack(side="right", padx=5)

    def load_data(self):
        try:
            _, _, self.x_test = get_dataset()
            self.load_selected_image()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {str(e)}")

    def load_selected_image(self):
        try:
            idx = min(max(0, self.current_image_idx.get()), len(self.x_test)-1)
            self.current_image = self.x_test[idx]
            self.display_image(self.current_image, self.input_label)
            self.status_var.set(f"Loaded image {idx}")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")

    def start_training(self):
        if self.is_training:
            return
        self.is_training = True
        self.status_var.set("Starting training...")
        
        thread = threading.Thread(target=self.training_process)
        thread.daemon = True
        thread.start()

    def training_process(self):
        try:
            train_snr = self.train_snr_var.get()
            block_size = self.block_size_var.get()
            
            x_train, x_val, x_test = get_dataset()
            
            self.status_var.set("Training DJSCC model...")
            self.model = build_model(train_snr, block_size)
            train(train_snr, x_train, x_val, x_test, block_size)
            
            self.status_var.set("Training complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training error: {str(e)}")
            self.status_var.set("Training failed!")
        
        finally:
            self.is_training = False

    def process_djscc(self):
        if self.is_training:
            messagebox.showwarning("Warning", "Please wait for training to complete")
            return

        try:
            # Load existing model if not loaded
            if self.model is None:
                train_snr = self.train_snr_var.get()
                block_size = self.block_size_var.get()
                self.model = build_model(train_snr, block_size)
                
                if os.path.exists('classifier_model_weights_rec_train.h5'):
                    self.model.load_weights('classifier_model_weights_rec_train.h5')
                else:
                    raise Exception("No trained weights found")

            # Process with channel SNR
            channel_snr = self.channel_snr_var.get()
            test_model = build_model(channel_snr, self.block_size_var.get())
            test_model.load_weights('classifier_model_weights_rec_train.h5')

            input_image = np.expand_dims(self.current_image, 0)
            
            # Get encoded representation
            intermediate_model = tf.keras.Model(
                inputs=test_model.input,
                outputs=test_model.get_layer('layer_out').output
            )
            encoded = intermediate_model.predict(input_image, verbose=0)
            
            # Get reconstructed output
            output = test_model.predict(input_image, verbose=0)
            
            # Display results
            self.display_image(self.current_image, self.input_label)
            self.display_encoded(encoded[0], self.encoded_label)
            self.display_image(output[0], self.output_label)
            
            # Calculate PSNR
            psnr = tf.image.psnr(self.current_image, output[0], max_val=1.0)
            self.status_var.set(f"DJSCC PSNR: {psnr.numpy():.2f} dB (Train SNR: {self.train_snr_var.get()}dB, Channel SNR: {channel_snr}dB)")
            
        except Exception as e:
            messagebox.showerror("Error", f"DJSCC processing error: {str(e)}")
            self.status_var.set("DJSCC processing failed")

    def display_image(self, img_array, label):
        """Display an image on a label"""
        img_array = np.clip(img_array * 255, 0, 255).astype('uint8')
        img = Image.fromarray(img_array)
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo)
        label.image = photo

    def display_encoded(self, encoded_array, label):
        """Display encoded representation"""
        encoded_channel = encoded_array[:, :, 0]
        normalized = (encoded_channel - encoded_channel.min()) / (encoded_channel.max() - encoded_channel.min() + 1e-8)
        img_array = (normalized * 255).astype('uint8')
        img = Image.fromarray(img_array)
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo)
        label.image = photo


def main():
    root = tk.Tk()
    root.title("DJSCC System Demo")
    root.geometry("800x600")
    
    app = DJSCCAnalysisFrame(root)
    app.pack(expand=True, fill="both", padx=10, pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    main()