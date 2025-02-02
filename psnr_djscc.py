import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import tensorflow as tf
from PIL import Image, ImageTk
import os

# Import from original PSNR code
from psnr_all import (build_model, get_dataset, train, test)

class ModernStyle:
    # Color scheme
    PRIMARY_COLOR = "#2c3e50"  # Dark blue-gray
    SECONDARY_COLOR = "#3498db"  # Bright blue
    ACCENT_COLOR = "#e74c3c"  # Red
    BG_COLOR = "#ecf0f1"  # Light gray
    TEXT_COLOR = "#2c3e50"  # Dark blue-gray
    BUTTON_BG = "#3498db"  # Bright blue
    BUTTON_ACTIVE = "#2980b9"  # Darker blue
    
    # Styles with enlarged fonts
    FRAME_STYLE = {
        "background": BG_COLOR,
        "padding": 10
    }
    
    LABEL_STYLE = {
        "background": BG_COLOR,
        "foreground": TEXT_COLOR,
        "font": ("Helvetica", 12)
    }
    
    HEADER_STYLE = {
        "background": BG_COLOR,
        "foreground": PRIMARY_COLOR,
        "font": ("Helvetica", 14, "bold")
    }

class DJSCCAnalysisFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.style = ModernStyle()
        self.configure(style="Modern.TFrame")
        self.init_styles()
        self.init_variables()
        self.create_widgets()
        self.load_data()
        
    def init_styles(self):
        # Configure ttk styles with larger fonts
        style = ttk.Style()
        style.configure("Modern.TFrame", background=self.style.BG_COLOR)
        style.configure("Modern.TLabel", **self.style.LABEL_STYLE)
        style.configure("Modern.TLabelframe", background=self.style.BG_COLOR)
        style.configure("Modern.TLabelframe.Label", **self.style.HEADER_STYLE)
        
        # Custom button style with larger font
        style.configure("Modern.TButton",
            background=self.style.BUTTON_BG,
            foreground="white",
            padding=(10, 5),
            font=("Helvetica", 11)
        )
        style.map("Modern.TButton",
            background=[("active", self.style.BUTTON_ACTIVE)],
            foreground=[("active", "white")]
        )
        
        # Entry style with larger font
        style.configure("Modern.TEntry",
            fieldbackground="white",
            padding=5,
            font=("Helvetica", 11)
        )

    def init_variables(self):
        # DJSCC variables
        self.train_snr_var = tk.DoubleVar(value=10.0)
        self.channel_snr_var = tk.DoubleVar(value=10.0)
        self.block_size_var = tk.IntVar(value=16)
        self.status_var = tk.StringVar(value="Ready")
        self.is_training = False
        self.model = None
        self.current_image_idx = tk.IntVar(value=0)

    def create_widgets(self):
        # Main container with padding and background
        main_container = ttk.Frame(self, style="Modern.TFrame")
        main_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.create_djscc_frame(main_container)
        self.create_control_frame(main_container)

    def create_djscc_frame(self, parent):
        djscc_frame = ttk.LabelFrame(parent, text="DJSCC System", style="Modern.TLabelframe")
        djscc_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Parameters Frame with modern styling
        param_frame = ttk.Frame(djscc_frame, style="Modern.TFrame")
        param_frame.pack(fill="x", padx=10, pady=10)
        
        # Training SNR with styled widgets
        ttk.Label(param_frame, text="Training SNR (dB):", style="Modern.TLabel").grid(row=0, column=0, pady=8, padx=8)
        ttk.Entry(param_frame, textvariable=self.train_snr_var, width=10, style="Modern.TEntry").grid(row=0, column=1)
        
        # Channel SNR
        ttk.Label(param_frame, text="Channel SNR (dB):", style="Modern.TLabel").grid(row=1, column=0, pady=8, padx=8)
        ttk.Entry(param_frame, textvariable=self.channel_snr_var, width=10, style="Modern.TEntry").grid(row=1, column=1)
        
        # Block Size controls with training-only indicator
        block_label_frame = ttk.Frame(param_frame, style="Modern.TFrame")
        block_label_frame.grid(row=2, column=0, pady=8, padx=8, sticky="w")
        
        ttk.Label(block_label_frame, text="Block Size:", style="Modern.TLabel").pack(side="left")
        ttk.Label(block_label_frame, text="(Training Only)", 
                 style="Modern.TLabel",
                 font=("Helvetica", 10, "italic")).pack(side="left", padx=(4, 0))
        
        # Create a frame for block size controls
        block_size_frame = ttk.Frame(param_frame, style="Modern.TFrame")
        block_size_frame.grid(row=2, column=1, sticky="w")
        
        self.block_entry = ttk.Entry(block_size_frame, textvariable=self.block_size_var, 
                                   width=6, style="Modern.TEntry")
        self.block_entry.pack(side="left", padx=2)
        
        self.block_inc_btn = ttk.Button(block_size_frame, text="+", style="Modern.TButton", width=2,
                                      command=lambda: self.block_size_var.set(self.block_size_var.get() + 4))
        self.block_inc_btn.pack(side="left", padx=1)
        
        self.block_dec_btn = ttk.Button(block_size_frame, text="-", style="Modern.TButton", width=2,
                                      command=lambda: self.block_size_var.set(max(4, self.block_size_var.get() - 4)))
        self.block_dec_btn.pack(side="left", padx=1)
        
        # Buttons Frame
        btn_frame = ttk.Frame(djscc_frame, style="Modern.TFrame")
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        # Styled buttons
        ttk.Button(btn_frame, text="Train New Model", style="Modern.TButton",
                  command=self.start_training).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Load & Process", style="Modern.TButton",
                  command=self.process_djscc).pack(side="left", padx=5)
        
        # Display Frame
        display_frame = ttk.Frame(djscc_frame, style="Modern.TFrame")
        display_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Image display with modern styling
        self.input_label = ttk.Label(display_frame, style="Modern.TLabel")
        self.input_label.grid(row=0, column=0, padx=10)
        
        arrow_label = ttk.Label(display_frame, text="→", style="Modern.TLabel", font=("Helvetica", 16, "bold"))
        arrow_label.grid(row=0, column=1)
        
        self.encoded_label = ttk.Label(display_frame, style="Modern.TLabel")
        self.encoded_label.grid(row=0, column=2, padx=10)
        
        arrow_label2 = ttk.Label(display_frame, text="→", style="Modern.TLabel", font=("Helvetica", 16, "bold"))
        arrow_label2.grid(row=0, column=3)
        
        self.output_label = ttk.Label(display_frame, style="Modern.TLabel")
        self.output_label.grid(row=0, column=4, padx=10)
        
        # Image labels with modern styling
        ttk.Label(display_frame, text="Input", style="Modern.TLabel", font=("Helvetica", 12, "bold")).grid(row=1, column=0)
        ttk.Label(display_frame, text="Encoded", style="Modern.TLabel", font=("Helvetica", 12, "bold")).grid(row=1, column=2)
        ttk.Label(display_frame, text="Output", style="Modern.TLabel", font=("Helvetica", 12, "bold")).grid(row=1, column=4)

    def create_control_frame(self, parent):
        control_frame = ttk.LabelFrame(parent, text="Image Control", style="Modern.TLabelframe")
        control_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        # Control elements with modern styling
        ttk.Label(control_frame, text="Image Index:", style="Modern.TLabel").pack(side="left", padx=8)
        ttk.Entry(control_frame, textvariable=self.current_image_idx, width=5, style="Modern.TEntry").pack(side="left", padx=8)
        ttk.Button(control_frame, text="Load Image", style="Modern.TButton",
                  command=self.load_selected_image).pack(side="left", padx=8)
        
        # Status label with modern styling
        status_label = ttk.Label(control_frame, textvariable=self.status_var, style="Modern.TLabel")
        status_label.pack(side="right", padx=8)

    def update_block_controls(self):
        """Enable/disable block size controls based on model state"""
        state = "normal" if not self.model else "disabled"
        self.block_entry.configure(state=state)
        self.block_inc_btn.configure(state=state)
        self.block_dec_btn.configure(state=state)

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
            self.update_block_controls()  # Disable block size controls after training
            
        except Exception as e:
            messagebox.showerror("Error", f"Training error: {str(e)}")
            self.status_var.set("Training failed!")
        
        finally:
            self.is_training = False

    def process_djscc(self):
        if self.is_training:
            messagebox.showwarning("Warning", "Please wait for training to complete")
            return
            
        # Show a message about block size being fixed after model loading
        if self.model is None:
            messagebox.showinfo("Info", "Note: Block size cannot be changed after loading the model.")

        try:
            # Load existing model if not loaded
            if self.model is None:
                train_snr = self.train_snr_var.get()
                block_size = self.block_size_var.get()
                self.model = build_model(train_snr, block_size)
                
                if os.path.exists('classifier_model_weights_rec_train.h5'):
                    self.model.load_weights('classifier_model_weights_rec_train.h5')
                    self.update_block_controls()  # Disable block size controls after loading
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