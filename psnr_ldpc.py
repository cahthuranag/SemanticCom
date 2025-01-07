import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import tensorflow as tf
from PIL import Image, ImageTk
import os

# Import from original code
from psnr_all import (get_dataset, calculate_psnr, LDPCTransmitter)

class LDPCAnalysisFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.init_variables()
        self.create_widgets()
        self.load_data()
        
    def init_variables(self):
        # LDPC parameters
        self.bw_ratio_var = tk.DoubleVar(value=1/3)
        self.k_var = tk.IntVar(value=3072)
        self.n_var = tk.IntVar(value=4608)
        self.m_var = tk.IntVar(value=4)
        self.snr_var = tk.DoubleVar(value=10.0)
        
        # General variables
        self.status_var = tk.StringVar(value="Ready")
        self.current_image_idx = tk.IntVar(value=0)

    def create_widgets(self):
        main_container = ttk.Frame(self)
        main_container.grid(row=0, column=0, sticky="nsew")
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Create frames
        self.create_ldpc_frame(main_container)
        self.create_control_frame(main_container)

    def create_ldpc_frame(self, parent):
        ldpc_frame = ttk.LabelFrame(parent, text="LDPC System")
        ldpc_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Parameters Frame
        param_frame = ttk.Frame(ldpc_frame)
        param_frame.pack(fill="x", padx=5, pady=5)
        
        # BW Ratio
        ttk.Label(param_frame, text="BW Ratio:").grid(row=0, column=0, pady=5, padx=5)
        ttk.Entry(param_frame, textvariable=self.bw_ratio_var, width=10).grid(row=0, column=1)
        
        # K value
        ttk.Label(param_frame, text="K:").grid(row=1, column=0, pady=5, padx=5)
        ttk.Entry(param_frame, textvariable=self.k_var, width=10).grid(row=1, column=1)
        
        # N value
        ttk.Label(param_frame, text="N:").grid(row=2, column=0, pady=5, padx=5)
        ttk.Entry(param_frame, textvariable=self.n_var, width=10).grid(row=2, column=1)
        
        # M value
        ttk.Label(param_frame, text="M:").grid(row=3, column=0, pady=5, padx=5)
        ttk.Entry(param_frame, textvariable=self.m_var, width=10).grid(row=3, column=1)
        
        # SNR
        ttk.Label(param_frame, text="SNR (dB):").grid(row=4, column=0, pady=5, padx=5)
        ttk.Entry(param_frame, textvariable=self.snr_var, width=10).grid(row=4, column=1)
        
        # Process Button
        btn_frame = ttk.Frame(ldpc_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(btn_frame, text="Process LDPC", 
                  command=self.process_ldpc).pack(side="left", padx=5)
        
        # Display Frame
        display_frame = ttk.Frame(ldpc_frame)
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

    def process_ldpc(self):
        try:
            # Get parameters
            bw_ratio = self.bw_ratio_var.get()
            k = self.k_var.get()
            n = self.n_var.get()
            m = self.m_var.get()
            snr = self.snr_var.get()

            # Create LDPC transmitter
            transmitter = LDPCTransmitter(k, n, m, snr, 'AWGN')

            # Prepare input image
            self.display_image(self.current_image, self.input_label)

            # Display encoded signal (simulated)
            self.display_ldpc_encoded(k, n, self.encoded_label)

            # Process through LDPC system
            psnr_val = calculate_psnr(bw_ratio, k, n, m, snr)
            
            # Display simulated output
            self.display_ldpc_output(self.output_label)
            
            # Update status with PSNR
            self.status_var.set(f"LDPC PSNR: {psnr_val[0]:.2f} dB")
            
        except Exception as e:
            messagebox.showerror("Error", f"LDPC processing error: {str(e)}")
            self.status_var.set("LDPC processing failed")

    def display_image(self, img_array, label):
        """Display an image on a label"""
        img_array = np.clip(img_array * 255, 0, 255).astype('uint8')
        img = Image.fromarray(img_array)
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo)
        label.image = photo

    def display_ldpc_encoded(self, k, n, label):
        """Simulate LDPC encoded signal visualization"""
        # Create a visualization of the encoded signal
        # This is a simplified visualization for demonstration
        encoded_size = int(np.sqrt(n))
        encoded_array = np.random.random((encoded_size, encoded_size))
        img_array = (encoded_array * 255).astype('uint8')
        img = Image.fromarray(img_array)
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo)
        label.image = photo

    def display_ldpc_output(self, label):
        """Display LDPC output (with simulated noise)"""
        # Add simulated noise to the original image
        noisy_image = np.clip(
            self.current_image + np.random.normal(0, 0.1, self.current_image.shape),
            0, 1
        )
        self.display_image(noisy_image, label)


def main():
    root = tk.Tk()
    root.title("LDPC System Demo")
    root.geometry("800x600")
    
    app = LDPCAnalysisFrame(root)
    app.pack(expand=True, fill="both", padx=10, pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    main()