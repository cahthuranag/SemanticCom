import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import tensorflow as tf
from PIL import Image, ImageTk
import os

# Import from original code
from psnr_all import (get_dataset, calculate_psnr, LDPCTransmitter)

class ModernStyle:
    # Color scheme
    PRIMARY_COLOR = "#2c3e50"  # Dark blue-gray
    SECONDARY_COLOR = "#3498db"  # Bright blue
    ACCENT_COLOR = "#e74c3c"  # Red
    BG_COLOR = "#ecf0f1"  # Light gray
    TEXT_COLOR = "#2c3e50"  # Dark blue-gray
    BUTTON_BG = "#3498db"  # Bright blue
    BUTTON_ACTIVE = "#2980b9"  # Darker blue
    
    # Styles
    FRAME_STYLE = {
        "background": BG_COLOR,
        "padding": 10
    }
    
    LABEL_STYLE = {
        "background": BG_COLOR,
        "foreground": TEXT_COLOR,
        "font": ("Helvetica", 10)
    }
    
    HEADER_STYLE = {
        "background": BG_COLOR,
        "foreground": PRIMARY_COLOR,
        "font": ("Helvetica", 12, "bold")
    }

class LDPCAnalysisFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.style = ModernStyle()
        self.configure(style="Modern.TFrame")
        self.init_styles()
        self.init_variables()
        self.create_widgets()
        self.load_data()
        
    def init_styles(self):
        # Configure ttk styles
        style = ttk.Style()
        style.configure("Modern.TFrame", background=self.style.BG_COLOR)
        style.configure("Modern.TLabel", **self.style.LABEL_STYLE)
        style.configure("Modern.TLabelframe", background=self.style.BG_COLOR)
        style.configure("Modern.TLabelframe.Label", **self.style.HEADER_STYLE)
        
        # Custom button style
        style.configure("Modern.TButton",
            background=self.style.BUTTON_BG,
            foreground="white",
            padding=(10, 5),
            font=("Helvetica", 9)
        )
        style.map("Modern.TButton",
            background=[("active", self.style.BUTTON_ACTIVE)],
            foreground=[("active", "white")]
        )
        
        # Entry style
        style.configure("Modern.TEntry",
            fieldbackground="white",
            padding=5
        )

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
        # Main container with padding and background
        main_container = ttk.Frame(self, style="Modern.TFrame")
        main_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.create_ldpc_frame(main_container)
        self.create_control_frame(main_container)

    def create_ldpc_frame(self, parent):
        ldpc_frame = ttk.LabelFrame(parent, text="LDPC + BPG System", style="Modern.TLabelframe")
        ldpc_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Parameters Frame with modern styling
        param_frame = ttk.Frame(ldpc_frame, style="Modern.TFrame")
        param_frame.pack(fill="x", padx=10, pady=10)
        
        # Create parameter entries with consistent spacing and styling
        params = [
            ("BW Ratio:", self.bw_ratio_var),
            ("K:", self.k_var),
            ("N:", self.n_var),
            ("M:", self.m_var),
            ("SNR (dB):", self.snr_var)
        ]
        
        for idx, (label_text, var) in enumerate(params):
            ttk.Label(param_frame, text=label_text, style="Modern.TLabel").grid(
                row=idx, column=0, pady=8, padx=8, sticky="e"
            )
            ttk.Entry(param_frame, textvariable=var, width=10, style="Modern.TEntry").grid(
                row=idx, column=1, pady=8, padx=8, sticky="w"
            )
        
        # Process Button with modern styling
        btn_frame = ttk.Frame(ldpc_frame, style="Modern.TFrame")
        btn_frame.pack(fill="x", padx=10, pady=10)
        ttk.Button(btn_frame, text="Process LDPC + BPG ", style="Modern.TButton",
                  command=self.process_ldpc).pack(side="left", padx=5)
        
        # Display Frame with improved layout
        display_frame = ttk.Frame(ldpc_frame, style="Modern.TFrame")
        display_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Image labels with consistent spacing
        self.input_label = ttk.Label(display_frame, style="Modern.TLabel")
        self.input_label.grid(row=0, column=0, padx=10)
        
        arrow_label = ttk.Label(display_frame, text="→", style="Modern.TLabel",
                              font=("Helvetica", 16, "bold"))
        arrow_label.grid(row=0, column=1)
        
        self.encoded_label = ttk.Label(display_frame, style="Modern.TLabel")
        self.encoded_label.grid(row=0, column=2, padx=10)
        
        arrow_label2 = ttk.Label(display_frame, text="→", style="Modern.TLabel",
                               font=("Helvetica", 16, "bold"))
        arrow_label2.grid(row=0, column=3)
        
        self.output_label = ttk.Label(display_frame, style="Modern.TLabel")
        self.output_label.grid(row=0, column=4, padx=10)
        
        # Image labels with modern styling
        for col, text in [(0, "Input"), (2, "Encoded"), (4, "Output")]:
            ttk.Label(display_frame, text=text, style="Modern.TLabel",
                     font=("Helvetica", 10, "bold")).grid(row=1, column=col)

    def create_control_frame(self, parent):
        control_frame = ttk.LabelFrame(parent, text="Image Control", style="Modern.TLabelframe")
        control_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        # Control elements with modern styling
        ttk.Label(control_frame, text="Image Index:", style="Modern.TLabel").pack(side="left", padx=8)
        ttk.Entry(control_frame, textvariable=self.current_image_idx,
                 width=5, style="Modern.TEntry").pack(side="left", padx=8)
        ttk.Button(control_frame, text="Load Image", style="Modern.TButton",
                  command=self.load_selected_image).pack(side="left", padx=8)
        
        # Status label with modern styling
        status_label = ttk.Label(control_frame, textvariable=self.status_var,
                               style="Modern.TLabel")
        status_label.pack(side="right", padx=8)

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