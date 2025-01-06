# psnr_analysis.py

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import tensorflow as tf

# Import from your original PSNR code
from psnr_all import (build_model, get_dataset, train, test, calculate_psnr)

class PSNRAnalysisFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.init_variables()
        self.create_widgets()
        
    def init_variables(self):
        self.snr_var = tk.DoubleVar(value=10.0)
        self.block_size_var = tk.IntVar(value=16)
        self.bw_ratio_var = tk.DoubleVar(value=1/3)
        self.k_var = tk.IntVar(value=3072)
        self.n_var = tk.IntVar(value=4608)
        self.m_var = tk.IntVar(value=4)
        self.status_var = tk.StringVar(value="Ready")
        self.is_training = False

    def create_widgets(self):
        # Parameters Frame
        param_frame = ttk.LabelFrame(self, text="Parameters")
        param_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # DJSCC Parameters
        djscc_frame = ttk.LabelFrame(param_frame, text="DJSCC")
        djscc_frame.grid(row=0, column=0, padx=5, pady=5)
        
        # SNR
        ttk.Label(djscc_frame, text="Training SNR:").grid(row=0, column=0, pady=5)
        ttk.Entry(djscc_frame, textvariable=self.snr_var, width=10).grid(row=0, column=1)
        
        # Block Size
        ttk.Label(djscc_frame, text="Block Size:").grid(row=1, column=0, pady=5)
        ttk.Entry(djscc_frame, textvariable=self.block_size_var, width=10).grid(row=1, column=1)
        
        # LDPC Parameters
        ldpc_frame = ttk.LabelFrame(param_frame, text="LDPC")
        ldpc_frame.grid(row=0, column=1, padx=5, pady=5)
        
        # BW Ratio
        ttk.Label(ldpc_frame, text="BW Ratio:").grid(row=0, column=0, pady=5)
        ttk.Entry(ldpc_frame, textvariable=self.bw_ratio_var, width=10).grid(row=0, column=1)
        
        # K, N, M values
        params = [("K:", self.k_var), ("N:", self.n_var), ("M:", self.m_var)]
        for i, (label, var) in enumerate(params, 1):
            ttk.Label(ldpc_frame, text=label).grid(row=i, column=0, pady=5)
            ttk.Entry(ldpc_frame, textvariable=var, width=10).grid(row=i, column=1)
        
        # Control Buttons
        btn_frame = ttk.Frame(param_frame)
        btn_frame.grid(row=0, column=2, padx=10)
        
        ttk.Button(btn_frame, text="Train Model",
                  command=self.start_training).grid(row=0, column=0, pady=5)
        ttk.Button(btn_frame, text="Compare PSNR",
                  command=self.compare_psnr).grid(row=1, column=0, pady=5)
        
        # Results Visualization
        viz_frame = ttk.LabelFrame(self, text="Results")
        viz_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        ttk.Label(self, textvariable=self.status_var).grid(row=2, column=0, sticky="w", padx=5)

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
            # Get parameters
            snr = self.snr_var.get()
            block_size = self.block_size_var.get()
            
            # Load dataset
            x_train, x_val, x_test = get_dataset()
            
            # Train model
            self.status_var.set("Training DJSCC model...")
            history = train(snr, x_train, x_val, x_test, block_size)
            
            # Plot training history
            self.ax1.clear()
            self.ax1.plot(history.history['psnr_metric'])
            self.ax1.plot(history.history['val_psnr_metric'])
            self.ax1.set_title('Model PSNR')
            self.ax1.set_ylabel('PSNR (dB)')
            self.ax1.set_xlabel('Epoch')
            self.ax1.legend(['Train', 'Validation'])
            self.canvas.draw()
            
            self.status_var.set("Training complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training error: {str(e)}")
            self.status_var.set("Training failed!")
        
        finally:
            self.is_training = False

    def compare_psnr(self):
        if self.is_training:
            messagebox.showwarning("Warning", "Please wait for training to complete")
            return
            
        self.status_var.set("Comparing PSNR...")
        thread = threading.Thread(target=self.comparison_process)
        thread.daemon = True
        thread.start()

    def comparison_process(self):
        try:
            snr_range = np.linspace(1, 20, num=5)
            block_size = self.block_size_var.get()
            bw_ratio = self.bw_ratio_var.get()
            k = self.k_var.get()
            n = self.n_var.get()
            m = self.m_var.get()
            
            _, _, x_test = get_dataset()
            
            djscc_psnr = []
            ldpc_psnr = []
            
            for snr in snr_range:
                self.status_var.set(f"Processing SNR: {snr:.1f} dB")
                djscc_val = test(snr, x_test, block_size)
                djscc_psnr.append(djscc_val)
                
                ldpc_val = calculate_psnr(bw_ratio, k, n, m, snr)
                ldpc_psnr.append(ldpc_val[0])
            
            self.ax2.clear()
            self.ax2.plot(snr_range, djscc_psnr, 'b-o', label='DJSCC')
            self.ax2.plot(snr_range, ldpc_psnr, 'r--x', label='LDPC')
            self.ax2.set_xlabel('SNR (dB)')
            self.ax2.set_ylabel('PSNR (dB)')
            self.ax2.set_title('PSNR Comparison')
            self.ax2.grid(True)
            self.ax2.legend()
            
            self.canvas.draw()
            self.status_var.set("Comparison complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Comparison error: {str(e)}")
            self.status_var.set("Comparison failed!")