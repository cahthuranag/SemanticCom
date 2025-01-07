import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import os
from ac_all import train_djscc, test_djscc
from tensorflow.keras.datasets import cifar10

class DJSCCAnalysisFrame_ac(ttk.Frame):
    
    def __init__(self, parent):
        super().__init__(parent)
        self.init_variables()
        self.create_widgets()
        self.check_model_files()
        
    def init_variables(self):
        # DJSCC variables
        self.train_snr_var = tk.DoubleVar(value=10.0)
        self.block_size_var = tk.IntVar(value=16)
        self.block_size_var2 = tk.IntVar(value=8)
        
        # Test SNR variable
        self.test_snr_var = tk.DoubleVar(value=10.0)
        
        # Bandwidth ratio selection
        self.br_var = tk.StringVar(value="1/3")
        
        # General variables
        self.status_var = tk.StringVar(value="Ready")
        self.is_running = False
        
        # Results storage
        self.accuracy_value = None

    def check_model_files(self):
        """Check if model weight files exist and enable/disable test button accordingly"""
        if os.path.exists('model_weights.h5'):
            self.test_button.config(state='normal')
            self.status_var.set("Model files found. Ready for analysis.")
        else:
            self.test_button.config(state='disabled')
            self.status_var.set("Training required.")

    def create_widgets(self):
        # Configure main grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Parameters Frame
        params_frame = ttk.LabelFrame(self, text="DJSCC Parameters")
        params_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        # Training Parameters Section
        train_params_frame = ttk.LabelFrame(params_frame, text="Training Parameters")
        train_params_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.create_labeled_entry(train_params_frame, "Training SNR (dB):", self.train_snr_var, 0)
        
        # Testing Parameters Section
        test_params_frame = ttk.LabelFrame(params_frame, text="Testing Parameters")
        test_params_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        # Bandwidth Ratio Selection
        ttk.Label(test_params_frame, text="Bandwidth Ratio:").grid(row=0, column=0, padx=5, pady=2, sticky="e")
        br_combo = ttk.Combobox(test_params_frame, textvariable=self.br_var, values=["1/3", "1/6"], 
                               state="readonly", width=9)
        br_combo.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        br_combo.bind('<<ComboboxSelected>>', self.on_br_changed)
        
        # Dynamic block size display
        self.block_size_label = ttk.Label(test_params_frame, text="")
        self.block_size_label.grid(row=1, column=0, columnspan=2, padx=5, pady=2)
        
        # Test SNR input
        self.create_labeled_entry(test_params_frame, "Test SNR (dB):", self.test_snr_var, 2)
        
        # Update block size display
        self.update_block_size_display()

        # Results Frame
        results_frame = ttk.LabelFrame(self, text="Results")
        results_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # Create Text widget for results
        self.results_text = tk.Text(results_frame, height=10, width=50)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Control Frame
        control_frame = ttk.Frame(self)
        control_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        # Train and Test buttons
        self.train_button = ttk.Button(control_frame, text="Train Model", 
                                     command=self.on_train_clicked)
        self.train_button.pack(side="left", padx=5)
        
        self.test_button = ttk.Button(control_frame, text="Run Analysis", 
                                    command=self.on_test_clicked)
        self.test_button.pack(side="left", padx=5)
        
        ttk.Label(control_frame, textvariable=self.status_var).pack(side="right", padx=5)

    def create_labeled_entry(self, parent, label_text, variable, row):
        """Helper method to create a labeled entry with consistent styling"""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, padx=5, pady=2, sticky="e")
        entry = ttk.Entry(parent, textvariable=variable, width=12)
        entry.grid(row=row, column=1, padx=5, pady=2, sticky="w")
        return entry

    def on_br_changed(self, event=None):
        """Handle bandwidth ratio selection change"""
        self.update_block_size_display()

    def update_block_size_display(self):
        """Update block size display based on selected bandwidth ratio"""
        block_size = self.block_size_var.get() if self.br_var.get() == "1/3" else self.block_size_var2.get()
        self.block_size_label.config(text=f"Block Size: {block_size}")

    def train_model(self):
        """Train DJSCC model"""
        try:
            # Load CIFAR-10 data
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            
            # Train DJSCC model
            self.status_var.set("Training DJSCC model...")
            train_djscc(self.train_snr_var.get(), x_train, y_train, x_test, y_test, 
                       self.block_size_var.get())
            
            self.check_model_files()
            self.status_var.set("Training complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training error: {str(e)}")
            self.status_var.set("Training failed!")
        
        finally:
            self.is_running = False
            self.train_button.config(state='normal')

    def run_analysis(self):
        """Execute DJSCC analysis"""
        try:
            if not os.path.exists('model_weights.h5'):
                messagebox.showwarning("Warning", "Model weight files not found. Please train the model first!")
                return

            self.results_text.delete("1.0", tk.END)  # Clear previous results
            
            # Test DJSCC model
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            
            # Get block size based on selected BR
            block_size = self.block_size_var.get() if self.br_var.get() == "1/3" else self.block_size_var2.get()
            
            # Test for selected SNR
            self.status_var.set(f"Testing DJSCC model (BR={self.br_var.get()})...")
            self.accuracy_value = test_djscc(self.test_snr_var.get(), x_test, y_test, block_size)
            
            # Display results
            self.display_results()
            self.status_var.set("Analysis complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis error: {str(e)}")
            self.status_var.set("Analysis failed!")
        
        finally:
            self.is_running = False
            self.test_button.config(state='normal')

    def display_results(self):
        """Display accuracy values"""
        result_text = f"Analysis Results:\n"
        result_text += f"Bandwidth Ratio: {self.br_var.get()}\n"
        result_text += f"Block Size: {self.block_size_var.get() if self.br_var.get() == '1/3' else self.block_size_var2.get()}\n"
        result_text += f"SNR: {self.test_snr_var.get():.1f} dB\n"
        result_text += f"Accuracy: {self.accuracy_value:.4f}\n"
        result_text += "-" * 50 + "\n"
        self.results_text.insert(tk.END, result_text)
        self.results_text.see(tk.END)

    def on_train_clicked(self):
        """Handle train button click"""
        if self.is_running:
            return
            
        self.is_running = True
        self.train_button.config(state='disabled')
        self.status_var.set("Starting training...")
        
        thread = threading.Thread(target=self.train_model)
        thread.daemon = True
        thread.start()

    def on_test_clicked(self):
        """Handle test button click"""
        if self.is_running:
            return
            
        self.is_running = True
        self.test_button.config(state='disabled')
        self.status_var.set("Starting analysis...")
        
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()

def main():
    root = tk.Tk()
    root.title("DJSCC Analysis")
    root.geometry("500x600")
    
    style = ttk.Style()
    style.configure('TLabelframe', padding=5)
    style.configure('TButton', padding=5)
    
    app = DJSCCAnalysisFrame_ac(root)
    app.pack(expand=True, fill="both", padx=10, pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    main()