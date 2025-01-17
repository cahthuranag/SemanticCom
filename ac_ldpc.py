import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import os
from ac_all import plot_accuracy_vs_snr_ldpc, build_classifier_model
from tensorflow.keras.datasets import cifar10

class LDPCAnalysisFrame_ac(ttk.Frame):
    # Define color scheme
    COLORS = {
        'primary': '#2c3e50',    # Dark blue-gray
        'secondary': '#3498db',  # Bright blue
        'accent': '#e74c3c',     # Red
        'bg': '#ecf0f1',        # Light gray
        'text': '#2c3e50',      # Dark blue-gray
        'success': '#27ae60',    # Green
        'warning': '#f39c12'     # Orange
    }
    
    def __init__(self, parent):
        super().__init__(parent)
        self.init_variables()
        self.setup_styles()
        self.create_widgets()
        self.check_model_files()
        
    def setup_styles(self):
        """Configure custom styles for widgets"""
        style = ttk.Style()
        
        # Configure main window
        style.configure('Main.TFrame', background=self.COLORS['bg'])
        
        # Configure labels
        style.configure('Custom.TLabel', 
                       background=self.COLORS['bg'],
                       foreground=self.COLORS['text'],
                       font=('Helvetica', 10))
                       
        # Configure label frames
        style.configure('Custom.TLabelframe', 
                       background=self.COLORS['bg'],
                       foreground=self.COLORS['primary'])
        style.configure('Custom.TLabelframe.Label', 
                       font=('Helvetica', 11, 'bold'),
                       foreground=self.COLORS['primary'])
        
        # Configure buttons
        style.configure('Train.TButton',
                       background=self.COLORS['secondary'],
                       foreground='white',
                       padding=(20, 10),
                       font=('Helvetica', 10, 'bold'))
        
        style.configure('Test.TButton',
                       background=self.COLORS['accent'],
                       foreground='white',
                       padding=(20, 10),
                       font=('Helvetica', 10, 'bold'))
                       
        # Configure entry fields
        style.configure('Custom.TEntry', 
                       fieldbackground='white',
                       padding=5)
                       
        # Configure combobox
        style.configure('Custom.TCombobox',
                       background='white',
                       padding=5)

    def init_variables(self):
        # LDPC variables
        self.k_var = tk.IntVar(value=1364)
        self.n_var = tk.IntVar(value=2048)
        self.k2_var = tk.IntVar(value=682)
        self.n2_var = tk.IntVar(value=1024)
        self.m_var = tk.IntVar(value=4)
        
        # SNR variable
        self.snr_var = tk.DoubleVar(value=10.0)
        
        # Bandwidth ratio selection
        self.br_var = tk.StringVar(value="1/3")
        
        # General variables
        self.status_var = tk.StringVar(value="Ready")
        self.is_running = False
        
        # Results storage
        self.accuracy_value = None

    def check_model_files(self):
        """Check if model weight files exist and enable/disable test button accordingly"""
        if os.path.exists('classifier_model_weights_ldpc.h5'):
            self.test_button.config(state='normal')
            self.status_var.set("Model files found. Ready for analysis.")
        else:
            self.test_button.config(state='disabled')
            self.status_var.set("Training required.")

    def create_widgets(self):
        self.configure(style='Main.TFrame')
        
        # Configure main grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Parameters Frame
        params_frame = ttk.LabelFrame(self, text="LDPC Parameters", style='Custom.TLabelframe')
        params_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        # Bandwidth Ratio Selection
        ttk.Label(params_frame, text="Bandwidth Ratio:", 
                 style='Custom.TLabel').grid(row=0, column=0, padx=5, pady=2, sticky="e")
        br_combo = ttk.Combobox(params_frame, textvariable=self.br_var,
                               values=["1/3", "1/6"], state="readonly", width=9,
                               style='Custom.TCombobox')
        br_combo.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        br_combo.bind('<<ComboboxSelected>>', self.on_br_changed)
        
        # Dynamic parameter frame
        self.dynamic_params_frame = ttk.Frame(params_frame, style='Main.TFrame')
        self.dynamic_params_frame.grid(row=1, column=0, columnspan=2, pady=5)
        self.update_parameter_inputs()
        
        # SNR input
        self.create_labeled_entry(params_frame, "SNR (dB):", self.snr_var, 2)

        # Results Frame
        results_frame = ttk.LabelFrame(self, text="Results", style='Custom.TLabelframe')
        results_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # Create Text widget for results
        self.results_text = tk.Text(results_frame, height=10, width=50,
                                  bg='white',
                                  fg=self.COLORS['text'],
                                  font=('Consolas', 10),
                                  relief='solid',
                                  borderwidth=1)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Control Frame
        control_frame = ttk.Frame(self, style='Main.TFrame')
        control_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        # Train and Test buttons
        self.train_button = ttk.Button(control_frame, text="Train Classifier",
                                     style='Train.TButton',
                                     command=self.on_train_clicked)
        self.train_button.pack(side="left", padx=5)
        
        self.test_button = ttk.Button(control_frame, text="Run Analysis",
                                    style='Test.TButton',
                                    command=self.on_test_clicked)
        self.test_button.pack(side="left", padx=5)
        
        # Status label
        status_label = ttk.Label(control_frame, textvariable=self.status_var,
                               style='Custom.TLabel')
        status_label.pack(side="right", padx=5)

    def create_labeled_entry(self, parent, label_text, variable, row, column_offset=0):
        """Helper method to create a labeled entry with custom styling"""
        ttk.Label(parent, text=label_text, 
                 style='Custom.TLabel').grid(row=row, column=0+column_offset, padx=5, pady=2, sticky="e")
        entry = ttk.Entry(parent, textvariable=variable, width=12, style='Custom.TEntry')
        entry.grid(row=row, column=1+column_offset, padx=5, pady=2, sticky="w")
        return entry

    def on_br_changed(self, event=None):
        """Handle bandwidth ratio selection change"""
        self.update_parameter_inputs()

    def update_parameter_inputs(self):
        """Update parameter inputs based on selected bandwidth ratio"""
        # Clear existing widgets
        for widget in self.dynamic_params_frame.winfo_children():
            widget.destroy()
            
        # Show relevant parameters based on selected BR
        if self.br_var.get() == "1/3":
            self.create_labeled_entry(self.dynamic_params_frame, "k:", self.k_var, 0)
            self.create_labeled_entry(self.dynamic_params_frame, "n:", self.n_var, 1)
        else:  # BR = 1/6
            self.create_labeled_entry(self.dynamic_params_frame, "k:", self.k2_var, 0)
            self.create_labeled_entry(self.dynamic_params_frame, "n:", self.n2_var, 1)
            
        self.create_labeled_entry(self.dynamic_params_frame, "m:", self.m_var, 2)

    def train_classifier(self):
        """Train classifier for LDPC"""
        try:
            # Load CIFAR-10 data
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            
            # Train classifier for LDPC
            self.status_var.set("Training classifier...")
            classifier_model = build_classifier_model()
            x_train = x_train.astype('float32') / 255.0
            classifier_model.fit(x_train, y_train, batch_size=128, epochs=200, 
                               validation_split=0.1, verbose=1)
            classifier_model.save_weights('classifier_model_weights_ldpc.h5')
            
            self.check_model_files()
            self.status_var.set("Training complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training error: {str(e)}")
            self.status_var.set("Training failed!")
        
        finally:
            self.is_running = False
            self.train_button.config(state='normal')

    def run_analysis(self):
        """Execute LDPC analysis"""
        try:
            if not os.path.exists('classifier_model_weights_ldpc.h5'):
                messagebox.showwarning("Warning", "Classifier model weights not found. Please train the model first!")
                return

            self.results_text.delete("1.0", tk.END)  # Clear previous results
            
            # Get parameters based on selected BR
            br = 1/3 if self.br_var.get() == "1/3" else 1/6
            k = self.k_var.get() if br == 1/3 else self.k2_var.get()
            n = self.n_var.get() if br == 1/3 else self.n2_var.get()
            
            # Get SNR value
            snr = self.snr_var.get()
            snr_array = np.array([snr])
            
            # Run LDPC analysis
            self.status_var.set(f"Running LDPC analysis (BR={self.br_var.get()})...")
            self.accuracy_value = plot_accuracy_vs_snr_ldpc(
                br, k, n, self.m_var.get(), snr_array)
            
            # Display results
            self.display_results(snr)
            self.status_var.set("Analysis complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis error: {str(e)}")
            self.status_var.set("Analysis failed!")
        
        finally:
            self.is_running = False
            self.test_button.config(state='normal')

    def display_results(self, snr):
        """Display accuracy values"""
        result_text = f"Analysis Results:\n"
        result_text += f"Bandwidth Ratio: {self.br_var.get()}\n"
        result_text += f"SNR: {snr:.1f} dB\n"
        result_text += f"Accuracy: {self.accuracy_value[0]:.4f}\n"
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
        
        thread = threading.Thread(target=self.train_classifier)
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
    root.title("LDPC Analysis")
    root.geometry("500x600")
    
    # Set window background color
    root.configure(bg=LDPCAnalysisFrame_ac.COLORS['bg'])
    
    app = LDPCAnalysisFrame_ac(root)
    app.pack(expand=True, fill="both", padx=10, pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    main()