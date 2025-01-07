import tkinter as tk
from tkinter import ttk
import numpy as np
from psnr_djscc import DJSCCAnalysisFrame
from psnr_ldpc import LDPCAnalysisFrame
from system import SystemModelFrame
from ac_djccc import DJSCCAnalysisFrame_ac
from ac_ldpc import LDPCAnalysisFrame_ac

class CombinedAccuracyFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()
    
    def create_widgets(self):
        # Create main frame that will hold both analyses side by side
        main_frame = ttk.Frame(self)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left side - DJSCC Accuracy Analysis
        djscc_frame = DJSCCAnalysisFrame_ac(main_frame)
        djscc_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Right side - LDPC Accuracy Analysis
        ldpc_frame = LDPCAnalysisFrame_ac(main_frame)
        ldpc_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Configure grid weights
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

class SystemAnalysisFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()
    
    def create_widgets(self):
        # Create main frame that will hold both systems side by side
        main_frame = ttk.Frame(self)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left side - DJSCC
        djscc_frame = DJSCCAnalysisFrame(main_frame)
        djscc_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Right side - LDPC
        ldpc_frame = LDPCAnalysisFrame(main_frame)
        ldpc_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Configure grid weights
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Semantic Communication System Simulation Model")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # System Model tab
        system_frame = SystemModelFrame(notebook)
        notebook.add(system_frame, text="System Model")
        
        # Combined Analysis tab
        analysis_frame = SystemAnalysisFrame(notebook)
        notebook.add(analysis_frame, text="PSNR Analysis")
        
        # Accuracy Analysis tab with both DJSCC and LDPC
        accuracy_frame = CombinedAccuracyFrame(notebook)
        notebook.add(accuracy_frame, text="Accuracy Analysis")

if __name__ == "__main__":
    app = MainApplication()
    app.geometry("1200x800")  # Increased size to accommodate both systems
    app.mainloop()