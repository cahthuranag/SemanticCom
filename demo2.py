# system_gui.py

import tkinter as tk
from tkinter import ttk
import numpy as np
from psnr_analysis import PSNRAnalysisFrame

class SystemModelFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_canvas()
        self.draw_system_model()
        
    def create_canvas(self):
        self.canvas = tk.Canvas(self, bg='white', width=1200, height=500)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
    def draw_system_model(self):
        self.canvas.delete("all")
        
        # Box dimensions and positions
        tx_width = 400
        tx_height = 250
        rx_width = 450
        rx_height = 330
        
        tx_x = 50
        tx_y = 100
        rx_x = 600
        rx_y = 100
        
        # Draw rounded rectangles for transmitter and receiver
        self.draw_rounded_rectangle(tx_x, tx_y, tx_width, tx_height)
        self.draw_rounded_rectangle(rx_x, rx_y, rx_width, rx_height)
        
        # Draw titles
        self.canvas.create_text(tx_x + tx_width//2, tx_y - 30, 
                              text="Transmitter", font=('Arial', 16))
        self.canvas.create_text(rx_x + rx_width//2, rx_y - 30, 
                              text="Receiver", font=('Arial', 16))
        
        # Draw input image
        img_size = 100
        self.draw_image_placeholder(tx_x + 30, tx_y + 60, img_size, img_size, "Input Image")
        
        # Draw encoder network
        self.draw_encoder(tx_x + 150, tx_y + 130)
        self.canvas.create_text(tx_x + 150, tx_y + 40, 
                              text="Encoder", font=('Arial', 12))
        
        # Draw reshaping blocks
        self.draw_block(tx_x + tx_width - 100, tx_y + 100, 
                       80, 80, "Reshaping &\nNormalization", "#228B22")  # Dark green
        
        # Draw wireless channel
        channel_width = 120
        channel_height = 80
        channel_x = (tx_x + tx_width + rx_x)//2 - channel_width//2
        channel_y = tx_y + tx_height//2 - channel_height//2
        self.draw_block(channel_x, channel_y, channel_width, channel_height, 
                       "Wireless Channel", "#00BFFF")  # Light blue
        
        # Draw reshaping block in receiver
        self.draw_block(rx_x + 20, rx_y + 100, 
                       80, 80, "Reshaping", "#1E90FF")  # Blue
        
        # Draw decoder network
        self.draw_decoder(rx_x + 150, rx_y + 110)
        self.canvas.create_text(rx_x + 150, rx_y + 40, 
                              text="Decoder", font=('Arial', 12))
        
        # Draw output image
        self.draw_image_placeholder(rx_x + rx_width - 130, rx_y + 60, img_size, img_size, "Output Image")
        
        # Draw classifier network
        self.draw_classifier(rx_x + 150, rx_y + 240)
        self.canvas.create_text(rx_x + 150, rx_y + 180, 
                              text="Classifier", font=('Arial', 12))
        
        # Draw labels
        label_x = rx_x + rx_width - 80
        base_y = rx_y + 210
        self.draw_labels(label_x, base_y)
        
        # Draw arrows and variable labels
        self.draw_arrows_and_labels(tx_x, tx_width, rx_x, channel_x, channel_width, 
                                  tx_y + tx_height//2)
        
    def draw_rounded_rectangle(self, x, y, width, height, radius=30):
        points = [
            x + radius, y,
            x + width - radius, y,
            x + width, y,
            x + width, y + radius,
            x + width, y + height - radius,
            x + width, y + height,
            x + width - radius, y + height,
            x + radius, y + height,
            x, y + height,
            x, y + height - radius,
            x, y + radius,
            x, y
        ]
        self.canvas.create_polygon(points, smooth=True, fill='white', outline='black', width=2)
        
    def draw_image_placeholder(self, x, y, width, height, text):
        # Draw border
        self.canvas.create_rectangle(x, y, x + width, y + height, 
                                   fill='white', outline='black', width=2)
        
        # Draw grid of small images (5x5)
        grid_size = 5
        cell_w = width / grid_size
        cell_h = height / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                self.canvas.create_rectangle(
                    x + i*cell_w, y + j*cell_h,
                    x + (i+1)*cell_w, y + (j+1)*cell_h,
                    outline='gray80'
                )
        
        # Draw label
        self.canvas.create_text(x + width//2, y + height + 25, 
                              text=text, font=('Arial', 12))
        
    def draw_network(self, x, y, layers, color='#4169E1', connection_color='#87CEEB'):
        prev_layer = []
        layer_spacing = 40
        neuron_spacing = 20
        
        for i, n_neurons in enumerate(layers):
            current_layer = []
            layer_height = (n_neurons - 1) * neuron_spacing
            start_y = y - layer_height/2
            
            for j in range(n_neurons):
                pos_x = x + i * layer_spacing
                pos_y = start_y + j * neuron_spacing
                
                # Draw neuron
                self.canvas.create_oval(pos_x-5, pos_y-5, pos_x+5, pos_y+5, 
                                     fill='white', outline=color, width=2)
                current_layer.append((pos_x, pos_y))
                
                # Draw connections to previous layer
                if prev_layer:
                    for prev_x, prev_y in prev_layer:
                        self.canvas.create_line(prev_x, prev_y, pos_x, pos_y, 
                                             fill=connection_color, width=1)
            
            prev_layer = current_layer
            
    def draw_encoder(self, x, y):
        self.draw_network(x, y, [8, 6, 4, 3])
        
    def draw_decoder(self, x, y):
        self.draw_network(x, y, [3, 4, 6, 8])
        
    def draw_classifier(self, x, y):
        self.draw_network(x, y, [4, 3, 2], color='#32CD32', connection_color='#90EE90')
        
    def draw_block(self, x, y, width, height, text, color):
        self.canvas.create_rectangle(x, y, x+width, y+height, 
                                   fill=color, outline='black', width=2)
        self.canvas.create_text(x+width/2, y+height/2, 
                              text=text, fill='white', 
                              font=('Arial', 10, 'bold'), justify='center')
        
    def draw_labels(self, x, y):
        self.canvas.create_text(x, y, text="Label 1", anchor='w', font=('Arial', 12))
        self.canvas.create_text(x, y + 25, text="Label 2", anchor='w', font=('Arial', 12))
        
        # Draw dots
        for i in range(3):
            self.canvas.create_text(x + 30, y + 60 + i*5, 
                                  text=".", anchor='center', font=('Arial', 12))
        
        self.canvas.create_text(x, y + 90, text="Label 10", anchor='w', font=('Arial', 12))
        
    def draw_arrows_and_labels(self, tx_x, tx_width, rx_x, channel_x, channel_width, y_pos):
        # Draw main flow arrows
        arrow_style = {'arrow': tk.LAST, 'width': 2, 'fill': 'black'}
        
        # Transmitter to Channel
        self.canvas.create_line(tx_x + tx_width - 20, y_pos,
                              channel_x, y_pos, **arrow_style)
        
        # Channel to Receiver
        self.canvas.create_line(channel_x + channel_width, y_pos,
                              rx_x + 20, y_pos, **arrow_style)


class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("DJSCC System")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # System Model tab
        system_frame = SystemModelFrame(notebook)
        notebook.add(system_frame, text="System Model")
        
        # PSNR Analysis tab
        analysis_frame = PSNRAnalysisFrame(notebook)
        notebook.add(analysis_frame, text="PSNR Analysis")

if __name__ == "__main__":
    app = MainApplication()
    # Set window size
    app.geometry("1000x500")
    app.mainloop()