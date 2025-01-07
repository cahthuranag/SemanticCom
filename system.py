# system_gui.py

import tkinter as tk
from tkinter import ttk
import numpy as np


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
        rx_height = 380  # Increased height to accommodate labels
        
        tx_x = 50
        tx_y = 100
        rx_x = 600
        rx_y = 100
        
        # Draw rounded rectangles for transmitter and receiver with light background
        self.draw_rounded_rectangle(tx_x, tx_y, tx_width, tx_height, fill='#F0F8FF')  # Light blue background
        self.draw_rounded_rectangle(rx_x, rx_y, rx_width, rx_height, fill='#F0FFF0')  # Light green background
        
        # Draw titles with gradient background
        self.draw_title(tx_x + tx_width//2, tx_y - 30, "Transmitter", '#4169E1')  # Royal blue
        self.draw_title(rx_x + rx_width//2, rx_y - 30, "Receiver", '#2E8B57')  # Sea green
        
        # Draw input image with colored border
        self.draw_image_placeholder(tx_x + 30, tx_y + 60, 100, 100, "Input Image", '#4169E1')  # Royal blue
        
        # Draw encoder network with colorful nodes
        self.draw_encoder(tx_x + 150, tx_y + 130, '#4169E1', '#87CEEB')  # Royal blue nodes, light blue connections
        self.canvas.create_text(tx_x + 150, tx_y + 40, text="Encoder", 
                              font=('Arial', 12, 'bold'), fill='#4169E1')
        
        # Draw reshaping blocks with gradient
        self.draw_block(tx_x + tx_width - 100, tx_y + 100, 80, 80, 
                       "Reshaping &\nNormalization", '#3CB371')  # Medium sea green
        
        # Draw wireless channel with glossy effect
        channel_width = 120
        channel_height = 80
        channel_x = (tx_x + tx_width + rx_x)//2 - channel_width//2
        channel_y = tx_y + tx_height//2 - channel_height//2
        self.draw_block(channel_x, channel_y, channel_width, channel_height, 
                       "Wireless Channel", '#00BFFF')  # Deep sky blue
        
        # Draw reshaping block in receiver with gradient
        self.draw_block(rx_x + 20, rx_y + 100, 80, 80, "Reshaping", '#4682B4')  # Steel blue
        
        # Draw decoder network with colorful nodes
        self.draw_decoder(rx_x + 150, rx_y + 110, '#2E8B57', '#98FB98')  # Sea green nodes, pale green connections
        self.canvas.create_text(rx_x + 150, rx_y + 40, text="Decoder", 
                              font=('Arial', 12, 'bold'), fill='#2E8B57')
        
        # Draw output image with colored border - same style as input
        self.draw_image_placeholder(rx_x + rx_width - 130, rx_y + 60, 100, 100, 
                                  "Output Image", '#2E8B57')  # Sea green
        
        # Draw classifier network with different colors
        self.draw_classifier(rx_x + 150, rx_y + 240, '#8B008B', '#DDA0DD')  # Dark magenta nodes, plum connections
        self.canvas.create_text(rx_x + 150, rx_y + 180, text="Classifier", 
                              font=('Arial', 12, 'bold'), fill='#8B008B')
        
        # Draw labels with icons inside receiver box
        self.draw_labels(rx_x + rx_width - 180, rx_y + 210, '#4B0082')  # Indigo
        
        # Draw arrows and variable labels with gradient
        self.draw_arrows_and_labels(tx_x, tx_width, rx_x, channel_x, channel_width, 
                                  tx_y + tx_height//2)

    def draw_rounded_rectangle(self, x, y, width, height, radius=30, fill='white'):
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
        return self.canvas.create_polygon(points, smooth=True, fill=fill, 
                                        outline='#333333', width=2)  # Dark gray outline

    def draw_title(self, x, y, text, color):
        self.canvas.create_text(x, y, text=text, font=('Arial', 16, 'bold'), 
                              fill=color)
        
    def draw_image_placeholder(self, x, y, width, height, text, border_color):
        # Draw border with color
        self.canvas.create_rectangle(x, y, x + width, y + height, 
                                   fill='#F0F8FF', outline=border_color, width=2)
        
        # Draw a colorful pattern (4x4 grid with different colors)
        grid_size = 4
        cell_w = width / grid_size
        cell_h = height / grid_size
        
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',  # Row 1
            '#FFEEAD', '#D4A5A5', '#9B59B6', '#3498DB',  # Row 2
            '#E74C3C', '#2ECC71', '#F1C40F', '#1ABC9C',  # Row 3
            '#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD'   # Row 4
        ]
        
        for i in range(grid_size):
            for j in range(grid_size):
                color_idx = i * grid_size + j
                self.canvas.create_rectangle(
                    x + i*cell_w, y + j*cell_h,
                    x + (i+1)*cell_w, y + (j+1)*cell_h,
                    fill=colors[color_idx],
                    outline='white',
                    width=1
                )
                
                # Add a simple pattern inside each cell
                cell_center_x = x + i*cell_w + cell_w/2
                cell_center_y = y + j*cell_h + cell_h/2
                
                if (i + j) % 4 == 0:  # Circle pattern
                    self.canvas.create_oval(
                        cell_center_x - 10, cell_center_y - 10,
                        cell_center_x + 10, cell_center_y + 10,
                        fill='white', outline='white', width=1
                    )
                elif (i + j) % 4 == 1:  # Cross pattern
                    size = 8
                    self.canvas.create_line(
                        cell_center_x - size, cell_center_y - size,
                        cell_center_x + size, cell_center_y + size,
                        fill='white', width=2
                    )
                    self.canvas.create_line(
                        cell_center_x - size, cell_center_y + size,
                        cell_center_x + size, cell_center_y - size,
                        fill='white', width=2
                    )
                elif (i + j) % 4 == 2:  # Square pattern
                    size = 8
                    self.canvas.create_rectangle(
                        cell_center_x - size, cell_center_y - size,
                        cell_center_x + size, cell_center_y + size,
                        fill='white', outline='white'
                    )
                else:  # Diamond pattern
                    size = 8
                    self.canvas.create_polygon(
                        cell_center_x, cell_center_y - size,
                        cell_center_x + size, cell_center_y,
                        cell_center_x, cell_center_y + size,
                        cell_center_x - size, cell_center_y,
                        fill='white', outline='white'
                    )
        
        # Add a glossy effect
        self.canvas.create_polygon(
            x, y,
            x + width, y,
            x + width*0.8, y + height*0.2,
            x + width*0.2, y + height*0.2,
            fill='white', stipple='gray25'
        )
        
        # Draw label with enhanced styling
        self.canvas.create_text(
            x + width//2, 
            y + height + 25,
            text=text,
            font=('Arial', 12, 'bold'),
            fill=border_color
        )
        
    def draw_network(self, x, y, layers, node_color, connection_color):
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
                
                # Draw neuron with gradient fill
                self.canvas.create_oval(pos_x-5, pos_y-5, pos_x+5, pos_y+5, 
                                     fill='white', outline=node_color, width=2)
                current_layer.append((pos_x, pos_y))
                
                # Draw connections with transparency
                if prev_layer:
                    for prev_x, prev_y in prev_layer:
                        self.canvas.create_line(prev_x, prev_y, pos_x, pos_y, 
                                             fill=connection_color, width=1)
            
            prev_layer = current_layer
            
    def draw_encoder(self, x, y, node_color, connection_color):
        self.draw_network(x, y, [8, 6, 4, 3], node_color, connection_color)
        
    def draw_decoder(self, x, y, node_color, connection_color):
        self.draw_network(x, y, [3, 4, 6, 8], node_color, connection_color)
        
    def draw_classifier(self, x, y, node_color, connection_color):
        self.draw_network(x, y, [4, 3, 2], node_color, connection_color)
        
    def draw_block(self, x, y, width, height, text, color):
        # Draw main block with gradient effect
        self.canvas.create_rectangle(x, y, x+width, y+height, 
                                   fill=color, outline='#333333', width=2)
        
        # Draw text with shadow effect
        offset = 1
        self.canvas.create_text(x+width/2+offset, y+height/2+offset, 
                              text=text, fill='#333333', 
                              font=('Arial', 10, 'bold'), justify='center')
        self.canvas.create_text(x+width/2, y+height/2, 
                              text=text, fill='white', 
                              font=('Arial', 10, 'bold'), justify='center')
        
    def draw_labels(self, x, y, color):
        # Define labels with their corresponding symbols
        labels = [
            ("Label 1: Car", self.draw_car),
            ("Label 2: House", self.draw_house),
            ("Label 3: Tree", self.draw_tree),
            ("Label 4: Bird", self.draw_bird)
        ]
        
        spacing = 35  # Increased spacing for icons
        
        for i, (label, draw_func) in enumerate(labels):
            # Draw label text
            self.canvas.create_text(x, y + i*spacing, text=label, 
                                  anchor='w', font=('Arial', 12, 'bold'), fill=color)
            
            # Draw icon next to label
            icon_x = x + 100  # Position icon after text
            icon_y = y + i*spacing - 10  # Align with text
            draw_func(icon_x, icon_y)
            
        # Draw ellipsis
        self.canvas.create_text(x, y + 4*spacing, text="...", 
                              anchor='w', font=('Arial', 12, 'bold'), fill=color)

    def draw_car(self, x, y):
        # Draw car body
        self.canvas.create_rectangle(x, y+10, x+30, y+20, fill='#FF6B6B', outline='black')
        self.canvas.create_rectangle(x+5, y+5, x+25, y+10, fill='#FF6B6B', outline='black')
        # Draw wheels
        self.canvas.create_oval(x+5, y+18, x+12, y+25, fill='black')
        self.canvas.create_oval(x+18, y+18, x+25, y+25, fill='black')

    def draw_house(self, x, y):
        # Draw house body
        self.canvas.create_rectangle(x+5, y+12, x+25, y+25, fill='#4ECDC4', outline='black')
        # Draw roof
        self.canvas.create_polygon(x, y+12, x+15, y, x+30, y+12, fill='#FF9999', outline='black')
        # Draw door
        self.canvas.create_rectangle(x+12, y+18, x+18, y+25, fill='#8B4513')

    def draw_tree(self, x, y):
        # Draw trunk
        self.canvas.create_rectangle(x+12, y+15, x+18, y+25, fill='#8B4513', outline='black')
        # Draw leaves
        self.canvas.create_oval(x+5, y, x+25, y+20, fill='#228B22', outline='black')

    def draw_bird(self, x, y):
        # Draw body
        self.canvas.create_oval(x+5, y+10, x+20, y+20, fill='#87CEEB', outline='black')
        # Draw head
        self.canvas.create_oval(x+15, y+5, x+25, y+15, fill='#87CEEB', outline='black')
        # Draw wing
        self.canvas.create_arc(x+8, y+8, x+18, y+18, start=30, extent=120, fill='#4682B4')
        
    def draw_arrows_and_labels(self, tx_x, tx_width, rx_x, channel_x, channel_width, y_pos):
        # Draw animated arrows
        arrow_style = {'arrow': tk.LAST, 'width': 2, 'fill': '#4169E1'}  # Royal blue arrows
        
        # Transmitter to Channel
        self.canvas.create_line(tx_x + tx_width - 20, y_pos,
                              channel_x, y_pos, **arrow_style)
        
        # Channel to Receiver
        self.canvas.create_line(channel_x + channel_width, y_pos,
                              rx_x + 20, y_pos, **arrow_style)
