import sys
import os
import numpy as np
import json
import subprocess
import re
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QListWidget, QListWidgetItem, QMessageBox, QGridLayout
)
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from sympy import sympify, symbols
from sympy.utilities.lambdify import lambdify

class AdvancedCalculator(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Advanced Graphing Calculator")
        self.setGeometry(100, 100, 1200, 800)
        
        self.plotted_equations = []
        self.plotted_data = []
        self.plot_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        self.color_index = 0

        self.selected_equation_for_points = None
        self.moving_point_index = -1
        self.moving_point_original_pos = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(300)

        self.equation_input = QLineEdit()
        self.equation_input.setPlaceholderText("Enter equation, e.g., sin(x) / x")
        self.add_button = QPushButton("Plot Equation")
        self.clear_equations_button = QPushButton("Clear Equations")
        self.equation_list_widget = QListWidget()

        self.add_button.clicked.connect(self.plot_equation)
        self.equation_input.returnPressed.connect(self.plot_equation)
        self.clear_equations_button.clicked.connect(self.clear_equations)

        left_layout.addWidget(QLabel("Equations:"))
        left_layout.addWidget(self.equation_input)
        left_layout.addWidget(self.add_button)
        left_layout.addWidget(self.clear_equations_button)
        left_layout.addWidget(self.equation_list_widget)

        # Equation selection and point placement
        equation_interaction_layout = QHBoxLayout()
        self.select_equation_button = QPushButton("Select Equation")
        self.select_equation_button.clicked.connect(self.select_equation_for_points)
        equation_interaction_layout.addWidget(self.select_equation_button)

        self.add_point_to_equation_button = QPushButton("Add Point to Eq.")
        self.add_point_to_equation_button.setCheckable(True)
        self.add_point_to_equation_button.clicked.connect(self.toggle_add_point_to_equation_mode)
        equation_interaction_layout.addWidget(self.add_point_to_equation_button)
        left_layout.addLayout(equation_interaction_layout)

        self.data_input = QLineEdit()
        self.data_input.setPlaceholderText("e.g., (1, 2), (3, 4) or (1,2);(3,4)")
        self.plot_data_button = QPushButton("Plot Data")
        self.clear_data_button = QPushButton("Clear Data")
        self.data_list_widget = QListWidget()

        self.plot_data_button.clicked.connect(self.plot_data)
        self.data_input.returnPressed.connect(self.plot_data)
        self.clear_data_button.clicked.connect(self.clear_data)

        self.add_point_mode_button = QPushButton("Add Point w/ Cursor")
        self.add_point_mode_button.setCheckable(True)
        self.add_point_mode_button.clicked.connect(self.toggle_add_point_mode)

        self.calc_angle_button = QPushButton("Angle Between Lines")
        self.calc_angle_button.clicked.connect(self.calculate_angle_between_lines)

        self.plot_polyline_button = QPushButton("Plot Continuous Line")
        self.plot_polyline_button.clicked.connect(self.plot_polyline)

        self.plot_infinite_line_button = QPushButton("Plot Infinite Line")
        self.plot_infinite_line_button.clicked.connect(self.plot_infinite_line)

        data_buttons_layout = QGridLayout()
        data_buttons_layout.addWidget(self.plot_data_button, 0, 0)
        data_buttons_layout.addWidget(self.clear_data_button, 0, 1)
        data_buttons_layout.addWidget(self.plot_polyline_button, 1, 0)
        data_buttons_layout.addWidget(self.plot_infinite_line_button, 1, 1)


        left_layout.addWidget(QLabel("Points/Lines:"))
        left_layout.addWidget(self.data_input)
        left_layout.addLayout(data_buttons_layout)
        
        interactive_buttons_layout = QHBoxLayout()
        interactive_buttons_layout.addWidget(self.add_point_mode_button)
        interactive_buttons_layout.addWidget(self.calc_angle_button)

        self.move_point_button = QPushButton("Move Point")
        self.move_point_button.setCheckable(True)
        self.move_point_button.clicked.connect(self.toggle_move_point_mode)
        interactive_buttons_layout.addWidget(self.move_point_button)

        left_layout.addLayout(interactive_buttons_layout)

        left_layout.addWidget(self.data_list_widget)

        # --- LLM Section ---
        llm_label = QLabel("\nAsk a question about the graph:")
        self.llm_question_input = QLineEdit()
        self.llm_question_input.setPlaceholderText("e.g., why does x/sin(x) look like this?")
        self.ask_llm_button = QPushButton("Ask Gemma")

        left_layout.addWidget(llm_label)
        left_layout.addWidget(self.llm_question_input)
        left_layout.addWidget(self.ask_llm_button)

        self.ask_llm_button.clicked.connect(self.handle_llm_question)

        # --- Right Panel ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'Y')
        self.plot_widget.setLabel('bottom', 'X')
        self.plot_widget.addLegend()
        right_layout.addWidget(self.plot_widget)

        # --- Control Panel ---
        control_panel = QWidget()
        control_layout = QGridLayout(control_panel)

        self.x_min_input = QLineEdit("-10")
        self.x_max_input = QLineEdit("10")
        self.y_min_input = QLineEdit("-10")
        self.y_max_input = QLineEdit("10")
        self.resolution_input = QLineEdit("2000")
        self.redraw_button = QPushButton("Redraw")

        control_layout.addWidget(QLabel("X-Min:"), 0, 0)
        control_layout.addWidget(self.x_min_input, 1, 0)
        control_layout.addWidget(QLabel("X-Max:"), 0, 1)
        control_layout.addWidget(self.x_max_input, 1, 1)
        control_layout.addWidget(QLabel("Y-Min:"), 0, 2)
        control_layout.addWidget(self.y_min_input, 1, 2)
        control_layout.addWidget(QLabel("Y-Max:"), 0, 3)
        control_layout.addWidget(self.y_max_input, 1, 3)
        control_layout.addWidget(QLabel("Resolution (points):"), 0, 4)
        control_layout.addWidget(self.resolution_input, 1, 4)
        control_layout.addWidget(self.redraw_button, 1, 5)

        # Zoom and Pan buttons
        self.zoom_in_button = QPushButton("Zoom In (+)")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        control_layout.addWidget(self.zoom_in_button, 2, 0)

        self.zoom_out_button = QPushButton("Zoom Out (-)")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        control_layout.addWidget(self.zoom_out_button, 2, 1)

        self.pan_left_button = QPushButton("Pan Left (<)")
        self.pan_left_button.clicked.connect(self.pan_left)
        control_layout.addWidget(self.pan_left_button, 2, 2)

        self.pan_right_button = QPushButton("Pan Right (>)")
        self.pan_right_button.clicked.connect(self.pan_right)
        control_layout.addWidget(self.pan_right_button, 2, 3)

        self.pan_up_button = QPushButton("Pan Up (^)")
        self.pan_up_button.clicked.connect(self.pan_up)
        control_layout.addWidget(self.pan_up_button, 3, 0)

        self.pan_down_button = QPushButton("Pan Down (v)")
        self.pan_down_button.clicked.connect(self.pan_down)
        control_layout.addWidget(self.pan_down_button, 3, 1)
        right_layout.addWidget(control_panel)

        self.redraw_button.clicked.connect(self.redraw_plots)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

    def plot_equation(self):
        raw_text = self.equation_input.text()
        if not raw_text:
            return

        equation_to_parse = raw_text
        # If user enters "y = sin(x)", only parse the part after "="
        if '=' in equation_to_parse:
            try:
                equation_to_parse = equation_to_parse.split('=', 1)[1].strip()
            except IndexError:
                QMessageBox.critical(self, "Error", f"Invalid equation format: {raw_text}")
                return
        
        if not equation_to_parse:
            QMessageBox.critical(self, "Error", f"No expression found after '=' in '{raw_text}'.")
            return

        try:
            # --- Pre-processing for user-friendly syntax ---
            # 1. Replace ^ with ** for exponents
            equation_to_parse = equation_to_parse.replace('^', '**')
            # 2. Add implicit multiplication, e.g., 4x -> 4*x or (x+1)(x-1) -> (x+1)*(x-1)
            equation_to_parse = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', equation_to_parse)
            equation_to_parse = re.sub(r'(\))([a-zA-Z(])', r'\1*\2', equation_to_parse)

            x = symbols('x')
            expr = sympify(equation_to_parse)
            func = lambdify(x, expr, 'numpy')

            color = self.plot_colors[self.color_index % len(self.plot_colors)]
            pen = pg.mkPen(color=color, width=2)
            
            # Store the original full text for display, but the parsed expression for calculation
            self.plotted_equations.append({'text': raw_text, 'expr': expr, 'pen': pen})
            self.color_index += 1
            
            self.equation_list_widget.addItem(raw_text)
            self.equation_input.clear()
            self.redraw_plots()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid equation: {e}")

    def select_equation_for_points(self):
        selected_items = self.equation_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Selection Error", "Please select an equation from the list first.")
            self.selected_equation_for_points = None
            return

        selected_text = selected_items[0].text()
        # Find the corresponding equation data
        for eq_data in self.plotted_equations:
            if eq_data['text'] == selected_text:
                self.selected_equation_for_points = eq_data
                QMessageBox.information(self, "Equation Selected", f"'{selected_text}' selected for point placement.")
                return
        self.selected_equation_for_points = None
        QMessageBox.critical(self, "Error", "Selected equation not found in plotted equations.")

    def toggle_add_point_to_equation_mode(self, checked):
        if checked:
            if self.selected_equation_for_points is None:
                QMessageBox.warning(self, "Mode Error", "Please select an equation first before enabling 'Add Point to Eq.' mode.")
                self.add_point_to_equation_button.setChecked(False)
                return
            self.setCursor(QtGui.QCursor(pg.QtCore.Qt.CursorShape.CrossCursor))
        else:
            self.setCursor(QtGui.QCursor(pg.QtCore.Qt.CursorShape.ArrowCursor))

    def toggle_move_point_mode(self, checked):
        if checked:
            self.setCursor(QtGui.QCursor(pg.QtCore.Qt.CursorShape.CrossCursor))
        else:
            self.setCursor(QtGui.QCursor(pg.QtCore.Qt.CursorShape.ArrowCursor))

    def plot_data(self):
        raw_text = self.data_input.text().strip()
        if not raw_text:
            return

        try:
            color = self.plot_colors[self.color_index % len(self.plot_colors)]
            pen = pg.mkPen(color=color, width=2)
            
            # Try to parse as a line: (x1, y1); (x2, y2)
            line_match = re.match(r'\s*\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)\s*;\s*\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)\s*', raw_text)
            if line_match:
                x1, y1, x2, y2 = map(float, line_match.groups())
                list_item = QListWidgetItem(raw_text)
                self.data_list_widget.addItem(list_item)
                self.plotted_data.append({'type': 'line', 'data': ([x1, x2], [y1, y2]), 'pen': pen, 'text': raw_text, 'list_item': list_item})
                self.data_input.clear()
                self.color_index += 1
                self.redraw_plots()
                return

            # Try to parse as a list of points: (x1, y1), (x2, y2), ...
            point_list = re.findall(r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)', raw_text)
            if point_list:
                points = np.array([(float(p[0]), float(p[1])) for p in point_list])
                list_item = QListWidgetItem(raw_text)
                self.data_list_widget.addItem(list_item)
                self.plotted_data.append({'type': 'points', 'data': points, 'pen': pen, 'text': raw_text, 'list_item': list_item})
                self.data_input.clear()
                self.color_index += 1
                self.redraw_plots()
                return

            QMessageBox.critical(self, "Error", "Invalid data format. Use e.g., '(1, 2), (3, 4)' for points or '(1, 2); (3, 4)' for a line.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not plot data: {e}")

    def plot_polyline(self):
        raw_text = self.data_input.text().strip()
        if not raw_text:
            return

        try:
            color = self.plot_colors[self.color_index % len(self.plot_colors)]
            pen = pg.mkPen(color=color, width=2)
            
            point_list = re.findall(r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)', raw_text)
            if point_list and len(point_list) > 1:
                points = np.array([(float(p[0]), float(p[1])) for p in point_list])
                list_item = QListWidgetItem(raw_text)
                self.data_list_widget.addItem(list_item)
                self.plotted_data.append({'type': 'polyline', 'data': points, 'pen': pen, 'text': raw_text, 'list_item': list_item})
                self.data_input.clear()
                self.color_index += 1
                self.redraw_plots()
            else:
                QMessageBox.critical(self, "Error", "Invalid data format or not enough points. Use e.g., '(1, 2), (3, 4), (5, 1)' for a continuous line.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not plot continuous line: {e}")

    def plot_infinite_line(self):
        raw_text = self.data_input.text().strip()
        if not raw_text:
            return

        try:
            color = self.plot_colors[self.color_index % len(self.plot_colors)]
            pen = pg.mkPen(color=color, width=2)
            
            line_match = re.match(r'\s*\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)\s*;\s*\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)\s*', raw_text)
            if line_match:
                x1, y1, x2, y2 = map(float, line_match.groups())
                
                # Calculate angle
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                
                list_item = QListWidgetItem(raw_text + " (infinite)")
                self.data_list_widget.addItem(list_item)
                self.plotted_data.append({'type': 'infinite_line', 'pos': [x1, y1], 'angle': angle, 'pen': pen, 'text': raw_text + " (infinite)", 'list_item': list_item})
                self.data_input.clear()
                self.color_index += 1
                self.redraw_plots()
            else:
                QMessageBox.critical(self, "Error", "Invalid data format for infinite line. Use e.g., '(1, 2); (3, 4)'.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not plot infinite line: {e}")

    def clear_equations(self):
        self.plotted_equations.clear()
        self.equation_list_widget.clear()
        self.selected_equation_for_points = None # Clear selected equation
        self.redraw_plots()

    def clear_data(self):
        self.plotted_data.clear()
        self.data_list_widget.clear()
        self.redraw_plots()

    def redraw_plots(self):
        try:
            x_min = float(self.x_min_input.text())
            x_max = float(self.x_max_input.text())
            y_min = float(self.y_min_input.text())
            y_max = float(self.y_max_input.text())
            resolution = int(self.resolution_input.text())

            if x_min >= x_max or y_min >= y_max or resolution <= 1:
                QMessageBox.critical(self, "Error", "Invalid range or resolution.")
                return

            self.plot_widget.clear()
            self.plot_widget.setXRange(x_min, x_max, padding=0)
            self.plot_widget.setYRange(y_min, y_max, padding=0)

            # Plot equations
            x_vals = np.linspace(x_min, x_max, resolution)
            x = symbols('x')
            for eq_data in self.plotted_equations:
                func = lambdify(x, eq_data['expr'], 'numpy')
                y_vals = func(x_vals)
                y_vals[np.isinf(y_vals)] = np.nan
                self.plot_widget.plot(x_vals, y_vals, pen=eq_data['pen'], name=eq_data['text'])

            # Plot data (points and lines)
            for i, data_item in enumerate(self.plotted_data):
                if data_item['type'] == 'line':
                    x_coords, y_coords = data_item['data']
                    # Use PlotDataItem for movable lines
                    plot_item = pg.PlotDataItem(x_coords, y_coords, pen=data_item['pen'], name=data_item['text'], movable=True)
                    # Connect the movement signal
                    plot_item.sigRegionChangeFinished.connect(lambda item, index=i: self.line_moved(item, index))
                    self.plot_widget.addItem(plot_item)
                    data_item['plot_item'] = plot_item
                elif data_item['type'] == 'points':
                    points = data_item['data']
                    plot_item = self.plot_widget.plot(points[:, 0], points[:, 1], pen=None, symbol='o', symbolPen=data_item['pen'], symbolBrush=data_item['pen'].color(), name=data_item['text'])
                    data_item['plot_item'] = plot_item
                elif data_item['type'] == 'polyline':
                    points = data_item['data']
                    plot_item = self.plot_widget.plot(points[:, 0], points[:, 1], pen=data_item['pen'], name=data_item['text'])
                    data_item['plot_item'] = plot_item
                elif data_item['type'] == 'infinite_line':
                    line = pg.InfiniteLine(pos=data_item['pos'], angle=data_item['angle'], pen=data_item['pen'], name=data_item['text'])
                    self.plot_widget.addItem(line)
                    data_item['plot_item'] = line
                elif data_item['type'] == 'equation_point':
                    points = data_item['data']
                    plot_item = self.plot_widget.plot(points[:, 0], points[:, 1], pen=None, symbol='x', symbolSize=10, symbolPen=data_item['pen'], symbolBrush=data_item['pen'].color(), name=data_item['text'])
                    data_item['plot_item'] = plot_item
                    # Connect the movement signal for equation points
                    # plot_item.sigRegionChangeFinished.connect(lambda item, index=i: self.equation_point_moved(item, index)) # Removed movable=True and signal connection

        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for range or resolution. Please enter numbers.")

    def line_moved(self, item, index):
        new_x, new_y = item.getData()
        self.plotted_data[index]['data'] = (new_x, new_y)
        
        # Update the text in the list widget
        new_text = f"({new_x[0]:.2f}, {new_y[0]:.2f}); ({new_x[1]:.2f}, {new_y[1]:.2f})"
        self.plotted_data[index]['text'] = new_text
        self.plotted_data[index]['list_item'].setText(new_text)

    def equation_point_moved(self, item, index):
        new_x, new_y = item.getData()
        # For now, just update the stored data.
        # In future, this is where the equation update logic or constraint logic would go.
        self.plotted_data[index]['data'] = np.array([[new_x[0], new_y[0]]])
        new_text = f"Eq. Point ({new_x[0]:.3f}, {new_y[0]:.3f})"
        self.plotted_data[index]['text'] = new_text
        self.plotted_data[index]['list_item'].setText(new_text)

    def handle_llm_question(self):
        question = self.llm_question_input.text()
        if not question:
            QMessageBox.warning(self, "Error", "Please enter a question.")
            return

        # 1. Gather Context
        equation_list = [eq['text'] for eq in self.plotted_equations]
        data_list = [d['text'] for d in self.plotted_data]
        context = {
            "equations": equation_list,
            "data": data_list,
            "view_range": {
                "x_min": self.x_min_input.text(),
                "x_max": self.x_max_input.text(),
                "y_min": self.y_min_input.text(),
                "y_max": self.y_max_input.text(),
            },
            "resolution": self.resolution_input.text(),
            "user_question": question
        }

        # 2. Construct the Prompt
        prompt = f"""You are a helpful mathematics assistant integrated into a graphing calculator application.
        Your purpose is to analyze the provided graph data and answer a user's question.
        
        Here is the current context from the calculator:
        - Plotted Equations: {context['equations']}
        - Plotted Data (Points/Lines): {context['data']}
        - Current Viewport: X-axis from {context['view_range']['x_min']} to {context['view_range']['x_max']}, Y-axis from {context['view_range']['y_min']} to {context['view_range']['y_max']}.
        - Plotting Resolution: {context['resolution']} points.
        
        User's Question: "{context['user_question']}"
        
        Based on all this information, please provide a concise and helpful answer to the user's question.
        Explain the mathematical concepts clearly. If the question is about a specific feature of a graph, relate it to the mathematical properties of the equation(s) or data.
        """

        # 3. Query Ollama
        try:
            payload = {
                "model": "gemma3:1b",
                "prompt": prompt,
                "stream": False
            }
            command = [
                'curl', '-s', 'http://localhost:11434/api/generate',
                '-d', json.dumps(payload)
            ]
            
            # Show a temporary message
            loading_msg = QMessageBox(self)
            loading_msg.setWindowTitle("Thinking...")
            loading_msg.setText("Asking Gemma... Please wait.")
            loading_msg.setStandardButtons(QMessageBox.StandardButton.NoButton)
            loading_msg.show()
            QApplication.processEvents() # Allow UI to update

            result = subprocess.run(command, capture_output=True, text=True, check=True)
            response_data = json.loads(result.stdout)
            llm_answer = response_data.get("response", "No answer received from model.")
            
            loading_msg.hide()
            QMessageBox.information(self, "Gemma's Answer", llm_answer)

        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "`curl` command not found. Please ensure it is installed and in your PATH.")
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"Error communicating with Ollama server: {e.stderr}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")

    def zoom_in(self):
        self._zoom(0.8)

    def zoom_out(self):
        self._zoom(1.25)

    def _zoom(self, factor):
        try:
            x_min = float(self.x_min_input.text())
            x_max = float(self.x_max_input.text())
            y_min = float(self.y_min_input.text())
            y_max = float(self.y_max_input.text())

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            new_x_range = (x_max - x_min) * factor
            new_y_range = (y_max - y_min) * factor

            self.x_min_input.setText(str(x_center - new_x_range / 2))
            self.x_max_input.setText(str(x_center + new_x_range / 2))
            self.y_min_input.setText(str(y_center - new_y_range / 2))
            self.y_max_input.setText(str(y_center + new_y_range / 2))

            self.redraw_plots()
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for range. Please enter numbers.")

    def pan_left(self):
        self._pan(-0.1, 0)

    def pan_right(self):
        self._pan(0.1, 0)

    def pan_up(self):
        self._pan(0, 0.1)

    def pan_down(self):
        self._pan(0, -0.1)

    def _pan(self, x_factor, y_factor):
        try:
            x_min = float(self.x_min_input.text())
            x_max = float(self.x_max_input.text())
            y_min = float(self.y_min_input.text())
            y_max = float(self.y_max_input.text())

            x_range = x_max - x_min
            y_range = y_max - y_min

            x_offset = x_range * x_factor
            y_offset = y_range * y_factor

            self.x_min_input.setText(str(x_min + x_offset))
            self.x_max_input.setText(str(x_max + x_offset))
            self.y_min_input.setText(str(y_min + y_offset))
            self.y_max_input.setText(str(y_max + y_offset))

            self.redraw_plots()
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for range. Please enter numbers.")

    def toggle_add_point_mode(self, checked):
        if checked:
            try:
                # Works for PyQt6 and PySide6
                self.setCursor(QtGui.QCursor(pg.QtCore.Qt.CursorShape.CrossCursor))
            except AttributeError:
                # Works for PyQt5 and PySide2
                self.setCursor(QtGui.QCursor(pg.QtCore.Qt.CrossCursor))
        else:
            try:
                # Works for PyQt6 and PySide6
                self.setCursor(QtGui.QCursor(pg.QtCore.Qt.CursorShape.ArrowCursor))
            except AttributeError:
                # Works for PyQt5 and PySide2
                self.setCursor(QtGui.QCursor(pg.QtCore.Qt.ArrowCursor))

    def mousePressEvent(self, event):
        try:
            # Works for PyQt6 and PySide6
            left_button = pg.QtCore.Qt.MouseButton.LeftButton
        except AttributeError:
            # Works for PyQt5 and PySide2
            left_button = pg.QtCore.Qt.LeftButton

        if self.add_point_mode_button.isChecked() and event.button() == left_button:
            # Get mouse position relative to PlotWidget
            local_pos = self.plot_widget.mapFromGlobal(event.globalPosition().toPoint())
            
            # Map local position to ViewBox coordinates
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(self.plot_widget.mapToScene(local_pos))
            x, y = round(mouse_point.x(), 3), round(mouse_point.y(), 3)
            
            # Add the new point to the data
            new_point_text = f"({x}, {y})"
            color = self.plot_colors[self.color_index % len(self.plot_colors)]
            pen = pg.mkPen(color=color, width=2)
            
            list_item = QListWidgetItem(new_point_text)
            self.data_list_widget.addItem(list_item)

            self.plotted_data.append({
                'type': 'points', 
                'data': np.array([[x, y]]), 
                'pen': pen, 
                'text': new_point_text,
                'list_item': list_item
            })
            self.color_index += 1
            self.redraw_plots()
        elif self.add_point_to_equation_button.isChecked() and event.button() == left_button:
            if self.selected_equation_for_points is None:
                QMessageBox.warning(self, "Error", "No equation selected for point placement.")
                return
            
            # Get mouse position relative to PlotWidget
            local_pos = self.plot_widget.mapFromGlobal(event.globalPosition().toPoint())
            
            # Map local position to ViewBox coordinates
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(self.plot_widget.mapToScene(local_pos))
            x_val = mouse_point.x()
            
            # Evaluate the selected equation at x_val
            try:
                x_sym = symbols('x')
                func = lambdify(x_sym, self.selected_equation_for_points['expr'], 'numpy')
                y_val = func(x_val)
                
                new_point_text = f"Eq. Point ({x_val:.3f}, {y_val:.3f})"
                color = self.selected_equation_for_points['pen'].color() # Use equation's color
                pen = pg.mkPen(color=color, width=2)
                
                list_item = QListWidgetItem(new_point_text)
                self.data_list_widget.addItem(list_item)

                self.plotted_data.append({
                    'type': 'equation_point', 
                    'data': np.array([[x_val, y_val]]), 
                    'pen': pen, 
                    'text': new_point_text,
                    'list_item': list_item
                })
                self.color_index += 1 # Still increment for color cycling for other plots
                self.redraw_plots()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not evaluate equation at this point: {e}")
        elif self.move_point_button.isChecked() and event.button() == left_button:
            # Get mouse position relative to PlotWidget
            local_pos = self.plot_widget.mapFromGlobal(event.globalPosition().toPoint())
            
            # Map local position to ViewBox coordinates
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(self.plot_widget.mapToScene(local_pos))
            # Check if an equation point is clicked
            for i, data_item in enumerate(self.plotted_data):
                if data_item['type'] == 'equation_point':
                    point_pos = data_item['data'][0]
                    # Simple distance check (adjust tolerance as needed)
                    distance = np.sqrt((mouse_point.x() - point_pos[0])**2 + (mouse_point.y() - point_pos[1])**2)
                    if distance < 0.5: # Tolerance for clicking a point
                        self.moving_point_index = i
                        self.moving_point_original_pos = point_pos
                        self.setCursor(QtGui.QCursor(pg.QtCore.Qt.CursorShape.ClosedHandCursor))
                        break
        super().mousePressEvent(event) # Call super class method

    def mouseMoveEvent(self, event):
        if self.moving_point_index != -1:
            # Get mouse position relative to PlotWidget
            local_pos = self.plot_widget.mapFromGlobal(event.globalPosition().toPoint())
            
            # Map local position to ViewBox coordinates
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(self.plot_widget.mapToScene(local_pos))
            x, y = mouse_point.x(), mouse_point.y()
            
            # Update the point's data
            self.plotted_data[self.moving_point_index]['data'] = np.array([[x, y]])
            new_text = f"Eq. Point ({x:.3f}, {y:.3f})"
            self.plotted_data[self.moving_point_index]['text'] = new_text
            self.plotted_data[self.moving_point_index]['list_item'].setText(new_text)
            
            self.redraw_plots()
        super().mouseMoveEvent(event) # Call super class method

    def mouseReleaseEvent(self, event):
        if self.moving_point_index != -1:
            self.moving_point_index = -1
            self.moving_point_original_pos = None
            self.setCursor(QtGui.QCursor(pg.QtCore.Qt.CursorShape.ArrowCursor))
        super().mouseReleaseEvent(event) # Call super class method

    def calculate_angle_between_lines(self):
        selected_items = self.data_list_widget.selectedItems()
        if len(selected_items) != 2:
            QMessageBox.warning(self, "Selection Error", "Please select exactly two lines from the 'Points/Lines' list.")
            return

        try:
            line1_text = selected_items[0].text()
            line2_text = selected_items[1].text()

            v1 = self.get_line_vector(line1_text)
            v2 = self.get_line_vector(line2_text)

            if v1 is None or v2 is None:
                QMessageBox.critical(self, "Error", "Could not parse one or both of the selected items as a line. Ensure they are in the format '(x1, y1); (x2, y2)'.")
                return

            unit_v1 = v1 / np.linalg.norm(v1)
            unit_v2 = v2 / np.linalg.norm(v2)
            dot_product = np.dot(unit_v1, unit_v2)
            angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            QMessageBox.information(self, "Angle Calculation", f"The angle between the two lines is:\n{angle_deg:.2f} degrees")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not calculate angle: {e}")

    def get_line_vector(self, line_text):
        line_match = re.match(r'\s*\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)\s*;\s*\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)\s*', line_text)
        if line_match:
            x1, y1, x2, y2 = map(float, line_match.groups())
            return np.array([x2 - x1, y2 - y1])
        return None

if __name__ == "__main__":
    # Force X11 backend to avoid Wayland crashes
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    app = QApplication(sys.argv)
    main_win = AdvancedCalculator()
    main_win.show()
    sys.exit(app.exec())