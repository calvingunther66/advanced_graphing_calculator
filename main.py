
import sys
import os
import numpy as np
import json
import subprocess
import re
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QListWidget, QMessageBox, QGridLayout
)
import pyqtgraph as pg
from sympy import sympify, symbols
from sympy.utilities.lambdify import lambdify

class AdvancedCalculator(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Advanced Graphing Calculator")
        self.setGeometry(100, 100, 1200, 800)
        
        self.plotted_equations = []
        self.plot_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        self.color_index = 0

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
        self.equation_list_widget = QListWidget()

        self.add_button.clicked.connect(self.plot_equation)
        self.equation_input.returnPressed.connect(self.plot_equation)

        left_layout.addWidget(QLabel("Equations:"))
        left_layout.addWidget(self.equation_input)
        left_layout.addWidget(self.add_button)
        left_layout.addWidget(self.equation_list_widget)

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
        control_panel.setFixedHeight(80)

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

            x_vals = np.linspace(x_min, x_max, resolution)
            x = symbols('x')

            for eq_data in self.plotted_equations:
                func = lambdify(x, eq_data['expr'], 'numpy')
                y_vals = func(x_vals)
                y_vals[np.isinf(y_vals)] = np.nan
                self.plot_widget.plot(x_vals, y_vals, pen=eq_data['pen'], name=eq_data['text'])

        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for range or resolution. Please enter numbers.")

    def handle_llm_question(self):
        question = self.llm_question_input.text()
        if not question:
            QMessageBox.warning(self, "Error", "Please enter a question.")
            return

        # 1. Gather Context
        equation_list = [eq['text'] for eq in self.plotted_equations]
        context = {
            "equations": equation_list,
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
        - Current Viewport: X-axis from {context['view_range']['x_min']} to {context['view_range']['x_max']}, Y-axis from {context['view_range']['y_min']} to {context['view_range']['y_max']}.
        - Plotting Resolution: {context['resolution']} points.
        
        User's Question: "{context['user_question']}"
        
        Based on all this information, please provide a concise and helpful answer to the user's question.
        Explain the mathematical concepts clearly. If the question is about a specific feature of a graph, relate it to the mathematical properties of the equation(s).
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

if __name__ == "__main__":
    # Force X11 backend to avoid Wayland crashes
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    app = QApplication(sys.argv)
    main_win = AdvancedCalculator()
    main_win.show()
    sys.exit(app.exec())
