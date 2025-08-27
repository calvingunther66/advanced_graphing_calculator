# Advanced Graphing Calculator

This is a high-performance, on-device graphing calculator built with Python. It is designed for precision and control, allowing for the graphing and computation of advanced equations with higher resolution and more granular control than many alternatives.

A key feature of this application is its integration with a local Large Language Model (LLM) via Ollama, enabling users to ask questions and receive AI-powered analysis of the plotted graphs.

## Features

- **Advanced Equation Plotting**: Plots multiple, complex mathematical functions simultaneously.
- **Flexible Parser**: The equation parser is designed for ease of use and understands common calculator-style syntax:
  - Handles function definitions like `f(x) = ...` or `y = ...`.
  - Accepts `^` for exponentiation (e.g., `x^2`) and automatically converts it.
  - Understands implicit multiplication (e.g., `4x` is treated as `4*x`).
- **Interactive Graph**: The plot view is fully interactive, supporting real-time panning and zooming.
- **Granular Control**: A dedicated control panel allows for precise adjustment of:
  - X-Axis and Y-Axis ranges.
  - Plotting resolution (the number of points used to draw a curve).
- **AI-Powered Analysis**: Ask questions in natural language about the plotted equations and receive insights from a locally running LLM (Ollama/gemma3:1b).

## Technology Stack

- **Backend**: Python
- **UI Framework**: PyQt6
- **Graphing Engine**: PyQtGraph
- **Numerical Computing**: NumPy
- **Symbolic Mathematics & Parsing**: SymPy
- **AI Integration**: Ollama

## Requirements

1.  **Python 3.x**
2.  **Ollama**: You must have Ollama installed and running.
3.  **Gemma Model**: The `gemma3:1b` model must be pulled and available in Ollama. You can get it by running:
    ```sh
    ollama pull gemma3:1b
    ```

## Installation & Setup

1.  **Clone the repository** (or ensure all files are in a single directory).

2.  **Create and activate a Python virtual environment**:
    ```sh
    # Navigate to the project directory
    cd /path/to/advanced_graphing_calculator

    # Create the venv
    python3 -m venv venv

    # Activate it (on Linux/macOS)
    source venv/bin/activate
    ```

3.  **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1.  Ensure your Ollama server is running in the background.

2.  Run the application from the project directory:
    ```sh
    python3 main.py
    ```

-   Enter an equation in the input box and press `Enter` or click "Plot Equation".
-   Use the control panel to adjust the view and resolution, then click "Redraw".
-   Type a question into the "Ask Gemma" input box and click the button to get AI analysis.
