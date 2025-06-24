# Cubic Regularization: Modular Python Implementation

This project implements advanced optimization algorithms for regularized logistic regression, including Gradient Descent, Conjugate Gradient, BFGS, ARC, and Cubic Newton, in a modular, production-grade Python package.

## Features
- Modular, extensible codebase
- Robust error handling and logging
- Statistical analysis and plotting utilities
- Ready for research and enterprise use

## Project Structure
```
cubic_regularization/
│
├── src/
│   ├── cubic_regularization/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── optimizers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── gradient_descent.py
│   │   │   ├── conjugate_gradient.py
│   │   │   ├── bfgs.py
│   │   │   ├── arc.py
│   │   │   └── cubic_newton.py
│   │   ├── experiment.py
│   │   ├── utils.py
│   │   └── logging_config.py
│   └── main.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_optimizers.py
│   └── test_experiment.py
│
├── requirements.txt
├── .gitignore
├── README.md
└── setup.py
```

## Setup
1. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage
- Run experiments:
  ```sh
  python -m src.main
  ```
- Add your data files (e.g., `a9a`, `a9a.t`) to the project root.

## Testing
- Run tests with pytest or unittest:
  ```sh
  pytest tests/
  ```

## Running from the Project Root

If you want to run the experiment from the project root (not inside src/), set the PYTHONPATH so Python can find the package:

On Windows PowerShell:
```powershell
$env:PYTHONPATH = ".\src"; python -m src.main
```
On Linux/macOS:
```sh
PYTHONPATH=./src python -m src.main
```

## License
MIT 