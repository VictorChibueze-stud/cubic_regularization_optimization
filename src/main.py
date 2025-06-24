import sys
from cubic_regularization.logging_config import setup_logging
from cubic_regularization.experiment import run_experiment

def main():
    setup_logging()
    try:
        run_experiment()
    except Exception as e:
        print(f"Experiment failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 