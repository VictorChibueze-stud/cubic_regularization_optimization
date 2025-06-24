import matplotlib.pyplot as plt
import logging
import numpy as np
from typing import Dict, List

def plot_loss_histories(avg_loss_histories: Dict[str, np.ndarray], filename: str = 'average_loss_comparison.png'):
    try:
        plt.figure(figsize=(12, 8))
        for algo_name, avg_loss_history in avg_loss_histories.items():
            plt.plot(avg_loss_history, label=algo_name)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Average Loss Over Iterations')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        logging.info(f"Saved loss plot to {filename}")
    except Exception as e:
        logging.error(f"Error plotting loss histories: {e}")

def plot_accuracy_histories(avg_accuracy_histories: Dict[str, np.ndarray], filename: str = 'average_accuracy_comparison.png'):
    try:
        plt.figure(figsize=(12, 8))
        for algo_name, avg_accuracy_history in avg_accuracy_histories.items():
            plt.plot(avg_accuracy_history, label=algo_name)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Average Accuracy Over Iterations')
        plt.xscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        logging.info(f"Saved accuracy plot to {filename}")
    except Exception as e:
        logging.error(f"Error plotting accuracy histories: {e}") 