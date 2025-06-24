import numpy as np
import logging
from typing import Any

class BaseOptimizer:
    """
    Base class for optimizers for regularized logistic regression.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, lambda_param: float = 0.001, alpha: float = 1, max_iter: int = 1000, tol: float = 1e-6):
        self.X = X
        self.y = y
        self.lambda_param = lambda_param
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.w = np.zeros(X.shape[1])
        self.loss_history = []
        self.accuracy_history = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        try:
            return 1 / (1 + np.exp(-z))
        except Exception as e:
            logging.error(f"Error in sigmoid computation: {e}")
            raise

    def f(self, w: np.ndarray) -> float:
        try:
            z = self.X.dot(w)
            log_loss = -np.mean(self.y * np.log(self.sigmoid(z) + 1e-15) + (1 - self.y) * np.log(1 - self.sigmoid(z) + 1e-15))
            reg = 0.5 * self.lambda_param * np.sum(w**2)
            return log_loss + reg
        except Exception as e:
            logging.error(f"Error in loss computation: {e}")
            raise

    def grad(self, w: np.ndarray) -> np.ndarray:
        try:
            z = self.X.dot(w)
            grad_loss = self.X.T.dot(self.sigmoid(z) - self.y) / len(self.y)
            grad_reg = self.lambda_param * w
            return grad_loss + grad_reg
        except Exception as e:
            logging.error(f"Error in gradient computation: {e}")
            raise

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        try:
            z = X.dot(self.w)
            y_pred = (self.sigmoid(z) >= 0.5).astype(int)
            accuracy = np.mean(y_pred == y)
            return accuracy
        except Exception as e:
            logging.error(f"Error in evaluation: {e}")
            raise

    def early_stopping(self, patience: int = 10, min_delta: float = 1e-4) -> bool:
        if len(self.loss_history) > patience:
            recent_losses = self.loss_history[-patience:]
            if all(abs(recent_losses[i] - recent_losses[i+1]) < min_delta for i in range(len(recent_losses)-1)):
                return True
        return False 