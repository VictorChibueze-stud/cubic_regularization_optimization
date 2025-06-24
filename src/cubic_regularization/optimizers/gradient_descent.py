import logging
import numpy as np
from .base import BaseOptimizer

class GradientDescent(BaseOptimizer):
    def __init__(self, X, y, learning_rate=0.1, **kwargs):
        super().__init__(X, y, **kwargs)
        self.learning_rate = learning_rate
        self.adaptive_lr = True

    def optimize(self):
        for i in range(self.max_iter):
            try:
                g = self.grad(self.w)
                if np.linalg.norm(g) < self.tol:
                    logging.info(f"Converged after {i} iterations")
                    break
                if self.adaptive_lr:
                    self.learning_rate = self.line_search(g)
                self.w -= self.learning_rate * g
                self.loss_history.append(self.f(self.w))
                self.accuracy_history.append(self.evaluate(self.X, self.y))
                if i > 50 and self.early_stopping(patience=10):
                    logging.info(f"Early stopping at iteration {i}")
                    break
            except Exception as e:
                logging.error(f"Error in GradientDescent at iteration {i}: {e}")
                break
        return self.w

    def line_search(self, gradient, c=0.5, tau=0.5):
        alpha = 1.0
        w_new = self.w - alpha * gradient
        try:
            while self.f(w_new) > self.f(self.w) - c * alpha * np.dot(gradient, gradient):
                alpha *= tau
                w_new = self.w - alpha * gradient
        except Exception as e:
            logging.warning(f"Line search failed: {e}")
        return alpha 