import logging
import numpy as np
from scipy.optimize import line_search
from .base import BaseOptimizer

class ConjugateGradientDescent(BaseOptimizer):
    def __init__(self, X, y, max_iter=1000, tol=1e-6, reset_interval=50, **kwargs):
        super().__init__(X, y, max_iter=max_iter, tol=tol, **kwargs)
        self.reset_interval = reset_interval

    def optimize(self):
        r = -self.grad(self.w)
        p = r.copy()
        for i in range(self.max_iter):
            try:
                grad_old = self.grad(self.w)
                alpha, _, _, _, _, _ = line_search(self.f, self.grad, self.w, p)
                if alpha is None:
                    alpha = 1e-4
                self.w += alpha * p
                grad_new = self.grad(self.w)
                r_new = -grad_new
                beta = max(0, np.dot(r_new, grad_new - grad_old) / (np.dot(r, r) + 1e-12))
                p = r_new + beta * p
                r = r_new
                self.loss_history.append(self.f(self.w))
                self.accuracy_history.append(self.evaluate(self.X, self.y))
                if np.linalg.norm(r) < self.tol:
                    logging.info(f"Converged after {i+1} iterations")
                    break
                if self.early_stopping():
                    logging.info(f"Early stopping at iteration {i+1}")
                    break
                if (i + 1) % self.reset_interval == 0:
                    p = r.copy()
            except Exception as e:
                logging.error(f"Error in ConjugateGradientDescent at iteration {i}: {e}")
                break
        return self.w 