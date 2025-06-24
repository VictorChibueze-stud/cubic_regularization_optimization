import logging
import numpy as np
from .base import BaseOptimizer

class BFGS(BaseOptimizer):
    """
    BFGS optimizer for regularized logistic regression.
    """
    def __init__(self, X, y, **kwargs):
        super().__init__(X, y, **kwargs)
        self.c = 1e-4
        self.rho = 0.8

    def line_search(self, d):
        alpha = 1.0
        f_init = self.f(self.w)
        grad_init = self.grad(self.w)
        for _ in range(20):
            try:
                if self.f(self.w + alpha * d) <= f_init + self.c * alpha * np.dot(grad_init, d):
                    return alpha
                alpha *= self.rho
            except Exception as e:
                logging.warning(f"BFGS line search error: {e}")
                break
        return alpha

    def optimize(self):
        n = len(self.w)
        I = np.eye(n)
        H = I
        for i in range(self.max_iter):
            try:
                g = self.grad(self.w)
                if np.linalg.norm(g) < self.tol:
                    logging.info(f"Converged after {i} iterations")
                    break
                d = -H.dot(g)
                alpha = self.line_search(d)
                s = alpha * d
                self.w += s
                y = self.grad(self.w) - g
                rho = 1.0 / (y.dot(s) + 1e-8)
                H_new = (I - rho * np.outer(s, y)).dot(H).dot(I - rho * np.outer(y, s)) + rho * np.outer(s, s)
                if np.all(np.linalg.eigvals(H_new) > 0):
                    H = H_new
                else:
                    logging.info("Resetting Hessian approximation")
                    H = I
                self.loss_history.append(self.f(self.w))
                self.accuracy_history.append(self.evaluate(self.X, self.y))
                if i % 10 == 0:
                    logging.info(f"Iteration {i}, Loss: {self.loss_history[-1]:.4f}, Accuracy: {self.accuracy_history[-1]:.4f}")
                if self.early_stopping():
                    logging.info(f"Early stopping at iteration {i}")
                    break
            except Exception as e:
                logging.error(f"Error in BFGS at iteration {i}: {e}")
                break
        return self.w 