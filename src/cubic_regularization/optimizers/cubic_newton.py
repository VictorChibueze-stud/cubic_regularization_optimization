import logging
import numpy as np
from scipy.linalg import solve
from .base import BaseOptimizer

class CubicNewtonLogisticRegression(BaseOptimizer):
    """
    Cubic Newton optimizer for regularized logistic regression.
    """
    def __init__(self, X, y, sigma0=0.1, **kwargs):
        super().__init__(X, y, **kwargs)
        self.sigma = sigma0

    def grad(self, w):
        z = self.X @ w
        grad_loss = self.X.T @ (self.sigmoid(z) - self.y) / len(self.y)
        grad_reg = self.lambda_param * w
        return grad_loss + grad_reg

    def hessian(self, w):
        z = self.X @ w
        S = self.sigmoid(z) * (1 - self.sigmoid(z))
        H = (self.X.T * S) @ self.X / len(self.y)
        H += self.lambda_param * np.eye(w.shape[0])
        return H

    def solve_cubic_subproblem(self, H, grad, sigma):
        eig_vals, _ = np.linalg.eigh(H)
        min_eig = np.min(eig_vals)
        if min_eig > 0:
            lambda_star = 0
        else:
            lambda_star = -min_eig + np.cbrt(sigma/2 * np.linalg.norm(grad))
        try:
            step = -solve(H + lambda_star * np.eye(H.shape[0]), grad)
        except Exception as e:
            logging.error(f"Cubic Newton subproblem solve error: {e}")
            step = -grad
        return step

    def line_search(self, w, step):
        alpha = 1.0
        c = 1e-4
        rho = 0.5
        max_ls_iter = 20
        f_init = self.f(w)
        grad_init = self.grad(w)
        for _ in range(max_ls_iter):
            try:
                if self.f(w + alpha * step) <= f_init + c * alpha * np.dot(grad_init, step):
                    return alpha
                alpha *= rho
            except Exception as e:
                logging.warning(f"Cubic Newton line search error: {e}")
                break
        return alpha

    def optimize(self):
        for i in range(self.max_iter):
            try:
                grad = self.grad(self.w)
                H = self.hessian(self.w)
                step = self.solve_cubic_subproblem(H, grad, self.sigma)
                alpha = self.line_search(self.w, step)
                self.w += alpha * step
                loss = self.f(self.w)
                self.loss_history.append(loss)
                accuracy = self.evaluate(self.X, self.y)
                self.accuracy_history.append(accuracy)
                if i % 10 == 0:
                    logging.info(f"Iteration {i}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
                if np.linalg.norm(grad) < self.tol:
                    logging.info(f"Converged in {i+1} iterations with gradient norm {np.linalg.norm(grad):.4e}")
                    break
                if i > 50 and self.early_stopping(patience=20):
                    logging.info(f"Early stopping at iteration {i}")
                    break
                self.sigma = max(self.sigma / 2, 1e-8)
            except Exception as e:
                logging.error(f"Error in Cubic Newton at iteration {i}: {e}")
                break
        return self.w 