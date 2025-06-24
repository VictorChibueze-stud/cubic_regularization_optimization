import logging
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import LinearOperator
from .base import BaseOptimizer

class ARC(BaseOptimizer):
    """
    Adaptive Regularization with Cubics (ARC) optimizer for regularized logistic regression.
    """
    def __init__(self, X, y, gamma2=2, gamma1=0.5, eta2=0.75, eta1=0.1, sigma0=1, learning_rate=0.01, lanczos_k=10, **kwargs):
        super().__init__(X, y, **kwargs)
        self.gamma2 = gamma2
        self.gamma1 = gamma1
        self.eta2 = eta2
        self.eta1 = eta1
        self.sigma = sigma0
        self.B = np.eye(X.shape[1])
        self.learning_rate = learning_rate
        self.lanczos_k = lanczos_k

    def line_search(self, s, c1=1e-4, alpha_max=1.0):
        alpha = alpha_max
        phi_0 = self.f(self.w)
        dphi_0 = np.dot(self.grad(self.w), s)
        for _ in range(10):
            try:
                if self.f(self.w + alpha * s) <= phi_0 + c1 * alpha * dphi_0:
                    return alpha
                alpha *= 0.5
            except Exception as e:
                logging.warning(f"ARC line search error: {e}")
                break
        return alpha

    def lanczos(self, A, v):
        n = len(v)
        V = np.zeros((n, self.lanczos_k))
        T = np.zeros((self.lanczos_k, self.lanczos_k))
        beta = np.linalg.norm(v)
        V[:, 0] = v / beta
        for j in range(self.lanczos_k):
            w = A @ V[:, j]
            alpha = np.dot(w, V[:, j])
            T[j, j] = alpha
            if j < self.lanczos_k - 1:
                w = w - alpha * V[:, j]
                if j > 0:
                    w = w - beta * V[:, j-1]
                beta = np.linalg.norm(w)
                if beta < 1e-8:
                    return V[:, :j+1], T[:j+1, :j+1]
                V[:, j+1] = w / beta
                T[j, j+1] = beta
                T[j+1, j] = beta
        return V, T

    def compute_step(self):
        g = self.grad(self.w)
        n = len(g)
        def hessian_vector_product(v):
            return self.B @ v
        A = LinearOperator(shape=(n, n), matvec=hessian_vector_product, dtype=np.float64)
        V, T = self.lanczos(A, g)
        eigenvalues, eigenvectors = eigh_tridiagonal(np.diag(T), np.diag(T, k=1))
        eigenvectors = V @ eigenvectors
        lambda_ = max(0, -np.min(eigenvalues) + np.sqrt(self.sigma * np.linalg.norm(g)))
        try:
            s = -np.linalg.solve(self.B + lambda_ * np.eye(n), g)
        except Exception as e:
            logging.error(f"ARC step solve error: {e}")
            s = -g
        return s

    def update_hessian_approx(self, s, y):
        sy = s.dot(y)
        if sy <= 0:
            return self.B
        rho = 1.0 / sy
        Bs = self.B.dot(s)
        B_new = self.B + np.outer(y, y) / sy - np.outer(Bs, Bs) / (s.dot(Bs) + 1e-12)
        return B_new

    def optimize(self):
        for i in range(self.max_iter):
            try:
                g = self.grad(self.w)
                s = self.compute_step()
                if np.linalg.norm(s) < self.tol:
                    logging.info(f"Step size too small at iteration {i+1}.")
                    break
                alpha = self.line_search(s)
                w_new = self.w + alpha * s
                g_new = self.grad(w_new)
                y = g_new - g
                self.B = self.update_hessian_approx(alpha * s, y)
                self.w = w_new
                loss = self.f(self.w)
                accuracy = self.evaluate(self.X, self.y)
                self.loss_history.append(loss)
                self.accuracy_history.append(accuracy)
                if i % 10 == 0:
                    logging.info(f"Iteration {i+1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
                if np.linalg.norm(g_new) < self.tol:
                    logging.info(f"Converged in {i+1} iterations with gradient norm {np.linalg.norm(g_new):.4e}")
                    break
                if i > 50 and self.early_stopping(patience=10):
                    logging.info(f"Early stopping at iteration {i+1}")
                    break
            except Exception as e:
                logging.error(f"Error in ARC at iteration {i}: {e}")
                break
        return self.w 