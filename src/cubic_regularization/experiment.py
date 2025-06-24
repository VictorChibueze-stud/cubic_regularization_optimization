import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from .data import load_data, create_feature_matrix
from .optimizers.gradient_descent import GradientDescent
from .optimizers.conjugate_gradient import ConjugateGradientDescent
from .optimizers.bfgs import BFGS
from .optimizers.arc import ARC
from .optimizers.cubic_newton import CubicNewtonLogisticRegression
from .utils import plot_loss_histories, plot_accuracy_histories

def run_experiment(train_file='a9a', test_file='a9a.t', n_runs=5, max_iter=2000, tol=1e-6, patience=10):
    try:
        train_data, train_labels = load_data(train_file)
        test_data, test_labels = load_data(test_file)
        max_feature_train = max(max(d.keys()) for d in train_data)
        max_feature_test = max(max(d.keys()) for d in test_data)
        n_features = max(max_feature_train, max_feature_test)
        X_train = create_feature_matrix(train_data, n_features)
        X_test = create_feature_matrix(test_data, n_features)
        y_train = ((train_labels + 1) // 2).astype(int)
        y_test = ((test_labels + 1) // 2).astype(int)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        algorithms = [
            ('ARC', lambda: ARC(X_train, y_train, max_iter=max_iter, tol=tol, lanczos_k=20)),
            ('Gradient Descent', lambda: GradientDescent(X_train, y_train, learning_rate=0.01, max_iter=max_iter, tol=tol)),
            ('Conjugate Gradient Descent', lambda: ConjugateGradientDescent(X_train, y_train, max_iter=max_iter, tol=tol)),
            ('BFGS', lambda: BFGS(X_train, y_train, max_iter=max_iter, tol=tol)),
            ('Cubic Newton', lambda: CubicNewtonLogisticRegression(X_train, y_train, max_iter=max_iter, tol=tol))
        ]
        all_loss_histories = {algo_name: [] for algo_name, _ in algorithms}
        all_accuracy_histories = {algo_name: [] for algo_name, _ in algorithms}
        for run in range(n_runs):
            logging.info(f"Run {run+1}/{n_runs}")
            np.random.seed(run)
            w0 = np.random.randn(X_train.shape[1])
            for algo_name, algo_constructor in algorithms:
                logging.info(f"Running {algo_name}")
                optimizer = algo_constructor()
                optimizer.w = w0.copy()
                optimizer.optimize()
                all_loss_histories[algo_name].append(optimizer.loss_history)
                all_accuracy_histories[algo_name].append(optimizer.accuracy_history)
                train_acc = optimizer.evaluate(X_train, y_train)
                test_acc = optimizer.evaluate(X_test, y_test)
                logging.info(f"{algo_name} Results: Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}, Final Loss: {optimizer.loss_history[-1]:.4e}, Iterations: {len(optimizer.loss_history)}")
        avg_loss_histories = {}
        avg_accuracy_histories = {}
        for algo_name in all_loss_histories:
            max_len = max(len(hist) for hist in all_loss_histories[algo_name])
            loss_matrix = np.array([np.pad(hist, (0, max_len - len(hist)), 'edge') for hist in all_loss_histories[algo_name]])
            avg_loss_histories[algo_name] = np.mean(loss_matrix, axis=0)
            max_len = max(len(hist) for hist in all_accuracy_histories[algo_name])
            acc_matrix = np.array([np.pad(hist, (0, max_len - len(hist)), 'edge') for hist in all_accuracy_histories[algo_name]])
            avg_accuracy_histories[algo_name] = np.mean(acc_matrix, axis=0)
        plot_loss_histories(avg_loss_histories)
        plot_accuracy_histories(avg_accuracy_histories)
        logging.info("Experiment completed and plots saved.")
    except Exception as e:
        logging.error(f"Experiment failed: {e}")
        raise 