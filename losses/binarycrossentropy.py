import numpy as np


class BinaryCrossEntropy:
    def __init__(self) -> None:
        pass

    def compute(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the binary cross entropy loss.
            args:
                y: true labels (n_classes, batch_size)
                y_hat: predicted labels (n_classes, batch_size)
            returns:
                binary cross entropy loss
        """
        # TODO: Implement binary cross entropy loss
        y_pred = np.clip(y_hat, 1e-7, 1 - 1e-7)
        cost = -np.mean((1 - y) * np.log(1 - y_pred + 1e-7) + y * np.log(y_pred + 1e-7), axis=0)
        return np.squeeze(cost)

    def backward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the binary cross entropy loss.
            args:
                y: true labels (n_classes, batch_size)
                y_hat: predicted labels (n_classes, batch_size)
            returns:
                derivative of the binary cross entropy loss
        """
        # hint: use the np.divide function
        # TODO: Implement backward pass for binary cross entropy loss
        y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)
        return np.mean((-1 * y / (y_hat + 1e-7) + (1 - y) / (1 - y_hat + 1e-7)))
