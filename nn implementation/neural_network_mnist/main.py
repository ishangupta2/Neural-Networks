# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.a0 = 1.0 / math.sqrt(d)
        self.a1  = 1.0 / math.sqrt(h)
        self.w0 = Parameter(-self.a0 + (2 * self.a0) * torch.rand(d, h))
        self.w1 = Parameter(-self.a1 + (2 * self.a1) * torch.rand(h, k))
        self.b0 = Parameter(-self.a0 + (2 * self.a0) * torch.rand(h))
        self.b1 = Parameter(-self.a1 + (2 * self.a1) * torch.rand(k))


    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        n = x.shape[0]
        activation1 = relu(x @ self.w0 + self.b0.repeat(n, 1))
        output = activation1 @ self.w1 + self.b1.repeat(n, 1)
        return output


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.a0 = 1.0 / math.sqrt(d)
        self.a1 = 1.0 / math.sqrt(h0)
        self.a2 = 1.0 / math.sqrt(h1)
        self.w0 = Parameter(-self.a0 + (2 * self.a0) * torch.rand(d, h0))
        self.w1 = Parameter(-self.a1 + (2 * self.a1) * torch.rand(h0, h1))
        self.w2 = Parameter(-self.a2 + (2 * self.a2) * torch.rand(h1, k))
        self.b0 = Parameter(-self.a0 + (2 * self.a0) * torch.rand(h0))
        self.b1 = Parameter(-self.a1 + (2 * self.a1) * torch.rand(h1))
        self.b2 = Parameter(-self.a2 + (2 * self.a2) * torch.rand(k))


    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        activation1 = relu(x @ self.w0 + self.b0.repeat(x.shape[0], 1))
        activation2 = relu(activation1 @ self.w1 + self.b1.repeat(activation1.shape[0], 1))
        return activation2 @ self.w2 + self.b2.repeat(activation2.shape[0], 1)


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    losses = []
    accuracy = 0
    while accuracy < 0.99:
        total_loss = 0
        accuracy = 0
        # on each epoch
        for x, y in train_loader:
            optimizer.zero_grad()
            predictions = model.forward(x)
            loss = cross_entropy(predictions, y)
            loss.backward()
            optimizer.step()
            prediction = torch.argmax(predictions, axis=1)
            accuracy += torch.sum(prediction == y) / len(prediction)
            total_loss += loss.item()
        accuracy /= len(train_loader)
        losses.append(total_loss / len(train_loader))
        print(accuracy)

    return losses


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    # A6 part a
    model_f1 = F1(64, 784, 10)
    optimizer_f1 = Adam(model_f1.parameters(), lr=0.0005)
    loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
    losses_f1 = train(model_f1, optimizer_f1, loader)

    # Test Accuracy and Loss
    predictions_f1 = model_f1.forward(x_test)
    test_loss_f1 = cross_entropy(predictions_f1, y_test).item()
    accuracy_f1 = torch.sum(y_test == torch.argmax(predictions_f1, 1)) / len(predictions_f1)

    print(f"F1 Accuracy: {accuracy_f1}")
    print(f"F1 Loss: {test_loss_f1}")

    plt.figure(figsize=(10, 5))
    plt.plot(torch.arange(len(losses_f1)), losses_f1)
    plt.title("F1 Training Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # A6 part b
    model_f2 = F2(32, 32, 784, 10)
    optimizer_f2 = Adam(model_f2.parameters(), lr=0.0005)
    losses_f2 = train(model_f2, optimizer_f2, loader)

    # Test Accuracy and Loss
    predictions_f2 = model_f2.forward(x_test)
    test_loss_f2 = cross_entropy(predictions_f2, y_test).item()
    accuracy_f2 = torch.sum(y_test == torch.argmax(predictions_f2, 1)) / len(predictions_f2)

    print(f"F2 Accuracy: {accuracy_f2}")
    print(f"F2 Loss: {test_loss_f2}")

    plt.figure(figsize=(10, 5))
    plt.plot(torch.arange(len(losses_f2)), losses_f2)
    plt.title("F2 Training Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # A6 part c
    params_f1 = sum(p.numel() for p in model_f1.parameters() if p.requires_grad)
    params_f2 = sum(p.numel() for p in model_f2.parameters() if p.requires_grad)

    print(f"F1 Params: {params_f1}")
    print(f"F2 Params: {params_f2}")

if __name__ == "__main__":
    main()
