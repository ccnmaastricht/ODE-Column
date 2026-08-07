from scripts.digits.train_digits import train_digit_classification
import torch

if __name__ == "__main__":

    device = torch.device("cuda")

    for seed in range(1, 11, 1):

        train_digit_classification(
            digits_to_include=[0,1,2,3,4,5,6,7,8,9],
            seed=seed,
            device=device,
            train_with_adjoint=False,
            train_with_noise=False,
            batch_size=64,
            nr_epochs=100,
            lr=1e-2,
            lambda_volatility=1e+0)
