from scripts.digits.train_digits import train_digit_classification
import torch

if __name__ == "__main__":

    device = torch.device("cuda")

    seed = 1

    train_digit_classification(
        digits_to_include=[0,1],
        seed=seed,
        device=device,
        train_with_adjoint=True,
        train_with_noise=False,
        batch_size=64,
        nr_epochs=100,
        lr=5e-2,
        lambda_suppression=1e-1,
        lambda_magnitude=1e-2,
        lambda_ei=1e+0)
