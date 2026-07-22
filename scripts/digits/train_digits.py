import torch
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader

from src.brain_network import BrainNetwork
from src.utils.paths import config_path, models_path
from src.utils.set_seed import set_seed
from src.utils.loss_functions import compute_suppression_penalty, compute_L2_regularization, compute_ei_ratio_penalty




def prepare_ds(digits_to_include, padding, seed):
    '''
    Prepare the sklearn digit dataset by padding the images, flattening
    images to vectors and splitting the data into train and test sets.
    '''
    # Load dataset
    digits = datasets.load_digits()
    X = digits.images  # shape: (n_samples, 8, 8)
    y = digits.target

    # Pad the images
    if padding > 0:
        X = np.pad(X, ((0,0), (padding,padding), (padding,padding)))

    # Only data instances with a label in digits_to_include
    mask = np.isin(y, digits_to_include)
    X = X[mask]
    y = y[mask]

    # Remap labels to consecutive class indices
    label_map = {digit: idx for idx, digit in enumerate(sorted(digits_to_include))}
    y = np.array([label_map[digit] for digit in y], dtype=np.int64)

    # Flatten the images
    n_samples = len(X)
    X = X.reshape((n_samples, -1))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, shuffle=True, random_state=seed)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test

def train_digit_classification(
        digits_to_include,
        seed,
        device,
        train_with_adjoint=False,
        train_with_noise=False,
        batch_size=64,
        nr_epochs=50,
        lr=5e-2,
        lambda_suppression=1e-1,
        lambda_magnitude=1e-2,
        lambda_ei=1e+0):
    """
    Train a BrainNetwork to classify handwritten digits.
    """
    set_seed(seed)

    # Initialize network
    config = config_path('digits_params.toml')
    network = BrainNetwork.from_toml(config)

    network.add_area('v1', 128)
    network.add_area('v2', len(digits_to_include))

    network.add_input_connection('v1', 100, receptive_field_size=3, stride=1,
                                 grid_organization=True, std=1.0, scale=0.2)
    network.add_feedforward_connection('v1', 'v2', std=1.0, scale=0.7)
    network.add_output_connection('v2')

    # Extra lateral inhibition connections
    network.add_lateral_connection('v1', receptive_field_size=2, stride=2)
    network.add_lateral_connection('v2')


    # Get train and test set
    X_train, X_test, y_train, y_test = prepare_ds(digits_to_include, padding=1, seed=seed)

    # DataLoader for train set
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)


    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)


    def run_digits_batch(stims, labels):
        """
        Runs a stimulus batch through the network and computes the loss
        between the model predictions and the labels.
        """
        output = network.run(
            stims,
            adjoint=train_with_adjoint,
            stochastic=train_with_noise,
            device=device)
        model_predictions = network.read_out(output, mode='classification')

        # print(model_predictions)
        # network.analysis.plot_firing_rates(output, area='v2', population='L23e')

        labels = labels.to(device)
        _, mapped_labels = torch.unique(labels, return_inverse=True)
        ce_loss = criterion(model_predictions, mapped_labels)

        suppression_penalty = compute_suppression_penalty(model_predictions, mapped_labels, len(digits_to_include))
        weight_penalty = compute_L2_regularization(network)
        ei_penalty = compute_ei_ratio_penalty(network)

        loss = ce_loss # + (lambda_suppression * suppression_penalty) + (lambda_magnitude * weight_penalty) + (lambda_ei * ei_penalty)
        acc = (labels == torch.argmax(model_predictions, dim=1)).float().mean()

        return loss, ce_loss, (lambda_suppression * suppression_penalty), (lambda_magnitude * weight_penalty), (lambda_ei * ei_penalty), acc, model_predictions


    # Store history during training
    history = {'train_losses': [],
               'test_losses': [],
               'suppression': [],
               'L2_reg': [],
               'ei_ratio': [],
               'accuracy': []}


    # Training loop
    for epoch in range(0, nr_epochs):
        print('Epoch {}'.format(epoch))

        for train_stims, train_labels in train_loader:
            start = time.time()
            optimizer.zero_grad()

            loss, ce_loss, suppression, magnitude, ei, acc, preds = run_digits_batch(train_stims.to(device), train_labels)

            loss.backward()
            optimizer.step()

            print('Train loss | {:.5f} | {:.1f}s'.format(loss.item(), time.time() - start))
            history['train_losses'].append(loss.item())

        # Evaluate with test set, after every epoch
        with torch.no_grad():

            test_loss, test_ce_loss, test_suppression, test_magnitude, test_ei, test_acc, test_preds = run_digits_batch(X_test, y_test)

            print('Test loss | {:.5f}'.format(test_loss.item()))
            print('Suppression {:.5f}'.format(test_suppression.item()))
            print('L2 regularization {:.5f}'.format(test_magnitude.item()))
            print('E/I ratio {:.5f}'.format(test_ei.item()))
            print('Test accuracy {:.2f}'.format(test_acc))

            print(torch.concat((test_preds, y_test.to(device).unsqueeze(1)), dim=-1))

            history['test_losses'].append(test_loss.item())
            history['suppression'].append(test_suppression.item())
            history['L2_reg'].append(test_magnitude.item())
            history['ei_ratio'].append(test_ei.item())
            history['accuracy'].append(test_acc.item())

        # Store training history and trained network
        torch.save(history, models_path('digits', f'digits_history_{seed}.pt'))
        network.save(models_path('digits', f'digits_{seed}.pt'))



if __name__ == '__main__':

    digits_to_include = [0,1,2,3,4,5,6,7,8,9]
    seed = 1
    device = torch.device('mps')

    train_digit_classification(
        digits_to_include=digits_to_include,
        seed=seed,
        device=device,
        train_with_adjoint=False,
        train_with_noise=False,
        batch_size=64,
        nr_epochs=100,
        lr=5e-2,
        lambda_suppression=1e-1,
        lambda_magnitude=1e-2,
        lambda_ei=1e+0)
