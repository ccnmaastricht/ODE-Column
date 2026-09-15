import torch
import time

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split


from src.brain_network import BrainNetwork
from src.utils.paths import config_path, models_path
from src.utils.set_seed import set_seed
from src.utils.loss_functions import compute_fr_volatility_penalty, compute_fr_ceiling_penalty



def train_mnist(
        seed,
        device,
        batch_size=64,
        lr=1e-2,
        lambda_volatility=1e+0,
        lambda_ceiling=1e-6,
        test_freq=30):
    """
    Training a network to classify the MNIST dataset of handwritten digits (28x28)
    """
    set_seed(seed)

    # Initialize network
    config = config_path('digits_params.toml')
    network = BrainNetwork.from_toml(config)

    network.add_area('v1', 169)
    network.add_area('v2', 10)

    network.add_input_connection('v1', 784, receptive_field_size=4, stride=2,
                                 grid_organization=True, std=1.0, scale=0.2)
    network.add_feedforward_connection('v1', 'v2', std=1.0, scale=0.3)
    network.add_lateral_connection('v2')
    network.add_output_connection('v2')


    # Prepare train set and validation set
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.flatten() * 10)])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    labels = train_dataset.targets

    generator = torch.Generator().manual_seed(seed)
    val_indices = []
    for class_idx in range(10):
        class_indices = torch.where(labels == class_idx)[0]
        perm = torch.randperm(len(class_indices), generator=generator)
        val_indices.extend(class_indices[perm[:20]])
    val_indices = torch.tensor(val_indices)

    X_val = train_dataset.data[val_indices].float() / 255.0 * 10
    X_val = X_val.flatten(start_dim=1)
    y_val = train_dataset.targets[val_indices]

    all_indices = torch.arange(len(train_dataset))
    train_indices = all_indices[~torch.isin(all_indices, val_indices)]
    train_dataset = Subset(train_dataset, train_indices)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)


    # Store training history
    history = {'train_losses': [],
               'train_ce': [],
               'train_volatility': [],
               'train_ceiling': [],
               'test_losses': [],
               'test_ce': [],
               'test_volatility': [],
               'test_ceiling': [],
               'test_accuracy': []}


    def run_mnist_batch(images, labels):
        """"""
        output = network.run(images, device=device)

        model_predictions = network.read_out(output, mode='classification')
        firing_rates = network.get_firing_rates(output, return_as_np_array=False)

        ce_loss = criterion(model_predictions, labels)
        volatility_penalty = compute_fr_volatility_penalty(firing_rates) * lambda_volatility
        ceiling_penalty = compute_fr_ceiling_penalty(firing_rates) * lambda_ceiling

        loss = ce_loss + volatility_penalty + ceiling_penalty
        acc = (labels == torch.argmax(model_predictions, dim=1)).float().mean()

        return loss, ce_loss, volatility_penalty, ceiling_penalty, acc


    # Training loop
    for iteration, (train_images, train_labels) in enumerate(train_loader):
        start = time.time()
        optimizer.zero_grad()

        loss, ce_loss, volatility, ceiling, acc = run_mnist_batch(train_images.to(device), train_labels.to(device))

        loss.backward()
        optimizer.step()

        print('Train loss, accuracy | {:.5f} | {:.2f} | {:.1f}s'.format(loss.item(), acc, time.time() - start))

        history['train_losses'].append(loss.item())
        history['train_ce'].append(ce_loss.item())
        history['train_volatility'].append(volatility.item())
        history['train_ceiling'].append(ceiling.item())

        # Test
        if iteration % test_freq == 0:
            with torch.no_grad():

                test_loss, test_ce, test_volatility, test_ceiling, test_acc = run_mnist_batch(
                    X_val.to(device), y_val.to(device))

                print('TEST loss, accuracy | {:.5f} | {:.2f}'.format(test_loss.item(), test_acc))

                history['test_losses'].append(test_loss.item())
                history['test_ce'].append(test_ce.item())
                history['test_volatility'].append(test_volatility.item())
                history['test_ceiling'].append(test_ceiling.item())
                history['test_accuracy'].append(test_acc.item())

        # Store training history and trained network
        torch.save(history, models_path('mnist', f'mnist_history_{seed}.pt'))
        network.save(models_path('mnist', f'mnist_{seed}.pt'))



if __name__ == '__main__':

    seed = 1
    device = torch.device('cuda')

    train_mnist(seed, device)


