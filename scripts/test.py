import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle


'''
Code for classifying two column outputs as the winner and the loser column :)
'''

'''
batch_size = 20

# Load the data
with open('../pickled_ds/states_dmf_17.pkl', 'rb') as f:
    ds = pickle.load(f)
states, stims = ds['states'], ds['stims']

# Prepare data into input data and targets
nr_samples = len(states)
input_data = torch.Tensor(nr_samples, 2)
targets = torch.Tensor(nr_samples)

for i in range(nr_samples):
    state = states[i]

    # # Compute mean firing rate of both columns, layer 2/3e
    # mean_fr_col_A = state[:, 2, 0].mean()
    # mean_fr_col_B = state[:, 2, 8].mean()
    # input_data[i] = torch.tensor([mean_fr_col_A, mean_fr_col_B])

    # Get last firing rate of both columns, layer 2/3e
    last_fr_col_A = state[-1, 2, 0]
    last_fr_col_B = state[-1, 2, 8]
    input_data[i] = torch.tensor([last_fr_col_A, last_fr_col_B])

    # Identify winning and losing column
    if stims[i, 2] > stims[i, 10]:
        targets[i] = 1.0  # column A wins
    else:
        targets[i] = 0.0  # column B wins

# Train and test set
train_x, test_x = input_data[:900], input_data[900:]
train_y, test_y = targets[:900], targets[900:]
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Init classifier and optimizer
classifier = nn.Linear(2, 1)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1.0)
loss_func = nn.BCEWithLogitsLoss()

# Training loop
for batch_i, (inputs, targets) in enumerate(train_loader):
    optimizer.zero_grad()

    raw_preds = classifier(inputs)
    preds = torch.sigmoid(raw_preds)

    loss = torch.mean(abs(preds - targets.unsqueeze(1)))
    # loss = loss_func(preds, targets.unsqueeze(1))
    loss.backward()
    optimizer.step()

    print('Training loss', loss.item())

    # for param in classifier.parameters():
    #     if param.grad is not None:
    #         print(param.grad.abs().max())

    with torch.no_grad():
        test_sample = test_x[batch_i]
        test_target = test_y[batch_i]

        test_pred_raw = classifier(test_sample)
        test_pred = torch.sigmoid(test_pred_raw)

        # print(test_sample[0] - test_sample[1])
        print(test_pred.item())
        # print(1.0 if test_pred > 0.1 else 0.0)
        print(test_target.item())
        # print(loss_func(test_pred, test_target.unsqueeze(0)))
        print(torch.mean(abs(test_pred - test_target)))
        print()
        
        
'''
