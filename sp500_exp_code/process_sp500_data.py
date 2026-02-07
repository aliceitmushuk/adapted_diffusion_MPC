import yfinance
import numpy as np
import torch
import os

def set_seed(seed=None):
    """Set random seed for reproducibility"""  
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    return seed

seed = 3407
set_seed(seed)

## Create data directory
data_dir = './data/sp500/'
os.makedirs(data_dir, exist_ok=True)

## Download S&P500 data from yahoo finance
raw_data = yfinance.download (tickers = "^GSPC", start = "1989-12-29", 
                              end = "2024-01-03", interval = "1d")
raw_data.to_csv(f'{data_dir}/sp500.csv')


## Equally split the data into 40 segments of 127 days (half a year) each
price_data = raw_data['Close'].to_numpy()
sample_path = [price_data[(i - 1) * 126: 126 * i + 1] for i in range(1, 41)]
sample_path = np.array(sample_path).reshape(-1, 127)
sample_path /= sample_path[:, [0]] # normalize the price path by the initial price

## Save the first 40 training paths (1989-2010) for training the SDE generative model
# torch.save(torch.tensor(sample_path), f'{data_dir}train_sde_path.pt')
# np.save(f'{data_dir}train_sde_path.npy', sample_path)

## Save the entire training price trajectory (1990-2010)
train_price = raw_data.iloc[:5041]['Close'].to_numpy()[:, 0]
# torch.save(torch.tensor(train_price), f'{data_dir}train_price_path.pt')


## Generate test paths from 2010 to 2019 by return block bootstrapping
test_price = raw_data.iloc[5041:7562,]['Close'].to_numpy()[:, 0]
test_return = (test_price[1:] - test_price[:-1]) / test_price[:-1]
test_return = test_return.reshape(-1, 21)
num_month = test_return.shape[0]
test_path = []
for _ in range(100000):
    month_idx = np.random.randint(0, num_month, 6)  # sample indices of 6 months
    temp_start = torch.tensor([test_price[month_idx[0] * 21]])  # starting price of the first month
    temp_return = torch.from_numpy(test_return[month_idx].reshape(-1))  # sampled returns
    temp_path = torch.cat([temp_start, temp_return])    # concatenate starting price and returns
    test_path.append(temp_path)
test_path = torch.stack(test_path)
for i in range(1, test_path.shape[1]):
        test_path[:, i] = test_path[:, i - 1] * (1 + test_path[:, i])   # reconstruct price path
test_path /= test_path[:, [0]]  # normalize the price path by the initial price

## Save the generated test paths
torch.save(torch.tensor(test_path), f'{data_dir}test_path_2010_2019_seed{seed}.pt')



