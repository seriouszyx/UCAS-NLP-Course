import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from cat_and_dog.dataset import MyData, dataset_split
from cat_and_dog.model import *
from cat_and_dog.train import train, evaluate

if __name__ == '__main__':
    # Prepare the data
    DATA_PATH = 'data'
    train_path = f'{DATA_PATH}/train/'
    val_path = f'{DATA_PATH}/val/'

    train_ds, test_ds = dataset_split(MyData(train_path), 0.8)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(MyData(val_path), batch_size=64, shuffle=True)

    # Build the model
    model = my_resnet_50(pretrained=True)

    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(20):
        train(model, epoch, train_loader, device, optimizer, criterion)
        evaluate(model, val_loader, device, criterion, mode='val')

    # Analyze the model's results
    evaluate(model, test_loader, device, criterion, mode='test')

    torch.save(model, 'model.pkl')