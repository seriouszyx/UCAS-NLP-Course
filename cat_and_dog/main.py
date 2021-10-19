import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from cat_and_dog import config
from cat_and_dog.dataset import MyData, dataset_split
from cat_and_dog.model import *
from cat_and_dog.train import train, evaluate

if __name__ == '__main__':
    # Prepare the data
    TRAIN_PATH = config.TRAIN_PATH
    VAL_PATH = config.VAL_PATH
    BATCH_SIZE = config.BATCH_SIZE

    train_ds, test_ds = dataset_split(MyData(TRAIN_PATH), config.TRAIN_RATE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MyData(VAL_PATH), batch_size=BATCH_SIZE, shuffle=True)

    # Build the model
    model = my_dnnnet()

    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(config.NUM_EPOCHS):
        train(model, epoch, train_loader, device, optimizer, criterion)
        evaluate(model, val_loader, device, criterion, mode='val')

    # Analyze the model's results
    evaluate(model, test_loader, device, criterion, mode='test')

    torch.save(model, 'model.pkl')