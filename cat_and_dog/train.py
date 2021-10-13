import torch


def train(model, epoch, train_loader, device, optimizer, criterion):
    model = model.to(device)
    model.train()

    total_num = len(train_loader.dataset)
    train_loss = 0
    correct_num = 0

    for i, data in enumerate(train_loader, 0):
        my_input, my_label = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()  # 初始为0，清除上个batch的梯度信息
        my_output = model(my_input)
        loss = criterion(my_output, my_label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * my_label.size(0)
        predict = torch.argmax(my_output, dim=-1)
        correct_num += my_label.eq(predict).sum()

    train_loss = train_loss / total_num
    train_acc = correct_num / total_num
    print('epoch: {} --> train_loss: {:.6f} - train_acc: {:.6f} - '.format(
        epoch, train_loss, train_acc), end='')


def evaluate(model, eval_loader, device, criterion, mode='val'):
    model = model.to(device)
    model.eval()

    total_num = len(eval_loader.dataset)
    eval_loss = 0
    correct_num = 0

    for i, data in enumerate(eval_loader, 0):
        my_input, my_label = data[0].to(device), data[1].to(device)

        my_output = model(my_input)
        loss = criterion(my_output, my_label)

        eval_loss += loss.item() * my_label.size(0)
        predict = torch.argmax(my_output, dim=-1)
        correct_num += my_label.eq(predict).sum()

    eval_loss = eval_loss / total_num
    eval_acc = correct_num / total_num

    print('{}_loss: {:.6f} - {}_acc: {:.6f}'.format(
        mode, eval_loss, mode, eval_acc))
