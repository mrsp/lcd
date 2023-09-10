import argparse
import torch
import torchvision

from ContactDataSet import ContactDataSet
from NeuralNetwork import NeuralNetwork
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    train_loss, train_accuracy = 0.0, 0.0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        model.train()
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss
        train_accuracy += (pred.argmax(1) == y.argmax(1)).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            current = (batch + 1) * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
    return train_loss / num_batches, train_accuracy / size

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, test_accuracy = 0.0, 0.0

    # Evaluating the model with torch.inference_mode() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.inference_mode():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            test_accuracy += (pred.argmax(1) == y.argmax(1)).sum().item()

    test_loss /= num_batches
    test_accuracy /= size
    print(f"Test Error: \n Accuracy: {(100*test_accuracy):>0.1f}% \n")
    return test_loss, test_accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset-csv", type = str, help = "name of the training csv file")
    parser.add_argument("--test-dataset-csv", type = str, help = "name of the testing csv file")
    parser.add_argument("--batch-size", type = int, help = "dataset batch size", default = 128)
    parser.add_argument("--learning-rate", type = float, help = "learning rate used in training",  
                        default = 1e-4)
    parser.add_argument("--epochs", type = int, help = "number of epochs used in training", 
                        default = 10)
    parser.add_argument("--add-noise", help = "adds noise to the dataset", default = False, 
                        action = "store_true")
    # Load the parameters
    args = parser.parse_args()
    training_dataset = ContactDataSet(
        csv_file = args.train_dataset_csv, root_dir="./data", transform=None)
    test_dataset = ContactDataSet(
        csv_file = args.test_dataset_csv, root_dir="./data", transform=None)
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs

    writer = SummaryWriter('./runs/' + args.train_dataset_csv.removesuffix(".csv"))
    # Create the loss function and the training/testing DataLoaders
    loss_fn = nn.BCELoss()
    train_dataloader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = NeuralNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    input, labels = next(iter(train_dataloader))
    writer.add_graph(model, input)
    writer.close()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        # Run the training loop
        train_loss, train_accuracy = train_loop(train_dataloader, model, loss_fn, optimizer)
        
        # Save the model weights
        torch.save(model.state_dict,"ATLAS_MODEL_WEIGHTS.pth")
        
        # Run the testing loop
        test_loss, test_accuracy = test_loop(test_dataloader, model, loss_fn)
        
        # Save training/testing data
        writer.add_scalar('train loss', train_loss, epoch)
        writer.add_scalar('train accuracy', train_accuracy, epoch)
        writer.add_scalar('test loss', test_loss, epoch)
        writer.add_scalar('test accuracy', test_accuracy, epoch)

    print("Training Finished!")

if __name__ == "__main__":
    main()
