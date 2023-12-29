import argparse
import torch

from ContactDataSet import ContactDataSet
from NeuralNetwork import NeuralNetwork
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import plot_confusion_matrix


def train_loop(dataloader, model, loss_fn, optimizer):
    """
        Runs the training loop for an epoch.
        
        Args:
            dataloader (DataLoader): Dataset for training.
            model (NeuralNetwork): Model to be trained.
            loss_fn (nn.BCELoss): Loss function to be minimized.
            optimizer (torch.optim): Optimizer used for minimized loss_fn.

        Returns:
            train_loss (float): Loss in this epoch.
            train_accuracy (float): Accuracy in this epoch. 
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    train_loss, train_accuracy = 0.0, 0.0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
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
    """
        Runs the testing loop for an epoch.

        Args:
            dataloader (Dataloader): Dataset used for testing.
            model (NeuralNetwork): Trained model used for inference.
            loss_fn (nn.BCELoss): Loss function used for computing performance metrics.

            Returns:
            test_loss (float): Loss in this epoch.
            test_accuracy (float): Accuracy in this epoch.
            test_precision (float): Precision in this epoch.
            test_recall (float): Recall in this epoch.
            test_f1_score (float): F1 score in this epoch.
            cm (confusion_matrix): Confusion matrix in this epoch.
    """
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0.0

    # Evaluating the model with torch.inference_mode() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.inference_mode():
        for _, (X, y) in enumerate(dataloader):
            pred = model(X)
            test_loss += loss_fn(pred, y)
        test_loss /= num_batches
        predictions = model(dataloader.dataset.data)
    
    # Compute the confusion matrix
    cm = confusion_matrix(dataloader.dataset.labels.argmax(1).detach().numpy(), 
                          predictions.argmax(1).detach().numpy())
    
    # Classification metrics
    TP = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    TN = cm[1,1]
    test_accuracy = (TP + TN) / (TP + FP + TN + FN)
    test_precision = TP / (TP + FP)
    test_recall =  TP / (TP + FN)
    test_f1_score = 2 * test_precision * test_recall / (test_precision + test_recall)
    print(f"Test Error: \n Accuracy: {(100*test_accuracy):>0.1f}% \n")

    return test_loss, test_accuracy, test_precision, test_recall, test_f1_score, cm

def run():
    """
        Runs the training and testing loop for a training and testing dataset. Datasets consist
        F/T and IMU data with the corresponding labels 0 for no contact, 1 for slipping contact and 
        2 for stable contact. 

        Args:
            --train-dataset-csv (csv file): Name of the training file in the csv format.
            --test-dataset-csv (csv file): Name of the testing file in the csv format.
            --batch-size (int): Batch sized used during training.
            --learning-rate (float): Learning rate used during training.
            --epochs (int): Number of epochs used in training.
            --add-noise (boolean): Whether to perturb the data with zero-mean Gaussian noise or not.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset-csv", type=str, help="name of the training csv file")
    parser.add_argument("--test-dataset-csv", type=str, help="name of the testing csv file")
    parser.add_argument("--batch-size", type=int, help="dataset batch size", default=128)
    parser.add_argument("--learning-rate", type=float, help="learning rate used in training",  
                        default=1e-4)
    parser.add_argument("--epochs", type=int, help="number of epochs used in training", 
                        default=10)
    parser.add_argument("--add-noise", help="adds noise to the dataset", default=False, 
                        action="store_true")
    
    # Load the parameters
    args = parser.parse_args()
    training_dataset = ContactDataSet(
        csv_file = args.train_dataset_csv, root_dir="./data", transform=None)
    test_dataset = ContactDataSet(
        csv_file = args.test_dataset_csv, root_dir="./data", transform=None)
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs

    model_name = args.train_dataset_csv[:-len(".csv")]
    writer = SummaryWriter('./runs/' + model_name)

    # Create the loss function and the training/testing DataLoaders
    loss_fn = nn.BCELoss()
    train_dataloader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Create the model and the optimizer
    model = NeuralNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    input, _ = next(iter(train_dataloader))
    writer.add_graph(model, input)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        # Run the training loop
        train_loss, train_accuracy = train_loop(train_dataloader, model, loss_fn, optimizer)
        
        # Save the model weights
        torch.save({'state-dict': model.state_dict}, model_name + ".pth")
        
        # Run the testing loop
        test_loss, test_accuracy, test_precision, test_recall, test_f1_score, cm = test_loop(
            test_dataloader, model, loss_fn)
        
        # Save training/testing data
        writer.add_scalar('Train loss', train_loss, epoch)
        writer.add_scalar('Train accuracy', train_accuracy, epoch)
        writer.add_scalar('Test loss', test_loss, epoch)
        writer.add_scalar('Test accuracy', test_accuracy, epoch)
        writer.add_scalar('Test precision', test_precision, epoch)
        writer.add_scalar('Test recall', test_recall, epoch)
        writer.add_scalar('Test f1 score', test_f1_score, epoch)
        writer.add_figure("Confusion Matrix", plot_confusion_matrix(cm, ["stable-contact", 
                                                                         "unstable-contact"]), 
                                                                         epoch)
    
    writer.close()
    print("Training Finished!")

if __name__ == "__main__":
    run()
