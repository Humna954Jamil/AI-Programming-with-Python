import torch
from torch import nn, optim
import torchvision
from torch.utils.data import DataLoader
import futility  
import fmodel  
import argparse

parser = argparse.ArgumentParser(description="Training an Image Classifier for the Flower Dataset")
parser.add_argument('data_dir', action="store", default="./flowers/")
parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
parser.add_argument('--arch', action="store", default="vgg16", choices=["vgg16", "densenet121", "resnet18"])
parser.add_argument('--learning_rate', action="store", type=float, default=0.001)
parser.add_argument('--hidden_units', action="store", type=int, default=512)
parser.add_argument('--epochs', action="store", default=3, type=int)
parser.add_argument('--dropout', action="store", type=float, default=0.2)
parser.add_argument('--gpu', action="store", default="gpu", choices=["cpu", "gpu"])

args = parser.parse_args()

# Read the arguments
data_dir = args.data_dir
save_dir = args.save_dir
learning_rate = args.learning_rate
architecture = args.arch
hidden_units = args.hidden_units
dropout = args.dropout
epochs = args.epochs
device_choice = args.gpu


device = torch.device("cuda" if torch.cuda.is_available() and device_choice == 'gpu' else "cpu")

def main():
    # Load the data using futility's helper function
    train_loader, valid_loader, test_loader, train_data = futility.load_data(data_dir)
    
    # Set up the model and loss function using fmodel
    model, criterion = fmodel.setup_network(architecture, dropout, hidden_units, learning_rate, device_choice)
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Training loop
    print("Training started...")
    steps = 0
    running_loss = 0
    print_every = 5  # Print the training stats every 5 steps

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        for inputs, labels in train_loader:
            steps += 1
            # Move data to GPU or CPU
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            
            if steps % print_every == 0:
                model.eval()  
                valid_loss = 0
                accuracy = 0

                with torch.no_grad():  
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        log_ps = model(inputs)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Accuracy: {accuracy/len(valid_loader):.3f}")
                
                running_loss = 0
                model.train()  # Set model back to training mode

    # Save the model checkpoint after training
    model.class_to_idx = train_data.class_to_idx  # Class indices for later use
    torch.save({
        'model_architecture': architecture,
        'hidden_units': hidden_units,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }, save_dir)

    print("Training complete. Model saved to", save_dir)

if __name__ == "__main__":
    main()
