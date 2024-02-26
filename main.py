# from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
# from torch_geometric.datasets import Planetoid
import torch_geometric.datasets.citation_full as citation_full
import torch
# import dgl
import torch.nn as nn
from gat import GAT


def main() -> None:
    """Main function for training GAT on the CORA dataset.
    """
    # Get CORA dataset from DGL

   # Create DataLoader
    # open ./data/cora/cora.content
    # open ./data/cora/cora.cites
    # Create DataLoader


    dataset = citation_full.CitationFull(root='./data/CoraML', name='cora_ML')

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



    # Create model, try setting to cuda, cpu if not available
    device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(dataset.num_features, dataset.num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)

    # Check if there is a save in the saves/ directory, if so load it
    try:
        model.load_state_dict(torch.load("saves/gat.pth"))
        print("Loaded model from saves/gat.pth")
    except FileNotFoundError:
        print("No save found, training from scratch")

    print("Training")

    # Training loop

    initial = 0
    best = 100000000000


    # Look at data
    for data in dataloader:
        freq = {}
        for i in data.y:
            if i.item() in freq:
                freq[i.item()] += 1
            else:
                freq[i.item()] = 1
        print(freq)

    for epoch in range(100):
        total_loss = 0
        for data in dataloader:
            # data = data.to(device)
            # Convert targets to float
            optimizer.zero_grad()
            output = model.forward(data)

            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if total_loss < best:
            best = total_loss
        if initial == 0:
            initial = total_loss
        print(f"Epoch {epoch+1}: Loss = {total_loss} (Initial = {initial}, Best = {best})")


    # Open saves directory and save GAT model
    torch.save(model.state_dict(), "saves/gat.pth")


    model.eval()
    for data in dataloader:
        output = model.forward(data)
        output = torch.argmax(output, 1)
        correct = 0
        for i in range(len(output)):
            if output[i] == data.y[i]:
                correct += 1
        print(f"Accuracy: {100*correct/len(output)}%")
    return

main()
