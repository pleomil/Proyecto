from torch.utils.data import DataLoader,random_split
from models import BinaryCRNN, BinaryCNN
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_clas_ruido import AudioDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

path_dataset = r'Scripts\data_proc'


if __name__ == "__main__":
    model = BinaryCRNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    # Para almacenar métricas
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float("inf")

    def accuracy(y_pred, y_true):
        preds = (torch.sigmoid(y_pred) > 0.5).float()
        return (preds == y_true).float().mean().item()

    # Datos sintéticos

    dataset = AudioDataset(root_dir=path_dataset)

    torch.manual_seed(42) # Keep subsets equal 
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size  # Ajusta el tamaño restante
    train_subset, val_subset, test = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=True  )
    
    # Entrenamiento
    num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Asegurar que el modelo está en modo entrenamiento
    train_loss, train_acc = 0.0, 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for batch in progress_bar:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        inputs = inputs.squeeze(2)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs.squeeze(1), labels.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy(outputs.squeeze(1), labels)

    # Promediar métricas
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # Evaluación en validación
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            inputs = inputs.squeeze(2)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), labels.float())

            val_loss += loss.item()
            val_acc += accuracy(outputs.squeeze(1), labels)

    # Promediar métricas de validación
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    # Guardar métricas
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    # Guardar el mejor modelo basado en la pérdida de validación
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f'Saving new model at {val_loss:.4f} val_loss')
        torch.save(model.state_dict(), "models/best_model_CRNN.pth")

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Graficar resultados
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1,2,2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")

plt.show()