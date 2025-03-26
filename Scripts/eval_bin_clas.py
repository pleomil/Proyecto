from torch.utils.data import DataLoader,random_split
from models import BinaryCRNN, BinaryCNN
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_clas_ruido import AudioDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo CNN
cnn_model = BinaryCNN().to(device)
cnn_model.load_state_dict(torch.load("models/best_model_CNN.pth", map_location=device))
cnn_model.eval()

# Cargar modelo CRNN
crnn_model = BinaryCRNN().to(device)
crnn_model.load_state_dict(torch.load("models/best_model_CRNN.pth", map_location=device))
crnn_model.eval()

path_dataset = r'Scripts\data_proc'
dataset = AudioDataset(root_dir=path_dataset)

torch.manual_seed(42) # Keep subsets equal 
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size  # Ajusta el tamaño restante
train_subset, val_subset, test = random_split(dataset, [train_size, val_size, test_size])
test_loader = DataLoader(test, batch_size=32, shuffle=False)


def get_predictions(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            inputs = inputs.squeeze(2)  # Ajustar la forma de entrada si es necesario
            
            outputs = model(inputs).squeeze(1)  # Convertir logits a valores escalares
            preds = torch.sigmoid(outputs) > 0.5  # Convertir logits a 0 o 1
            
            all_preds.extend(preds.cpu().numpy())  # Guardar predicciones
            all_labels.extend(labels.cpu().numpy())  # Guardar etiquetas reales

    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Voz", "Ruido"], yticklabels=["Voz", "Ruido"])
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta Real")
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.show()

# Obtener etiquetas y predicciones para cada modelo
y_true_cnn, y_pred_cnn = get_predictions(cnn_model, test_loader)
y_true_crnn, y_pred_crnn = get_predictions(crnn_model, test_loader)

# Graficar matriz de confusión
plot_confusion_matrix(y_true_cnn, y_pred_cnn, "CNN")
plot_confusion_matrix(y_true_crnn, y_pred_crnn, "CRNN")

# Reporte de clasificación
print("CNN Classification Report:\n", classification_report(y_true_cnn, y_pred_cnn))
print("CRNN Classification Report:\n", classification_report(y_true_crnn, y_pred_crnn))