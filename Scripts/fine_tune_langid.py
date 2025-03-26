import torch
import torchaudio
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier
import torch.nn as nn
import torch.optim as optim
from dataset_audio_multiclas import AudioMultiDataset
from torch.utils.data import DataLoader,random_split

model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", 
                                       savedir="tmp_model")

num_classes = 3 # Ajusta esto según tu dataset
model.classifier = nn.Linear(in_features=256, out_features=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

path_dataset = r'F:\common_voice\Proyecto\Scripts\data_proc\audios_proc'
dataset = AudioMultiDataset(audios_dir=path_dataset)

torch.manual_seed(42) # Keep subsets equal 
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size  # Ajusta el tamaño restante
train_subset, val_subset, test = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=True  )



def train_model(model, train_loader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), torch.tensor(labels).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")

train_model(model, train_loader)