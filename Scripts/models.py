import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import timm
import torchvision.transforms as transforms


class BinaryCRNN(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2, cnn_channels=16):
        super(BinaryCRNN, self).__init__()
        
        # Convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels, track_running_stats=False),  # Evitar errores con batch=1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(cnn_channels, cnn_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels * 2, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calcular la salida de la CNN para determinar input_size del LSTM
        self._output_size = self._get_cnn_output_size(input_dim, cnn_channels)

        # LSTM layers
        self.rnn = nn.LSTM(input_size=self._output_size,  
                           hidden_size=hidden_dim, 
                           num_layers=num_layers, 
                           batch_first=True, 
                           bidirectional=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim * 2, 1)  # Salida con un logit para BCEWithLogitsLoss

    def _get_cnn_output_size(self, input_dim, cnn_channels):
        """Calcula el tamaño de la salida después de convoluciones."""
        x = torch.zeros(1, 1, input_dim, 431)  # Simulación de entrada (1 canal, 128 freq, 431 time)
        x = self.cnn(x)
        return x.shape[1] * x.shape[2]  # (canales * altura después de convoluciones)

    def forward(self, x):
        batch_size = x.size(0)

        # Asegurar que la entrada tiene 4 dimensiones (batch, channels, freq_bins, time_frames)
        if x.dim() == 3:  # Si la entrada es (batch, freq_bins, time), añadir canal
            x = x.unsqueeze(1)  # Ahora será (batch, 1, freq_bins, time)

        # CNN feature extraction
        x = self.cnn(x)

        # Reshape para LSTM: (batch, time, features)
        x = x.permute(0, 2, 3, 1).contiguous()  # (batch, freq, time, channels)
        x = x.view(batch_size, x.shape[2], -1)  # (batch, time, features)

        # RNN processing
        x, _ = self.rnn(x)

        # Tomar solo el último time step
        x = self.fc(x[:, -1, :])  
        return x


class BinaryCNN(nn.Module):
    def __init__(self):
        super(BinaryCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),  # Agregar BatchNorm
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        '''
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        '''

        # Calcular salida
        self.fc_input_size = self._get_conv_output_size()

        # Capas densas con activaciones
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        #self.fc2 = nn.Linear(512, 256)
        #self.bn2 = nn.BatchNorm1d(256)
        self.output = nn.Linear(256, 1)

    def _get_conv_output_size(self):
        device = next(self.parameters()).device
        x = torch.zeros(1, 1, 128, 431, device=device)  
        x = self.conv1(x)
        x = self.conv2(x)
        #x = self.conv3(x)
        #x = self.conv4(x)
        return x.numel()

    def forward(self, input_data):
        if input_data.dim() == 3:  
            input_data = input_data.unsqueeze(1)  
        
        x = self.conv1(input_data)
        x = self.conv2(x)
        #x = self.conv3(x)
        #x = self.conv4(x)
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        #x = self.fc2(x)
        #x = self.bn2(x)
        #x = F.relu(x)
        logits = self.output(x)
        return logits 

class ViTClassifier(nn.Module):
    def __init__(self, num_classes=3, img_size=(128, 431), patch_size=16):
        super(ViTClassifier, self).__init__()

        # Cargar ViT preentrenado
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)

        # Modificar la primera capa Conv2d para aceptar 1 canal en vez de 3
        old_weight = self.vit.patch_embed.proj.weight  # (3, 768, 16, 16)
        new_weight = old_weight.mean(dim=1, keepdim=True)  # (1, 768, 16, 16)

        self.vit.patch_embed.proj = nn.Conv2d(
            1, self.vit.patch_embed.proj.out_channels,  # De 1 canal en vez de 3
            kernel_size=self.vit.patch_embed.proj.kernel_size,
            stride=self.vit.patch_embed.proj.stride,
            padding=self.vit.patch_embed.proj.padding,
            bias=False
        )
        self.vit.patch_embed.proj.weight = nn.Parameter(new_weight)

        # Redimensionar espectrogramas a (224, 224)
        self.resize = transforms.Resize((224, 224))

        # Ajustar la capa final para 3 clases
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        x = self.resize(x)  # Redimensionar a (B, 1, 224, 224)
        return self.vit(x)


class MulticlassCRNN(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2, cnn_channels=16):
        super(MulticlassCRNN, self).__init__()
        
        # Convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels, track_running_stats=False),  # Evitar errores con batch=1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(cnn_channels, cnn_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels * 2, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(cnn_channels* 2, cnn_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels * 4, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calcular la salida de la CNN para determinar input_size del LSTM
        self._output_size = self._get_cnn_output_size(input_dim, cnn_channels)

        # LSTM layers
        self.rnn = nn.LSTM(input_size=self._output_size,  
                           hidden_size=hidden_dim, 
                           num_layers=num_layers, 
                           batch_first=True, 
                           bidirectional=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim * 2, 3)  # Salida con tres logit para BCEWithLogitsLoss

    def _get_cnn_output_size(self, input_dim, cnn_channels):
        """Calcula el tamaño de la salida después de convoluciones."""
        x = torch.zeros(1, 1, input_dim, 431)  # Simulación de entrada (1 canal, 128 freq, 431 time)
        x = self.cnn(x)
        return x.shape[1] * x.shape[2]  # (canales * altura después de convoluciones)

    def forward(self, x):
        batch_size = x.size(0)

        # Asegurar que la entrada tiene 4 dimensiones (batch, channels, freq_bins, time_frames)
        if x.dim() == 3:  # Si la entrada es (batch, freq_bins, time), añadir canal
            x = x.unsqueeze(1)  # Ahora será (batch, 1, freq_bins, time)

        # CNN feature extraction
        x = self.cnn(x)

        # Reshape para LSTM: (batch, time, features)
        x = x.permute(0, 2, 3, 1).contiguous()  # (batch, freq, time, channels)
        x = x.view(batch_size, x.shape[2], -1)  # (batch, time, features)

        # RNN processing
        x, _ = self.rnn(x)

        # Tomar solo el último time step
        x = self.fc(x[:, -1, :])  
        return x


if __name__ == "__main__":
    cnn = BinaryCRNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = ViTClassifier(num_classes=3)
    model.to(device)
    print(model)