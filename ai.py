import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import time

# Cihazı ayarlayın (GPU varsa CUDA, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan cihaz:", device)

# Veriyi yükle
file_path = "C:/Users/alica/Desktop/10k_data.txt"
data = pd.read_csv(file_path, header=None, names=["M", "Peri", "Node", "Incl.", "E", "N", "A", "Epoch"], sep="_")

# Gelecekteki pozisyonu (Future_Position) rastgele değerlerle dolduralım
data["Future_Position"] = np.random.uniform(2.0, 3.5, size=len(data))

# Özellikleri (features) ve hedef değeri (target) ayır
X = data[["M", "Peri", "Node", "Incl.", "E", "N", "A", "Epoch"]].values
y = data["Future_Position"].values

# Veriyi eğitim, doğrulama ve test setlerine bölme (%70 Eğitim, %20 Doğrulama, %10 Test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # %70 Eğitim
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)  # %20 Doğrulama, %10 Test

# Veriyi ölçeklendir
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Veriyi PyTorch tensörlerine dönüştür ve GPU'ya taşı
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# PyTorch DataLoader ile veri kümelerini oluştur
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=512, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=512, num_workers=0, pin_memory=True)

# Modeli oluştur
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = SimpleNN().to(device)

# Optimizasyon ve düzenlileştirme
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

# Eğitim döngüsü
epochs = 100
start_time = time.time()
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Scheduler adımı
    scheduler.step()

    # Doğrulama seti üzerinde değerlendirme
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    val_loss /= len(val_loader)

    # Her %10 ilerlemede çıktı
    if (epoch + 1) % (epochs // 10) == 0:
        print(f"%{(epoch + 1) / epochs * 100:.0f} tamamlandı - Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

end_time = time.time()
print(f"Eğitim süresi: {end_time - start_time:.2f} saniye")

# Test seti üzerinde değerlendirme
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        test_loss += criterion(outputs, targets).item()
test_loss /= len(test_loader)
print(f"Test seti üzerinde ortalama kayıp değeri: {test_loss:.4f}")
