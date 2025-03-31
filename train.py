
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

class Haze4kDataset(Dataset):
    def __init__(self, hazy_dir, gt_dir, transform=None):
        self.hazy_dir = hazy_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.hazy_images = sorted(os.listdir(hazy_dir))
        self.gt_images = sorted(os.listdir(gt_dir))
        assert len(self.hazy_images) == len(self.gt_images), "Mismatch between hazy and GT images"

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_images[idx])

        hazy_img = Image.open(hazy_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        if self.transform:
            hazy_img = self.transform(hazy_img)
            gt_img = self.transform(gt_img)

        return hazy_img, gt_img

class FrequencyCompensationBlock(nn.Module):
    def __init__(self, channels):
        super(FrequencyCompensationBlock, self).__init__()
        self.high_freq = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.mid_freq = nn.Conv2d(channels, channels, 5, padding=2, bias=False)
        self.fusion = nn.Conv2d(channels * 2, channels, 1)  # Combine features
        self.relu = nn.ReLU()

        with torch.no_grad():
            # High-frequency filter (Sobel-like, repeated across input channels)
            high_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
            self.high_freq.weight.data = high_kernel.repeat(channels, channels, 1, 1) / channels  # Normalize
            # Mid-frequency filter (larger kernel, repeated across input channels)
            mid_kernel = torch.tensor([[[[-1, -1, 0, 1, 1], [-2, -2, 0, 2, 2], [-3, -3, 0, 3, 3], [-2, -2, 0, 2, 2], [-1, -1, 0, 1, 1]]]], dtype=torch.float32)
            self.mid_freq.weight.data = mid_kernel.repeat(channels, channels, 1, 1) / channels  

    def forward(self, x):
        high = self.high_freq(x)
        mid = self.mid_freq(x)
        combined = torch.cat([high, mid], dim=1)
        return self.relu(self.fusion(combined))

class DehazeUNet(nn.Module):
    def __init__(self):
        super(DehazeUNet, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

        self.middle = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.fcb = FrequencyCompensationBlock(256)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        m = self.middle(e3)
        m = m + self.fcb(m)  # Residual connection with FCB

        d1 = self.up1(m)
        d1 = torch.cat([d1, e2], dim=1)  # Skip connection
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)  # Skip connection
        d2 = self.dec2(d2)
        out = torch.tanh(self.out(d2))  # Output in [-1, 1]
        return out

def tensor_to_image(tensor):
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0, 1)  
    return tensor.permute(1, 2, 0).cpu().numpy()  # Convert CHW to HWC for matplotlib

transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Paths to Datasets (Corrected validation paths)
train_hazy_dir = '/kaggle/input/haze4k-t/Haze4K-T/IN'
train_gt_dir = '/kaggle/input/haze4k-t/Haze4K-T/GT'
val_hazy_dir = '/kaggle/input/haze4k-v/Haze4K-V/IN'  # Fixed typo from Haze4K-V to Haze4K_V
val_gt_dir = '/kaggle/input/haze4k-v/Haze4K-V/GT'    # Fixed typo from Haze4K-V to Haze4K_V

train_dataset = Haze4kDataset(train_hazy_dir, train_gt_dir, transform=transform)
val_dataset = Haze4kDataset(val_hazy_dir, val_gt_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = DehazeUNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10  # Adjust as needed
output_dir = '/kaggle/working/checkpoints'
Path(output_dir).mkdir(parents=True, exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for hazy, gt in train_loader:
        hazy, gt = hazy.to(device), gt.to(device)

        optimizer.zero_grad()
        output = model(hazy)
        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (hazy, gt) in enumerate(val_loader):
            hazy, gt = hazy.to(device), gt.to(device)
            output = model(hazy)
            loss = criterion(output, gt)
            val_loss += loss.item()

            if i == 0:
                hazy_img = tensor_to_image(hazy[0])
                gt_img = tensor_to_image(gt[0])
                output_img = tensor_to_image(output[0])
            break  # Only need one batch for visualization

    val_loss /= len(val_loader)
    print(f'Validation Loss: {val_loss:.4f}')

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(hazy_img)
    plt.title('Hazy Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(gt_img)
    plt.title('Ground Truth')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(output_img)
    plt.title('Model Output')
    plt.axis('off')
    plt.show()

    torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{epoch+1}.pth'))

#print("Training complete!")
