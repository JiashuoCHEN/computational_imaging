import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# 定义优化后的模型（含残差和注意力机制）
class EnhancedDeblurNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        # 通道注意力模块
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 1),
            nn.Sigmoid()
        )
        self.final_conv = nn.Conv2d(64, 1, 3, padding=1)
        self.residual_conv = nn.Conv2d(1, 1, 1)  # 残差连接卷积

    def forward(self, x):
        x_input = x
        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        attn_weight = self.attn(feat2)
        weighted_feat = feat2 * attn_weight
        output = self.final_conv(weighted_feat)
        residual = self.residual_conv(x_input)
        return output + residual  # 残差连接

# 感知损失（基于VGG，修复通道问题）
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torch.hub.load('pytorch/vision', 'vgg16', pretrained=True)
        # 修改输入通道为1，保持输出通道64
        vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        # 初始化新卷积层权重
        nn.init.kaiming_normal_(vgg.features[0].weight, mode='fan_out', nonlinearity='relu')
        self.vgg = vgg.features[:16].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, deblur_img, clear_img):
        # 确保输入维度正确（B, C, H, W）
        if deblur_img.ndim != 4:
            deblur_img = deblur_img.unsqueeze(0)
        if clear_img.ndim != 4:
            clear_img = clear_img.unsqueeze(0)
        
        # 归一化到VGG输入范围[-1, 1]
        mean = torch.tensor([0.485]).view(1,1,1,1)  # 单通道均值
        std = torch.tensor([0.229]).view(1,1,1,1)   # 单通道标准差
        deblur_normalized = (deblur_img * 255.0 - mean) / std
        clear_normalized = (clear_img * 255.0 - mean) / std
        
        vgg_deblur = self.vgg(deblur_normalized)
        vgg_clear = self.vgg(clear_normalized)
        return nn.functional.mse_loss(vgg_deblur, vgg_clear)

# 可视化函数
def plot_images(blur, clear, deblur, save_dir='./visualization_results'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Blurred Image")
    plt.imshow(blur.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title("Clear Image")
    plt.imshow(clear.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title("Deblurred Image")
    plt.imshow(deblur.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'comparison.png'))
    plt.close()

if __name__ == '__main__':
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    batch_size = 16
    epochs = 15
    learning_rate = 0.001

    # 数据加载与增强
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomApply([transforms.GaussianBlur(5, (0.5, 2.0))], p=1.0),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10)
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_clear_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    test_clear_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = EnhancedDeblurNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_mse = nn.MSELoss()
    criterion_perceptual = PerceptualLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (blur_images, _) in enumerate(train_loader):
            blur_images = blur_images.to(device)
            start_idx = batch_idx * batch_size
            clear_images = train_clear_dataset.data[start_idx:start_idx+batch_size]
            clear_images = clear_images.unsqueeze(1).float().to(device) / 255.0

            optimizer.zero_grad()
            outputs = model(blur_images)
            
            loss_mse = criterion_mse(outputs, clear_images)
            loss_perceptual = criterion_perceptual(outputs, clear_images)
            loss = loss_mse + 0.1 * loss_perceptual
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        scheduler.step()
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}')

    # 测试与可视化
    model.eval()
    with torch.no_grad():
        sample_idx = 0
        blur_image, _ = test_dataset[sample_idx]
        clear_image = test_clear_dataset.data[sample_idx].unsqueeze(0).unsqueeze(1).float() / 255.0
        blur_tensor = blur_image.unsqueeze(0).to(device)
        deblur_tensor = model(blur_tensor)
        
        plot_images(blur_image, clear_image, deblur_tensor)
    
    torch.save(model.state_dict(), 'optimized_mnist_deblur.pth')