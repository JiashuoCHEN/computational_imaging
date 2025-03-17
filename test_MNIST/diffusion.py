import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from pathlib import Path

class SimpleDiffusionModel(nn.Module):
    """定义一个简单的扩散模型，包括编码器和解码器"""
    def __init__(self):
        super(SimpleDiffusionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def load_image(image_path: str) -> np.ndarray:
    """加载并验证图像"""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot decode image file: {image_path}")
    
    print(f"Successfully loaded image | Shape: {image.shape} | Data type: {image.dtype}")
    return image

def apply_gaussian_noise(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """应用高斯噪声"""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    cv2.imwrite('noisy_image.jpg', noisy_image)
    print(f"Generated Gaussian noisy image | sigma={sigma}")
    return noisy_image

def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """图像预处理"""
    # 转换为RGB并归一化
    processed = image.astype(np.float32) / 255.0
    transform = transforms.ToTensor()
    return transform(processed).unsqueeze(0)

def postprocess_image(tensor: torch.Tensor) -> Image.Image:
    """后处理生成PIL图像"""
    output = tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0) * 255
    output = np.clip(output, 0, 255).astype(np.uint8)
    return Image.fromarray(output)

def train_diffusion_model(model, noisy_image, clean_image, epochs=100, lr=1e-3):
    """训练扩散模型"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(noisy_image)
        loss = criterion(output, clean_image)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    print("Training completed")

def main():
    try:
        # 配置路径
        IMAGE_PATH = 'test.jpg'
        MODEL_PATH = 'diffusion_model.pth'
        TARGET_SIZE = (128, 128)  # 调整后的目标尺寸

        # 主流程
        original = load_image(IMAGE_PATH)
        resized_original = cv2.resize(original, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        noisy = apply_gaussian_noise(resized_original)
        
        # 构建和训练模型
        model = SimpleDiffusionModel()
        model = model.to('cpu')  # 使用CPU

        # 训练数据（这里只是一个示例，实际训练需要更多数据）
        x_train = preprocess_image(noisy)
        y_train = preprocess_image(resized_original)
        
        train_diffusion_model(model, x_train, y_train, epochs=100)  # 增加训练轮数
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved: {MODEL_PATH}")
        
        # 模型推理
        model.eval()
        with torch.no_grad():
            input_tensor = preprocess_image(noisy)
            prediction = model(input_tensor)
            deblurred = postprocess_image(prediction)
        
        # 结果保存
        deblurred.save('deblurred_result.jpg')
        print("Deblurred result saved: deblurred_result.jpg")

        # 可视化
        display_results(
            ('Original Image', Image.fromarray(resized_original)),
            ('Noisy Image', Image.fromarray(noisy)),
            ('Deblurred Image', deblurred)
        )

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        exit(1)

def display_results(*images: tuple[str, Image.Image]):
    """可视化结果"""
    for title, img in images:
        img.show(title=title)

if __name__ == "__main__":
    main()