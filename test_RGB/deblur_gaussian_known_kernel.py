import cv2
import numpy as np

def deblur_gaussian_known_kernel(blur_path, kernel_size=(5,5), sigma=0):
    """
    已知高斯核的逆卷积恢复（分通道处理）
    
    参数:
        blur_path: 模糊图像路径
        kernel_size: 高斯核大小（高, 宽）
        sigma: 高斯核标准差（0表示自动计算）
    """
    # 读取模糊图像
    blurred = cv2.imread(blur_path)
    if blurred is None:
        print(f"Error: Could not read {blur_path}")
        return
    
    # 构造原高斯核
    kernel = cv2.getGaussianKernel(kernel_size[0], sigma)
    kernel = kernel @ kernel.T  # 生成二维高斯核
    
    # 频域逆滤波（分通道处理）
    restored_channels = []
    for channel in range(3):  # 遍历每个颜色通道
        # 当前通道图像
        channel_img = blurred[:, :, channel]
        
        # 扩展核到图像尺寸
        kernel_padded = np.zeros_like(channel_img, dtype=np.float32)
        h, w = kernel.shape
        kernel_padded[:h, :w] = kernel
        
        # 频域处理
        kernel_fft = np.fft.fft2(kernel_padded)
        inv_kernel_fft = 1 / kernel_fft  # 逆核
        
        # 恢复当前通道
        channel_fft = np.fft.fft2(channel_img)
        restored_channel_fft = channel_fft * inv_kernel_fft
        restored_channel = np.fft.ifft2(restored_channel_fft).astype(np.uint8)
        
        restored_channels.append(restored_channel)
    
    # 合并通道
    restored = np.stack(restored_channels, axis=-1)
    
    # 保存结果
    save_path = blur_path.replace('.jpg', '_restored_gaussian_known.jpg')
    cv2.imwrite(save_path, restored)
    return restored

def deblur_gaussian_wiener(blur_path, kernel_size=(5,5), sigma=0, snr=0.1):
    """
    基于维纳滤波的高斯模糊恢复（分通道处理）
    
    参数:
        snr: 信噪比（噪声方差/信号方差）
    """
    blurred = cv2.imread(blur_path)
    if blurred is None:
        print(f"Error: Could not read {blur_path}")
        return
    
    # 构造原高斯核
    kernel = cv2.getGaussianKernel(kernel_size[0], sigma)
    kernel = kernel @ kernel.T
    
    # 分通道维纳滤波
    restored_channels = []
    for channel in range(3):
        channel_img = blurred[:, :, channel]
        
        # 扩展核
        kernel_padded = np.zeros_like(channel_img, dtype=np.float32)
        h, w = kernel.shape
        kernel_padded[:h, :w] = kernel
        
        # 频域处理
        kernel_fft = np.fft.fft2(kernel_padded)
        inv_kernel_fft = np.conj(kernel_fft) / (np.abs(kernel_fft)**2 + snr)
        
        # 恢复通道
        channel_fft = np.fft.fft2(channel_img)
        restored_channel_fft = channel_fft * inv_kernel_fft
        restored_channel = np.fft.ifft2(restored_channel_fft).astype(np.uint8)
        
        restored_channels.append(restored_channel)
    
    # 合并通道
    restored = np.stack(restored_channels, axis=-1)
    
    # 保存结果
    save_path = blur_path.replace('.jpg', '_restored_wiener.jpg')
    cv2.imwrite(save_path, restored)
    return restored

# 使用示例
if __name__ == "__main__":
    input_image = 'gaussian_blur.jpg'  # 输入模糊图像路径
    
    # 基础逆卷积恢复
    deblur_gaussian_known_kernel(input_image, kernel_size=(5,5), sigma=2)
    
    # 维纳滤波恢复（推荐）
    deblur_gaussian_wiener(input_image, kernel_size=(5,5), sigma=2, snr=0.01)