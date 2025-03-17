import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def load_image(image_path: str) -> np.ndarray:
    """加载并验证图像"""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法解码图像文件: {image_path}")
    
    print(f"成功加载图像 | 尺寸: {image.shape} | 数据类型: {image.dtype}")
    return image

def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """应用高斯模糊"""
    kernel = cv2.getGaussianKernel(kernel_size, sigma) @ cv2.getGaussianKernel(kernel_size, sigma).T
    blurred = cv2.filter2D(image, -1, kernel)
    cv2.imwrite('blurred_image.jpg', blurred)
    print(f"已生成高斯模糊图像 | 核大小: {kernel_size}x{kernel_size} | sigma={sigma}")
    return blurred

def build_unet_model(input_shape: tuple) -> tf.keras.Model:
    """构建 U-Net 模型用于去模糊"""
    inputs = Input(shape=input_shape)
    
    # 下采样路径
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), padding='same')(conv1)
    
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), padding='same')(conv2)
    
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    
    # 上采样路径
    up4 = UpSampling2D((2, 2))(conv3)
    merge4 = Concatenate()([conv2, up4])
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    
    up5 = UpSampling2D((2, 2))(conv4)
    merge5 = Concatenate()([conv1, up5])
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    
    outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv5)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    print("成功构建 U-Net 模型")
    return model

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """图像预处理"""
    # 转换为RGB并归一化
    processed = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    return np.expand_dims(processed, axis=0)

def postprocess_image(prediction: np.ndarray) -> Image.Image:
    """后处理生成PIL图像"""
    output = (np.squeeze(prediction, axis=0) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output)

def display_results(*images: tuple[str, Image.Image]):
    """可视化结果"""
    for title, img in images:
        img.show(title=title)

def main():
    try:
        # 配置路径
        IMAGE_PATH = 'blurred_test.jpg'
        MODEL_PATH = 'unet_deblur_model.keras'
        TARGET_SIZE = (128, 128)

        # 主流程
        original = load_image(IMAGE_PATH)
        resized_original = cv2.resize(original, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        blurred = apply_gaussian_blur(resized_original)
        
        # 构建和训练模型
        input_shape = (TARGET_SIZE[0], TARGET_SIZE[1], 3)
        model = build_unet_model(input_shape)
        
        # 训练数据
        x_train = preprocess_image(blurred)
        y_train = preprocess_image(resized_original)
        
        model.fit(x_train, y_train, epochs=50, batch_size=1)
        model.save(MODEL_PATH)
        print(f"已保存模型: {MODEL_PATH}")
        
        # 模型推理
        input_tensor = preprocess_image(blurred)
        prediction = model.predict(input_tensor)
        deblurred = postprocess_image(prediction)
        
        # 结果保存
        deblurred.save('deblurred_result.jpg')
        print("已保存去模糊结果: deblurred_result.jpg")

        # 可视化
        display_results(
            ('Original Image', Image.fromarray(resized_original)),
            ('Blurred Image', Image.fromarray(blurred)),
            ('Deblurred Image', deblurred)
        )

    except Exception as e:
        print(f"程序异常: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()