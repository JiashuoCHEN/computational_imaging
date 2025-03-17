import cv2
import numpy as np

def apply_filters(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not open or find the image.")
        return
    
    # 显示原始图像
    cv2.imshow('Original Image', image)

    # 平均核滤波
    kernel_size = (5, 5)  # 核大小可以根据需要调整
    blurred_avg = cv2.blur(image, kernel_size)
    cv2.imshow('Average Blur', blurred_avg)
    cv2.imwrite('average_blur.jpg', blurred_avg)

    # 高斯核滤波
    sigmaX = 0  # 标准差可以根据需要调整
    blurred_gaussian = cv2.GaussianBlur(image, kernel_size, sigmaX)
    cv2.imshow('Gaussian Blur', blurred_gaussian)
    cv2.imwrite('gaussian_blur.jpg', blurred_gaussian)

    # 运动模糊
    size = 15  # 模糊条带的长度
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    blurred_motion = cv2.filter2D(image, -1, kernel_motion_blur)
    cv2.imshow('Motion Blur', blurred_motion)
    cv2.imwrite('motion_blur.jpg', blurred_motion)

    # 中值滤波
    median_kernel_size = 5  # 内核大小必须是正奇数
    blurred_median = cv2.medianBlur(image, median_kernel_size)
    cv2.imshow('Median Blur', blurred_median)
    cv2.imwrite('median_blur.jpg', blurred_median)

    # 双边滤波
    d = 9       # 每个像素邻域直径
    sigmaColor = 75   # 色彩空间标准差
    sigmaSpace = 75   # 坐标空间标准差
    blurred_bilateral = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
    cv2.imshow('Bilateral Blur', blurred_bilateral)
    cv2.imwrite('bilateral_blur.jpg', blurred_bilateral)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用示例
apply_filters('test.jpg')