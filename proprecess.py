
from sklearn.cluster import KMeans
import pywt
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
def extract_region(image, center, size):
    """
    从图像中提取一个区域。
    :param image: 输入图像
    :param center: 区域中心坐标 (y, x)
    :param size: 区域的宽高 (height, width)
    :return: 截取的区域图像
    """
    y, x = center
    h, w = size
    start_x = max(x - w // 2, 0)
    start_y = max(y - h // 2, 0)
    end_x = min(x + w // 2, image.shape[1])
    end_y = min(y + h // 2, image.shape[0])
    new_w = end_x-start_x
    new_h = end_y-start_y
    # print(new_h)
    if new_w % 2 != 0:
        if new_w<size[0] and start_x==0:
            end_x = end_x - 1
        if new_w < size[0] and end_x == image.shape[1]:
            start_x = start_x+1
    if new_h % 2 != 0:
        if new_h<size[1] and start_y==0:
            end_y = end_y - 1
        if new_h < size[1] and end_y == image.shape[0]:
            start_y = start_y + 1

    return image[start_y:end_y ,start_x:end_x],start_x,start_y,end_x,end_y

def guided_filter(I, p, r, eps):
    """
    引导滤波
    :param I: 引导图像
    :param p: 输入图像
    :param r: 半径
    :param eps: 正则化参数
    :return: 滤波后的图像
    """
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))

    q = mean_a * I + mean_b
    return q

def rolling_guidance_filter(image, r, eps, iterations):
    """
    滚动导向滤波
    :param image: 输入图像
    :param r: 半径
    :param eps: 正则化参数
    :param iterations: 迭代次数
    :return: 滤波后的图像
    """
    I = image
    for i in range(iterations):
        I = guided_filter(I, image, r, eps)
    return I
def calculate_entropy(data):
    # 计算数据的熵
    hist, _ = np.histogram(data, bins=256, range=(0, 256))
    hist = hist / hist.sum()  # 归一化
    hist = hist[hist > 0]  # 只保留非零的概率
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


def calculate_spatial_entropy(image):
    # 提取255像素的坐标
    coordinates = np.column_stack(np.where(image == 255))

    if len(coordinates) == 0:
        return 0.0  # 如果没有255像素区域，熵为0

    # 计算坐标的熵
    x_coords = coordinates[:, 1]  # x坐标
    y_coords = coordinates[:, 0]  # y坐标

    # 计算x和y坐标的熵
    x_entropy = calculate_entropy(x_coords)
    y_entropy = calculate_entropy(y_coords)

    # 总熵为x坐标熵和y坐标熵的平均值
    spatial_entropy = (x_entropy + y_entropy) / 2
    return spatial_entropy






def canny_detect(image,seed_point):
    # Step 3: 使用Canny边缘检测算法检测边缘
    edges = cv2.Canny(image, threshold1=50, threshold2=150)

    # Step 4: 边缘链成封闭区域
    # 使用 findContours 方法查找边缘
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Step 5: 检查是否存在闭环区域
    has_closed_loop = False

    for contour in contours:
        # 计算轮廓的周长
        perimeter = cv2.arcLength(contour, True)
        # 计算轮廓的面积
        area = cv2.contourArea(contour)

        # 使用面积和周长检查轮廓是否为闭环
        if perimeter > 0 and area > 0:
            # 如果轮廓有面积且周长大于0，基本上可以认为是闭环
            has_closed_loop = True
            break
    if not has_closed_loop:
        return np.zeros_like(image)

    # Step 5: 生成二值掩码
    # 创建一个空白的掩码图像
    canny_mask = np.zeros_like(image)

    # 将封闭区域内部的像素设置为前景（255），其他设置为背景（0）
    cv2.drawContours(canny_mask, contours, -1, color=255, thickness=cv2.FILLED)
    difference = cv2.absdiff(edges, canny_mask)

    # 计算图像差异的非零像素数
    non_zero_count = np.count_nonzero(difference)

    if non_zero_count == 0 and has_closed_loop !=0 :
        return np.zeros_like(image)
    canny_mask =Cluster_localization(canny_mask,seed_point=seed_point)
    return canny_mask

def region_growing(image, seed, threshold=10):
    """
    Perform region growing segmentation starting from a seed point.

    :param image: Grayscale image to segment
    :param seed: Seed point (y, x) from which to start growing
    :param threshold: Intensity difference threshold for region growing
    :return: Binary segmented image
    """
    height, width = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    x_seed, y_seed = seed
    seed_intensity = image[y_seed, x_seed]

    # Initialize a list of pixels to check
    pixels_to_check = [(y_seed, x_seed)]
    segmented[y_seed, x_seed] = 255

    while pixels_to_check:
        y, x = pixels_to_check.pop(0)

        # Check neighboring pixels
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx

            if 0 <= ny < height and 0 <= nx < width:
                if segmented[ny, nx] == 0 and abs(int(image[ny, nx]) - seed_intensity) < threshold:
                    segmented[ny, nx] = 255
                    pixels_to_check.append((ny, nx))

    return segmented
# 读取灰度图像
def Cluster_localization(segmented_image, seed_point):
    # 使用connectedComponents函数来进行连通区域标记
    num_labels, labels_im = cv2.connectedComponents(segmented_image)

    # 给定点坐标
    x, y = seed_point[0], seed_point[1]

    # 获取该点所在的簇标签
    target_label = labels_im[y, x]

    # 创建一个掩膜，只保留目标簇
    mask = np.zeros_like(segmented_image)
    mask[labels_im == target_label] = 255

    # 应用掩膜
    result = cv2.bitwise_and(segmented_image, segmented_image, mask=mask)
    return result
def proprecess(image,seed_point):
    # 定义滚动导向滤波参数
    r = 8  # 半径
    eps = 0.01  # 正则化参数
    iterations = 4  # 迭代次数
    # 应用滚动导向滤波
    rgf_image = rolling_guidance_filter(image, r, eps, iterations)  #滚动引导滤波

    reconstructed_image = cv2.convertScaleAbs(rgf_image)

    # 执行区域生长分割
    segmented_image1 = region_growing(reconstructed_image, seed_point, threshold=20)
    kernel = np.ones((5, 5), np.uint8)  # 定义结构元素
    filled_image = cv2.morphologyEx(segmented_image1, cv2.MORPH_CLOSE, kernel)  #填充

    canny_mask = canny_detect(reconstructed_image,seed_point=seed_point)

    # 图像相加
    added_image = canny_mask + filled_image

    # 将像素值大于1的设置为1
    result_image = np.where(added_image > 1, 255, added_image)
    # return rgf_image,reconstructed_image,filled_image,canny_mask,result_image
    return result_image


def MDA(image,mask,size=(40,40)):
    result_image = np.zeros_like(mask)  # mask 0 255
    unique_values = np.unique(mask)
    if len(unique_values) == 1:
        return result_image
    # 使用连通组件分析找到所有簇
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)


    # # 绘制每个簇的中心
    for i in range(1, num_labels):  # 从1开始，跳过背景
        center = centroids[i]
        center = (int(center[1]), int(center[0]))  # y x
        region_image, start_x, start_y, end_x, end_y = extract_region(image, center, size)


        output_region_image, _, _, _, _ = extract_region(mask, center, size)

        result_region_image = proprecess(region_image, seed_point=(center[1] - start_x, center[0] - start_y))

        result_image[start_y:end_y, start_x:end_x] = result_region_image

    return result_image