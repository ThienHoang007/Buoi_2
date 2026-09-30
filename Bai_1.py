import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os


# =========================================================
# 1. MEAN FILTER
# =========================================================
def mean_filter(image, kernel_size=3):
    pad = kernel_size // 2

    padded = np.pad(
        image,
        ((pad, pad), (pad, pad), (0, 0)),
        mode='reflect'
    )

    result = np.zeros_like(image, dtype=np.float32)

    height, width, channels = image.shape

    for i in range(height):
        for j in range(width):
            for c in range(channels):

                window = padded[
                    i:i + kernel_size,
                    j:j + kernel_size,
                    c
                ]

                result[i, j, c] = np.mean(window)

    return np.clip(result, 0, 255).astype(np.uint8)


# =========================================================
# 2. MEDIAN FILTER
# =========================================================
def median_filter(image, kernel_size=3):
    pad = kernel_size // 2

    padded = np.pad(
        image,
        ((pad, pad), (pad, pad), (0, 0)),
        mode='reflect'
    )

    result = np.zeros_like(image, dtype=np.uint8)

    height, width, channels = image.shape

    for i in range(height):
        for j in range(width):
            for c in range(channels):

                window = padded[
                    i:i + kernel_size,
                    j:j + kernel_size,
                    c
                ]

                values = window.flatten()
                values.sort()

                middle = len(values) // 2

                result[i, j, c] = values[middle]

    return result


# =========================================================
# 3. TẠO GAUSSIAN KERNEL
# =========================================================
def create_gaussian_kernel(kernel_size=5, sigma=1.0):

    kernel = np.zeros(
        (kernel_size, kernel_size),
        dtype=np.float32
    )

    center = kernel_size // 2

    for i in range(kernel_size):
        for j in range(kernel_size):

            x = i - center
            y = j - center

            kernel[i, j] = np.exp(
                -(x * x + y * y) /
                (2 * sigma * sigma)
            )

    # Chuẩn hóa kernel
    kernel = kernel / np.sum(kernel)

    return kernel


# =========================================================
# 4. GAUSSIAN FILTER
# =========================================================
def gaussian_filter(image, kernel_size=5, sigma=1.0):

    kernel = create_gaussian_kernel(
        kernel_size,
        sigma
    )

    pad = kernel_size // 2

    padded = np.pad(
        image,
        ((pad, pad), (pad, pad), (0, 0)),
        mode='reflect'
    )

    result = np.zeros_like(
        image,
        dtype=np.float32
    )

    height, width, channels = image.shape

    for i in range(height):
        for j in range(width):
            for c in range(channels):

                window = padded[
                    i:i + kernel_size,
                    j:j + kernel_size,
                    c
                ]

                result[i, j, c] = np.sum(
                    window * kernel
                )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


# =========================================================
# 5. CHỌN ẢNH
# =========================================================
root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Chọn ảnh cần xử lý",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
        ("All files", "*.*")
    ]
)

# Nếu người dùng không chọn ảnh
if not image_path:
    print("Bạn chưa chọn ảnh!")
    exit()


# =========================================================
# 6. ĐỌC ẢNH
# =========================================================
image = cv2.imread(image_path)

if image is None:
    print("Không thể đọc ảnh!")
    exit()


# =========================================================
# 7. CHỌN THƯ MỤC LƯU
# =========================================================
output_folder = filedialog.askdirectory(
    title="Chọn thư mục lưu kết quả"
)

# Nếu người dùng không chọn thư mục
if not output_folder:
    print("Bạn chưa chọn thư mục lưu!")
    exit()


# =========================================================
# 8. THỰC HIỆN CÁC BỘ LỌC
# =========================================================
print("Đang xử lý...")

mean_result = mean_filter(
    image,
    kernel_size=3
)

median_result = median_filter(
    image,
    kernel_size=3
)

gaussian_result = gaussian_filter(
    image,
    kernel_size=5,
    sigma=1.0
)


# =========================================================
# 9. TẠO ĐƯỜNG DẪN LƯU
# =========================================================
mean_path = os.path.join(
    output_folder,
    "mean_result.jpg"
)

median_path = os.path.join(
    output_folder,
    "median_result.jpg"
)

gaussian_path = os.path.join(
    output_folder,
    "gaussian_result.jpg"
)


# =========================================================
# 10. LƯU ẢNH
# =========================================================
cv2.imwrite(
    mean_path,
    mean_result
)

cv2.imwrite(
    median_path,
    median_result
)

cv2.imwrite(
    gaussian_path,
    gaussian_result
)


# =========================================================
# 11. HIỂN THỊ KẾT QUẢ
# =========================================================
cv2.imshow(
    "Original",
    image
)

cv2.imshow(
    "Mean Filter",
    mean_result
)

cv2.imshow(
    "Median Filter",
    median_result
)

cv2.imshow(
    "Gaussian Filter",
    gaussian_result
)

print("Đã xử lý xong!")
print("Kết quả được lưu tại:")
print(output_folder)

print("\nCác file:")
print(mean_path)
print(median_path)
print(gaussian_path)

cv2.waitKey(0)
cv2.destroyAllWindows()