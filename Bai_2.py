import numpy as np
import tkinter as tk
from tkinter import filedialog
import os


I = np.array([
    [0,   0,   0,   0, 200, 200,   0,   0,   0,   0],
    [0,   0,   0,   0, 200, 200,   0,   0,   0,   0],
    [0,   0,   0, 200, 200, 200, 200,   0,   0,   0],
    [0,   0, 200, 200, 200, 200, 200, 200,   0,   0],
    [200, 200, 200, 200, 200, 200, 200, 200, 200,   0],
    [200, 200, 200, 200, 200, 200, 200, 200, 200,   0],
    [0,   0, 200, 200, 200, 200, 200, 200,   0,   0],
    [0,   0,   0, 200, 200, 200, 200, 200,   0,   0],
    [0,   0,   0,   0, 200, 200,   0,   0,   0,   0],
    [0,   0,   0,   0, 200, 200,   0,   0,   0,   0]
], dtype=np.float64)


def convolution(image, kernel):

    height, width = image.shape
    k = kernel.shape[0]
    pad = k // 2

    padded = np.pad(
        image,
        ((pad, pad), (pad, pad)),
        mode="reflect"
    )

    result = np.zeros(
        (height, width),
        dtype=np.float64
    )

    for i in range(height):
        for j in range(width):

            window = padded[
                i:i + k,
                j:j + k
            ]

            result[i, j] = np.sum(
                window * kernel
            )

    return result


def sobel_filter(image):

    Gx_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    Gy_kernel = np.array([
        [-1, -2, -1],
        [0,   0,  0],
        [1,   2,  1]
    ], dtype=np.float64)

    Gx = convolution(image, Gx_kernel)
    Gy = convolution(image, Gy_kernel)

    magnitude = np.sqrt(
        Gx ** 2 + Gy ** 2
    )

    return Gx, Gy, magnitude


def laplacian_filter(image):

    kernel = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float64)

    result = convolution(
        image,
        kernel
    )

    return result


def create_gaussian_kernel(
    size=5,
    sigma=1.0
):

    kernel = np.zeros(
        (size, size),
        dtype=np.float64
    )

    center = size // 2

    for i in range(size):
        for j in range(size):

            x = i - center
            y = j - center

            kernel[i, j] = np.exp(
                -(x*x + y*y) /
                (2 * sigma*sigma)
            )

    kernel /= np.sum(kernel)

    return kernel


def non_maximum_suppression(
    magnitude,
    angle
):

    height, width = magnitude.shape

    result = np.zeros_like(
        magnitude
    )

    angle = angle % 180

    for i in range(1, height - 1):
        for j in range(1, width - 1):

            direction = angle[i, j]

            if (
                direction < 22.5 or
                direction >= 157.5
            ):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]

            elif direction < 67.5:
                q = magnitude[i - 1, j + 1]
                r = magnitude[i + 1, j - 1]

            elif direction < 112.5:
                q = magnitude[i - 1, j]
                r = magnitude[i + 1, j]

            else:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if (
                magnitude[i, j] >= q and
                magnitude[i, j] >= r
            ):
                result[i, j] = magnitude[i, j]

    return result


# =========================================================
# DOUBLE THRESHOLD
# =========================================================
def double_threshold(
    image,
    low_threshold,
    high_threshold
):

    strong = 255
    weak = 75

    result = np.zeros_like(
        image,
        dtype=np.uint8
    )

    result[image >= high_threshold] = strong

    result[
        (image >= low_threshold) &
        (image < high_threshold)
    ] = weak

    return result


# =========================================================
# HYSTERESIS
# =========================================================
def hysteresis(image):

    strong = 255
    weak = 75

    result = image.copy()

    height, width = result.shape

    for i in range(1, height - 1):
        for j in range(1, width - 1):

            if result[i, j] == weak:

                neighborhood = result[
                    i - 1:i + 2,
                    j - 1:j + 2
                ]

                if np.any(
                    neighborhood == strong
                ):
                    result[i, j] = strong
                else:
                    result[i, j] = 0

    return result


# =========================================================
# CANNY
# =========================================================
def canny_filter(image):

    # ----- 1. Gaussian -----
    kernel = create_gaussian_kernel(
        size=5,
        sigma=1.0
    )

    blurred = convolution(
        image,
        kernel
    )

    # ----- 2. Sobel -----
    Gx_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    Gy_kernel = np.array([
        [-1, -2, -1],
        [0,   0,  0],
        [1,   2,  1]
    ], dtype=np.float64)

    Gx = convolution(
        blurred,
        Gx_kernel
    )

    Gy = convolution(
        blurred,
        Gy_kernel
    )

    # ----- 3. Gradient magnitude -----
    magnitude = np.sqrt(
        Gx ** 2 + Gy ** 2
    )

    # ----- 4. Gradient angle -----
    angle = np.arctan2(
        Gy,
        Gx
    ) * 180 / np.pi

    # ----- 5. Non-Maximum Suppression -----
    nms = non_maximum_suppression(
        magnitude,
        angle
    )

    # ----- 6. Double Threshold -----
    high = np.max(nms) * 0.30
    low = high * 0.50

    threshold = double_threshold(
        nms,
        low,
        high
    )

    # ----- 7. Hysteresis -----
    result = hysteresis(
        threshold
    )

    return result


# =========================================================
# HÀM LƯU MA TRẬN
# =========================================================
def save_matrix(
    matrix,
    folder,
    filename
):

    txt_path = os.path.join(
        folder,
        filename + ".txt"
    )

    csv_path = os.path.join(
        folder,
        filename + ".csv"
    )

    # Lưu TXT
    np.savetxt(
        txt_path,
        matrix,
        fmt="%.2f"
    )

    # Lưu CSV
    np.savetxt(
        csv_path,
        matrix,
        delimiter=",",
        fmt="%.2f"
    )


# =========================================================
# CHỌN THƯ MỤC
# =========================================================
root = tk.Tk()
root.withdraw()

output_folder = filedialog.askdirectory(
    title="Chọn thư mục lưu ma trận kết quả"
)

if not output_folder:
    print("Bạn chưa chọn thư mục!")
    exit()


# =========================================================
# CHẠY SOBEL
# =========================================================
Gx, Gy, Sobel = sobel_filter(I)


# =========================================================
# CHẠY LAPLACIAN
# =========================================================
Laplacian = laplacian_filter(I)


# =========================================================
# CHẠY CANNY
# =========================================================
Canny = canny_filter(I)


# =========================================================
# LƯU CÁC MA TRẬN
# =========================================================

# Ma trận ban đầu
save_matrix(
    I,
    output_folder,
    "input_matrix"
)

# Sobel
save_matrix(
    Gx,
    output_folder,
    "sobel_Gx"
)

save_matrix(
    Gy,
    output_folder,
    "sobel_Gy"
)

save_matrix(
    Sobel,
    output_folder,
    "sobel_magnitude"
)

# Laplacian
save_matrix(
    Laplacian,
    output_folder,
    "laplacian"
)

# Canny
save_matrix(
    Canny,
    output_folder,
    "canny"
)


# =========================================================
# IN RA MÀN HÌNH
# =========================================================

print("\n==============================")
print("       MA TRẬN BAN ĐẦU")
print("==============================")
print(I)


print("\n==============================")
print("          SOBEL Gx")
print("==============================")
print(Gx)


print("\n==============================")
print("          SOBEL Gy")
print("==============================")
print(Gy)


print("\n==============================")
print("       SOBEL MAGNITUDE")
print("==============================")
print(Sobel)


print("\n==============================")
print("         LAPLACIAN")
print("==============================")
print(Laplacian)


print("\n==============================")
print("            CANNY")
print("==============================")
print(Canny)


print("\n==============================")
print("Đã lưu ma trận tại:")
print(output_folder)
print("==============================")