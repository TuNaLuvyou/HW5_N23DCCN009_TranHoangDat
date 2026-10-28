# -*- coding: utf-8 -*-
"""
BÀI 1(b) - LỌC TRUNG BÌNH 7x7 BẰNG PHÉP NHÂN ĐIỂM TRONG MIỀN DFT (EXAMPLE 3)
MSSV: N23DCCN009
Họ và tên: Trần Hoàng Đạt
Lớp: D23CQCNPM01-N
Môn học: Xử lý ảnh số (INT 13146)

Yêu cầu bài toán (theo hw5.pdf):
- Thực hiện lọc tuyến tính (linear convolution) thông qua biến đổi DFT 2 chiều theo Example 3.
- Ma trận đáp ứng xung H kích thước 128x128, gồm cửa sổ 7x7 giá trị 1/49 tại tâm (64, 64).
- Kích thước zero-padding là 384x384.
- Nhân điểm trong miền tần số: Y_pad = X_pad * H_pad, sau đó IFFT và trích xuất vùng [64:320, 64:320].
- Xuất 8 ảnh theo đúng yêu cầu đề bài:
  1. Ảnh gốc ban đầu (original input).
  2. Ảnh gốc zero-padded 384x384.
  3. Đáp ứng xung zero-padded 384x384.
  4. Phổ log-magnitude căn giữa của ảnh vào.
  5. Phổ log-magnitude căn giữa của đáp ứng xung.
  6. Phổ log-magnitude căn giữa của ảnh ra.
  7. Ảnh ra zero-padded 384x384.
  8. Ảnh kết quả cuối cùng kích thước 256x256.
- So sánh sai khác pixel với Bài 1(a): max difference from part (a) = 0.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Cấu hình encoding console tiếng Việt trên Windows
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Đường dẫn thư mục
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
OUT_DIR = BASE_DIR / 'output' / 'Bai_1'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N = 256  # Kích thước ảnh 256x256


def read_bin(filename: str) -> np.ndarray:
    """Đọc ảnh raw .bin 256x256 uint8 -> ndarray (256, 256) kiểu float64."""
    path = DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {path}")
    data = np.fromfile(path, dtype=np.uint8)
    if data.size != N * N:
        raise ValueError(f"Kích thước file {path} không đúng {N*N} byte: {data.size}")
    return data.reshape((N, N)).astype(np.float64)


def full_scale(image: np.ndarray) -> np.ndarray:
    """Full-scale contrast stretch về khoảng [0, 255] và làm tròn số nguyên gần nhất."""
    a = np.asarray(image, dtype=np.float64)
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi == lo:
        return np.zeros(a.shape, dtype=np.uint8)
    normalized = np.clip((a - lo) * 255.0 / (hi - lo), 0.0, 255.0)
    return np.floor(np.round(normalized, 10) + 0.5).astype(np.uint8)


def save_gray(path: Path, image: np.ndarray) -> None:
    """Lưu ảnh xám PNG với full-scale contrast."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(full_scale(image)).save(path)


def spectrum_log(fft_image: np.ndarray) -> np.ndarray:
    """Phổ biên độ dạng log đã căn giữa: log(1 + |fftshift(FFT)|)."""
    return np.log1p(np.abs(np.fft.fftshift(fft_image)))


def loc_trung_binh_7x7_khong_gian(x: np.ndarray) -> np.ndarray:
    """Lọc 1(a) dùng để đối chiếu xác minh ngay trong file."""
    pad = np.pad(x, pad_width=3, mode='constant', constant_values=0)
    out = np.zeros((N, N), dtype=np.float64)
    for r in range(7):
        for c in range(7):
            out += pad[r:r + N, c:c + N]
    return out / 49.0


def loc_trung_binh_7x7_dft_384(x: np.ndarray) -> tuple[np.ndarray, dict]:
    """Lọc tích chập tuyến tính qua DFT 384x384 theo Example 3."""
    h = np.zeros((128, 128), dtype=np.float64)
    h[61:68, 61:68] = 1.0 / 49.0  # Tâm tại (64, 64)
    xpad = np.zeros((384, 384), dtype=np.float64)
    hpad = np.zeros((384, 384), dtype=np.float64)
    xpad[:N, :N] = x
    hpad[:128, :128] = h

    fx = np.fft.fft2(xpad)
    fh = np.fft.fft2(hpad)
    fy = fx * fh
    ypad = np.fft.ifft2(fy).real
    result = ypad[64:320, 64:320]

    return result, {
        'xpad': xpad,
        'hpad': hpad,
        'fx': fx,
        'fh': fh,
        'fy': fy,
        'ypad': ypad
    }


def main():
    print("=" * 65)
    print("BÀI 1(b): LỌC TRUNG BÌNH 7x7 QUA PHÉP NHÂN DFT (384x384)")
    print("Sinh viên: Trần Hoàng Đạt - MSSV: N23DCCN009")
    print("=" * 65)

    # 1. Đọc dữ liệu ảnh
    print("[1] Đọc dữ liệu ảnh salesman.bin...")
    img_in = read_bin('salesman.bin')

    # 2. Thực hiện lọc DFT 384x384
    print("[2] Biến đổi FFT2, nhân phổ và biến đổi IFFT2...")
    img_out_b, details = loc_trung_binh_7x7_dft_384(img_in)

    # 3. Xuất 8 ảnh theo đúng yêu cầu đề bài
    print("[3] Lưu 8 ảnh kết quả vào output/Bai_1/:")
    items = [
        ('1b_anh_goc.png', img_in, '1. Original input image'),
        ('1b_anh_padding_384.png', details['xpad'], '2. Zero padded original image'),
        ('1b_kernel_padding_384.png', details['hpad'], '3. Zero padded impulse response'),
        ('1b_pho_log_input.png', spectrum_log(details['fx']), '4. Centered DFT log-spectrum input'),
        ('1b_pho_log_kernel.png', spectrum_log(details['fh']), '5. Centered DFT log-spectrum kernel'),
        ('1b_pho_log_output.png', spectrum_log(details['fy']), '6. Centered DFT log-spectrum output'),
        ('1b_output_padding_384.png', details['ypad'], '7. Zero padded output image'),
        ('1b_ket_qua.png', img_out_b, '8. Final 256x256 output image')
    ]

    for fname, img_data, desc in items:
        p = OUT_DIR / fname
        save_gray(p, img_data)
        print(f"    - {desc:<45} -> {fname}")

    # 4. Kiểm tra so sánh với Bài 1(a)
    print("\n[4] Xác thực kết quả với Bài 1(a):")
    img_out_a = loc_trung_binh_7x7_khong_gian(img_in)
    Y1a = full_scale(img_out_a)
    Y1b = full_scale(img_out_b)
    diff_pixel = int(np.max(np.abs(Y1b.astype(np.int16) - Y1a.astype(np.int16))))
    diff_float = float(np.max(np.abs(img_out_b - img_out_a)))

    print(f"    disp: (b): max difference from part (a): {diff_pixel}")
    print(f"    - Sai khác pixel lớn nhất: {diff_pixel}")
    print(f"    - Sai khác float lớn nhất: {diff_float:.3e}")
    assert diff_pixel == 0, "LỖI: Kết quả không trùng khớp với bài 1(a)!"
    print("    => Kết quả trùng khớp 100% với Bài 1(a)!")

    # 5. Hiển thị hình ảnh trực quan 8 ảnh bằng Matplotlib (lưới 2x4)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('BÀI 1(b) - LỌC TUYẾN TÍNH DFT 384x384 (N23DCCN009 - Trần Hoàng Đạt)', fontsize=13, fontweight='bold')

    for idx, (fname, img_data, desc) in enumerate(items):
        r, c = idx // 4, idx % 4
        axes[r, c].imshow(full_scale(img_data), cmap='gray', vmin=0, vmax=255)
        axes[r, c].set_title(desc, fontsize=9)
        axes[r, c].axis('off')

    plt.tight_layout()
    fig_path = OUT_DIR / 'bai1_b_figure.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"    - Đã lưu đồ thị hiển thị: {fig_path.name}")
    print("=" * 65)
    print("HOÀN THÀNH BÀI 1(b)!\n")

    if '--no-gui' not in sys.argv:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
