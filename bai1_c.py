# -*- coding: utf-8 -*-
"""
BÀI 1(c) - LỌC TRUNG BÌNH 7x7 BẰNG ĐÁP ỨNG XUNG ZERO-PHASE VÀ DFT (EXAMPLE 5)
MSSV: N23DCCN009
Họ và tên: Trần Hoàng Đạt
Lớp: D23CQCNPM01-N
Môn học: Xử lý ảnh số (INT 13146)

Yêu cầu bài toán (theo hw5.pdf):
- Thực hiện lọc tuyến tính (linear convolution) qua biến đổi DFT sử dụng đáp ứng xung zero-phase.
- Khởi tạo ma trận đáp ứng xung h kích thước 256x256 chứa hình vuông 7x7 giá trị 1/49 tại tâm (128, 128).
- Dùng np.fft.fftshift(h) để đưa tâm về gốc tọa độ (0, 0), tạo đáp ứng xung zero-phase h2.
- Zero-padding đáp ứng xung zero-phase lên 512x512 (đưa các phần vào 4 góc của ma trận 512x512).
- Zero-padding ảnh đầu vào lên 512x512: xpad[:256, :256] = x.
- Nhân điểm trong miền FFT 512x512, IFFT2 và lấy vùng [0:256, 0:256].
- Xuất 4 ảnh theo đúng yêu cầu đề bài:
  1. Ảnh gốc ban đầu (original input).
  2. Đáp ứng xung zero-phase 256x256 (h2).
  3. Đáp ứng xung zero-phase đã đệm zero lên 512x512 (h2ZP).
  4. Ảnh kết quả cuối cùng kích thước 256x256.
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


def loc_trung_binh_7x7_khong_gian(x: np.ndarray) -> np.ndarray:
    """Lọc 1(a) dùng để đối chiếu xác minh ngay trong file."""
    pad = np.pad(x, pad_width=3, mode='constant', constant_values=0)
    out = np.zeros((N, N), dtype=np.float64)
    for r in range(7):
        for c in range(7):
            out += pad[r:r + N, c:c + N]
    return out / 49.0


def pad_zero_phase_512(h_zero_phase: np.ndarray) -> np.ndarray:
    """Đệm zero cho đáp ứng xung zero-phase 256x256 lên 512x512 theo đúng 4 góc."""
    padded = np.zeros((512, 512), dtype=np.float64)
    padded[:128, :128] = h_zero_phase[:128, :128]
    padded[:128, 384:] = h_zero_phase[:128, 128:]
    padded[384:, :128] = h_zero_phase[128:, :128]
    padded[384:, 384:] = h_zero_phase[128:, 128:]
    return padded


def loc_trung_binh_7x7_zero_phase(x: np.ndarray) -> tuple[np.ndarray, dict]:
    """Lọc DFT zero-phase theo Example 5."""
    h = np.zeros((256, 256), dtype=np.float64)
    h[125:132, 125:132] = 1.0 / 49.0  # Cửa sổ 7x7 tâm tại (128, 128)

    h2 = np.fft.fftshift(h)  # Đưa tâm về gốc tọa độ (zero-phase)
    h2ZP = pad_zero_phase_512(h2)

    xpad = np.zeros((512, 512), dtype=np.float64)
    xpad[:N, :N] = x

    ypad = np.fft.ifft2(np.fft.fft2(xpad) * np.fft.fft2(h2ZP)).real
    result = ypad[:N, :N]

    return result, {
        'h2': h2,
        'h2ZP': h2ZP,
        'xpad': xpad,
        'ypad': ypad
    }


def main():
    print("=" * 65)
    print("BÀI 1(c): LỌC ZERO-PHASE IMPULSE RESPONSE VÀ DFT (512x512)")
    print("Sinh viên: Trần Hoàng Đạt - MSSV: N23DCCN009")
    print("=" * 65)

    # 1. Đọc dữ liệu ảnh
    print("[1] Đọc dữ liệu ảnh salesman.bin...")
    img_in = read_bin('salesman.bin')

    # 2. Thực hiện lọc zero-phase DFT 512x512
    print("[2] Khởi tạo kernel zero-phase, đệm zero 512x512 và tính FFT2...")
    img_out_c, details = loc_trung_binh_7x7_zero_phase(img_in)

    # 3. Xuất 4 ảnh theo đúng yêu cầu đề bài
    print("[3] Lưu 4 ảnh kết quả vào output/Bai_1/:")
    items = [
        ('1c_anh_goc.png', img_in, '1. Original input image'),
        ('1c_kernel_zero_phase_256.png', details['h2'], '2. 256x256 zero-phase impulse response (h2)'),
        ('1c_kernel_padding_512.png', details['h2ZP'], '3. 512x512 zero padded zero-phase (h2ZP)'),
        ('1c_ket_qua.png', img_out_c, '4. Final 256x256 output image')
    ]

    for fname, img_data, desc in items:
        p = OUT_DIR / fname
        save_gray(p, img_data)
        print(f"    - {desc:<45} -> {fname}")

    # 4. Kiểm tra so sánh với Bài 1(a)
    print("\n[4] Xác thực kết quả với Bài 1(a):")
    img_out_a = loc_trung_binh_7x7_khong_gian(img_in)
    Y1a = full_scale(img_out_a)
    Y1c = full_scale(img_out_c)
    diff_pixel = int(np.max(np.abs(Y1c.astype(np.int16) - Y1a.astype(np.int16))))
    diff_float = float(np.max(np.abs(img_out_c - img_out_a)))

    print(f"    disp: (c): max difference from part (a): {diff_pixel}")
    print(f"    - Sai khác pixel lớn nhất: {diff_pixel}")
    print(f"    - Sai khác float lớn nhất: {diff_float:.3e}")
    assert diff_pixel == 0, "LỖI: Kết quả không trùng khớp với bài 1(a)!"
    print("    => Kết quả trùng khớp 100% với Bài 1(a)!")

    # 5. Hiển thị hình ảnh trực quan 4 ảnh bằng Matplotlib (lưới 1x4)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('BÀI 1(c) - ZERO-PHASE DFT 512x512 (N23DCCN009 - Trần Hoàng Đạt)', fontsize=13, fontweight='bold')

    for idx, (fname, img_data, desc) in enumerate(items):
        axes[idx].imshow(full_scale(img_data), cmap='gray', vmin=0, vmax=255)
        axes[idx].set_title(desc, fontsize=9)
        axes[idx].axis('off')

    plt.tight_layout()
    fig_path = OUT_DIR / 'bai1_c_figure.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"    - Đã lưu đồ thị hiển thị: {fig_path.name}")
    print("=" * 65)
    print("HOÀN THÀNH BÀI 1(c)!\n")

    if '--no-gui' not in sys.argv:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
