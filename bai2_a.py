# -*- coding: utf-8 -*-
"""
BÀI 2(a) - ĐỌC VÀ HIỂN THỊ ẢNH GỐC VÀ CÁC ẢNH NHIỄU, TÍNH MSE BAN ĐẦU
MSSV: N23DCCN009
Họ và tên: Trần Hoàng Đạt
Lớp: D23CQCNPM01-N
Môn học: Xử lý ảnh số (INT 13146)

Yêu cầu bài toán (theo hw5.pdf):
- Đọc 3 file ảnh xám raw 256x256 8-bit:
  1. girl2.bin: Ảnh gốc (Tiffany).
  2. girl2Noise32Hi.bin: Ảnh bị nhiễu Gauss trắng tần số cao (hi-pass Gaussian white noise),
     với phổ nhiễu khác 0 khi sqrt(u^2 + v^2) > 64 cpi và bằng 0 khi <= 64 cpi.
  3. girl2Noise32.bin: Ảnh bị nhiễu Gauss trắng toàn dải (broadband Gaussian white noise).
- Hiển thị và lưu cả 3 ảnh với full-scale contrast.
- Tính sai số toàn phương trung bình (MSE) của 2 ảnh nhiễu so với ảnh gốc girl2:
  MSE(reference, noisy) = (1 / (N*N)) * sum((reference - noisy)^2)
- Đối chiếu với kết quả chuẩn trong đáp án:
  + Nhiễu cao tần (high-pass noise):   692.505020
  + Nhiễu toàn dải (broadband noise):  744.467911
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
OUT_DIR = BASE_DIR / 'output' / 'Bai_2'
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


def mse(reference: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Squared Error (MSE) tính trên dữ liệu ảnh số thực float64."""
    return float(np.mean((reference - predicted) ** 2))


def main():
    print("=" * 65)
    print("BÀI 2(a): ĐỌC 3 ẢNH ĐẦU VÀO VÀ TÍNH MSE TRƯỚC KHI LỌC")
    print("Sinh viên: Trần Hoàng Đạt - MSSV: N23DCCN009")
    print("=" * 65)

    # 1. Đọc 3 ảnh đầu vào
    print("[1] Đọc dữ liệu 3 file ảnh:")
    girl2 = read_bin('girl2.bin')
    noise_hi = read_bin('girl2Noise32Hi.bin')
    noise_bb = read_bin('girl2Noise32.bin')

    print(f"    - girl2.bin:           kích thước {girl2.shape}, min={girl2.min():.1f}, max={girl2.max():.1f}")
    print(f"    - girl2Noise32Hi.bin: kích thước {noise_hi.shape}, min={noise_hi.min():.1f}, max={noise_hi.max():.1f}")
    print(f"    - girl2Noise32.bin:   kích thước {noise_bb.shape}, min={noise_bb.min():.1f}, max={noise_bb.max():.1f}")

    # 2. Lưu 3 ảnh riêng lẻ
    path_orig = OUT_DIR / '2a_original.png'
    path_hi = OUT_DIR / '2a_high_pass.png'
    path_bb = OUT_DIR / '2a_broadband.png'

    save_gray(path_orig, girl2)
    save_gray(path_hi, noise_hi)
    save_gray(path_bb, noise_bb)

    print("\n[2] Đã lưu 3 ảnh vào thư mục output/Bai_2/:")
    print(f"    - Ảnh gốc Tiffany:        {path_orig.name}")
    print(f"    - Ảnh nhiễu cao tần:       {path_hi.name}")
    print(f"    - Ảnh nhiễu toàn dải:      {path_bb.name}")

    # 3. Tính MSE của các ảnh nhiễu so với ảnh gốc girl2 (trên float64 gốc)
    mse_hi = mse(girl2, noise_hi)
    mse_bb = mse(girl2, noise_bb)

    print("\n[3] KẾT QUẢ ĐO MSE BAN ĐẦU (SO VỚI ẢNH GỐC GIRL2):")
    print("-" * 65)
    print(f"{'ẢNH ĐẦU VÀO':<28}{'ĐẶC ĐIỂM NHIỄU':<22}{'MSE BAN ĐẦU':>15}")
    print("-" * 65)
    print(f"{'girl2Noise32Hi (Hi-pass)':<28}{'sqrt(u^2+v^2) > 64':<22}{mse_hi:>15.6f}")
    print(f"{'girl2Noise32 (Broadband)':<28}{'Toàn dải tần số':<22}{mse_bb:>15.6f}")
    print("-" * 65)
    print(f"(*) Đối chiếu đáp án chuẩn:")
    print(f"    - MSE nhiễu cao tần  = {mse_hi:.6f} (Khớp chuẩn: 692.505020)")
    print(f"    - MSE nhiễu toàn dải = {mse_bb:.6f} (Khớp chuẩn: 744.467911)")

    # 4. Hiển thị hình ảnh trực quan 3 ảnh bằng Matplotlib (lưới 1x3)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('BÀI 2(a) - 3 ẢNH ĐẦU VÀO & MSE BAN ĐẦU (N23DCCN009 - Trần Hoàng Đạt)', fontsize=12, fontweight='bold')

    axes[0].imshow(full_scale(girl2), cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Ảnh chuẩn girl2 (Tiffany)', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(full_scale(noise_hi), cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'Nhiễu cao tần (Hi-pass)\nMSE = {mse_hi:.4f}', fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(full_scale(noise_bb), cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f'Nhiễu toàn dải (Broadband)\nMSE = {mse_bb:.4f}', fontsize=11)
    axes[2].axis('off')

    plt.tight_layout()
    fig_path = OUT_DIR / 'bai2_a_figure.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"    - Đã lưu đồ thị hiển thị: {fig_path.name}")
    print("=" * 65)
    print("HOÀN THÀNH BÀI 2(a)!\n")

    if '--no-gui' not in sys.argv:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
