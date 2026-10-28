# -*- coding: utf-8 -*-
"""
BÀI 1(a) - LỌC TRUNG BÌNH 7x7 TRONG MIỀN KHÔNG GIAN (IMAGE DOMAIN CONVOLUTION)
MSSV: N23DCCN009
Họ và tên: Trần Hoàng Đạt
Lớp: D23CQCNPM01-N
Môn học: Xử lý ảnh số (INT 13146)

Yêu cầu bài toán (theo hw5.pdf):
- Đọc ảnh salesman.bin (raw grayscale 256x256, 8-bit).
- Thực hiện bộ lọc trung bình 7x7 trực tiếp trong miền không gian (LTI filter, mỗi pixel có trọng số 1/49).
- Xử lý hiệu ứng biên bằng zero-padding: đệm 3 hàng/cột số 0 ở 4 cạnh tạo ảnh 262x262.
- Hiển thị và lưu ảnh gốc cùng ảnh kết quả với độ tương phản toàn dải (full scale contrast).
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
    """Lọc trung bình 7x7 miền không gian với zero padding 3 pixel biên (tạo ma trận 262x262)."""
    pad = np.pad(x, pad_width=3, mode='constant', constant_values=0)
    out = np.zeros((N, N), dtype=np.float64)
    for r in range(7):
        for c in range(7):
            out += pad[r:r + N, c:c + N]
    return out / 49.0


def main():
    print("=" * 65)
    print("BÀI 1(a): LỌC TRUNG BÌNH 7x7 TRONG MIỀN KHÔNG GIAN")
    print("Sinh viên: Trần Hoàng Đạt - MSSV: N23DCCN009")
    print("=" * 65)

    # 1. Đọc ảnh salesman.bin
    print("[1] Đọc dữ liệu ảnh salesman.bin...")
    img_in = read_bin('salesman.bin')
    print(f"    - Kích thước ảnh gốc: {img_in.shape}")
    print(f"    - Khoảng mức xám: [{img_in.min():.1f}, {img_in.max():.1f}]")

    # 2. Thực hiện lọc trong miền không gian
    print("[2] Lọc trung bình 7x7 với zero padding 3 pixel biên (262x262)...")
    img_out = loc_trung_binh_7x7_khong_gian(img_in)
    print(f"    - Kích thước ảnh sau lọc: {img_out.shape}")
    print(f"    - Khoảng giá trị sau lọc: [{img_out.min():.2f}, {img_out.max():.2f}]")

    # 3. Lưu ảnh riêng lẻ vào output/Bai_1/
    path_goc = OUT_DIR / '1a_anh_goc.png'
    path_kq = OUT_DIR / '1a_ket_qua.png'
    save_gray(path_goc, img_in)
    save_gray(path_kq, img_out)
    print(f"[3] Đã lưu ảnh kết quả vào thư mục: {OUT_DIR}")
    print(f"    - {path_goc.name}")
    print(f"    - {path_kq.name}")

    # 4. Hiển thị hình ảnh trực quan bằng Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('BÀI 1(a) - LỌC TRUNG BÌNH 7x7 MIỀN KHÔNG GIAN (N23DCCN009 - Trần Hoàng Đạt)', fontsize=12, fontweight='bold')

    axes[0].imshow(full_scale(img_in), cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Ảnh gốc salesman (256x256)', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(full_scale(img_out), cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Ảnh kết quả lọc trung bình 7x7', fontsize=11)
    axes[1].axis('off')

    plt.tight_layout()
    fig_path = OUT_DIR / 'bai1_a_figure.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"    - Đã lưu đồ thị hiển thị: {fig_path.name}")
    print("=" * 65)
    print("HOÀN THÀNH BÀI 1(a)!\n")

    if '--no-gui' not in sys.argv:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
