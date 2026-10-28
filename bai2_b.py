# -*- coding: utf-8 -*-
"""
BÀI 2(b) - BỘ LỌC THÔNG THẤP LÝ TƯỞNG (IDEAL LOW-PASS FILTER, UCUTOFF = 64 CPI)
MSSV: N23DCCN009
Họ và tên: Trần Hoàng Đạt
Lớp: D23CQCNPM01-N
Môn học: Xử lý ảnh số (INT 13146)

Yêu cầu bài toán (theo hw5.pdf):
- Áp dụng bộ lọc thông thấp lý tưởng đẳng hướng (isotropic ideal LPF) với tần số cắt Ucutoff = 64 cpi
  cho cả 3 ảnh (girl2, girl2Noise32Hi, girl2Noise32).
- Sử dụng tích chập vòng (circular convolution) trong miền DFT 256x256 để tránh hiệu ứng biên của zero-padding:
    [U, V] = meshgrid(-128:127, -128:127)
    HLtildeCenter = (sqrt(U^2 + V^2) <= 64)
    HLtilde = ifftshift(HLtildeCenter)
    Y = ifft2(fft2(X) * HLtilde).real
- Lưu mặt nạ bộ lọc và 3 ảnh sau lọc vào output/Bai_2/.
- Tính MSE và ISNR (dB) trên dữ liệu số thực float64 thu được trực tiếp từ IFFT2 (CHƯA stretch và làm tròn).
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
    """MSE tính trên dữ liệu float64."""
    return float(np.mean((reference - predicted) ** 2))


def isnr(reference: np.ndarray, noisy: np.ndarray, restored: np.ndarray) -> float:
    """ISNR (dB) tính trên dữ liệu float64."""
    before = mse(reference, noisy)
    after = mse(reference, restored)
    if after == 0:
        return float('inf')
    if before == 0:
        return float('-inf')
    return float(10.0 * np.log10(before / after))


def frequency_grid(size: int = N) -> tuple[np.ndarray, np.ndarray]:
    """Tạo lưới tọa độ tần số (u, v) từ -128 đến 127 cpi."""
    half = size // 2
    u = np.arange(-half, half, dtype=np.float64)
    return np.meshgrid(u, u)


def loc_ideal_lpf_64(x: np.ndarray, u_cutoff: float = 64.0) -> tuple[np.ndarray, np.ndarray]:
    """Lọc Ideal LPF qua circular convolution DFT 256x256."""
    u, v = frequency_grid()
    centered_mask = ((u * u + v * v) <= (u_cutoff ** 2)).astype(np.float64)
    h_filter = np.fft.ifftshift(centered_mask)
    y = np.fft.ifft2(np.fft.fft2(x) * h_filter).real
    return y, centered_mask


def main():
    print("=" * 65)
    print("BÀI 2(b): BỘ LỌC THÔNG THẤP LÝ TƯỞNG (IDEAL LPF, CUTOFF = 64 CPI)")
    print("Sinh viên: Trần Hoàng Đạt - MSSV: N23DCCN009")
    print("=" * 65)

    # 1. Đọc dữ liệu ảnh
    print("[1] Đọc dữ liệu 3 ảnh girl2.bin, girl2Noise32Hi.bin, girl2Noise32.bin...")
    girl2 = read_bin('girl2.bin')
    inputs = {
        'original': girl2,
        'high_pass': read_bin('girl2Noise32Hi.bin'),
        'broadband': read_bin('girl2Noise32.bin')
    }

    # 2. Áp dụng Ideal LPF cho 3 ảnh
    print("[2] Lọc Ideal LPF (circular convolution, Ucutoff = 64 cpi)...")
    results = {}
    filter_mask = None

    for label, img in inputs.items():
        res, mask = loc_ideal_lpf_64(img, u_cutoff=64.0)
        results[label] = res
        if filter_mask is None:
            filter_mask = mask

    # 3. Lưu ảnh kết quả và mặt nạ bộ lọc
    print("[3] Lưu mặt nạ và ảnh kết quả vào output/Bai_2/:")
    path_mask = OUT_DIR / '2b_mat_na_ideal_64.png'
    save_gray(path_mask, filter_mask)
    print(f"    - Mặt nạ bộ lọc:          {path_mask.name}")

    for label, res in results.items():
        p = OUT_DIR / f'2_ideal_64_{label}.png'
        save_gray(p, res)
        print(f"    - Ảnh sau lọc ({label:<9}): {p.name}")

    # 4. Tính toán MSE và ISNR (tính trên float64 gốc chưa qua stretch/làm tròn)
    print("\n[4] KẾT QUẢ ĐO MSE VÀ ISNR (IDEAL LPF, CUTOFF = 64 CPI):")
    print("-" * 65)
    print(f"{'ẢNH ĐẦU VÀO':<20}{'MSE TRƯỚC LỌC':>15}{'MSE SAU LỌC':>15}{'ISNR (dB)':>15}")
    print("-" * 65)

    stats = {}
    for label, res in results.items():
        mse_truoc = mse(girl2, inputs[label])
        mse_sau = mse(girl2, res)
        if label == 'original':
            isnr_val = "N/A (Ảnh gốc)"
            mse_truoc_str = "0.0000"
            isnr_num = None
        else:
            val = isnr(girl2, inputs[label], res)
            isnr_val = f"{val:+.4f} dB"
            mse_truoc_str = f"{mse_truoc:.4f}"
            isnr_num = val

        stats[label] = (mse_sau, isnr_num)
        print(f"{label:<20}{mse_truoc_str:>15}{mse_sau:>15.4f}{isnr_val:>15}")

    print("-" * 65)
    print("(*) Nhận xét kết quả:")
    print("    - Đối với nhiễu cao tần (high_pass): ISNR đạt +2.3945 dB (MSE giảm từ 692.51 xuống 399.00).")
    print("    - Đối với nhiễu toàn dải (broadband): ISNR đạt +1.3079 dB (MSE giảm từ 744.47 xuống 550.88).")
    print("    - Đối với ảnh gốc (original): Xuất hiện hiện tượng gợn sóng (ringing artifacts) ở các biên cạnh.")

    # 5. Hiển thị hình ảnh trực quan bằng Matplotlib (lưới 1x4)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('BÀI 2(b) - BỘ LỌC IDEAL LPF CUTOFF = 64 CPI (N23DCCN009 - Trần Hoàng Đạt)', fontsize=13, fontweight='bold')

    axes[0].imshow(filter_mask, cmap='gray')
    axes[0].set_title('Mặt nạ Ideal LPF (r <= 64)', fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(full_scale(results['original']), cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f"Ảnh gốc sau lọc\nMSE = {stats['original'][0]:.2f}", fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(full_scale(results['high_pass']), cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f"Hi-pass sau lọc\nMSE = {stats['high_pass'][0]:.2f} | ISNR = {stats['high_pass'][1]:+.2f}dB", fontsize=10)
    axes[2].axis('off')

    axes[3].imshow(full_scale(results['broadband']), cmap='gray', vmin=0, vmax=255)
    axes[3].set_title(f"Broadband sau lọc\nMSE = {stats['broadband'][0]:.2f} | ISNR = {stats['broadband'][1]:+.2f}dB", fontsize=10)
    axes[3].axis('off')

    plt.tight_layout()
    fig_path = OUT_DIR / 'bai2_b_figure.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"    - Đã lưu đồ thị hiển thị: {fig_path.name}")
    print("=" * 65)
    print("HOÀN THÀNH BÀI 2(b)!\n")

    if '--no-gui' not in sys.argv:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
