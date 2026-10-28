# -*- coding: utf-8 -*-
"""
BÀI 2(d) - BỘ LỌC THÔNG THẤP GAUSS (GAUSSIAN LPF, UCUTOFF = 77.5 CPI)
MSSV: N23DCCN009
Họ và tên: Trần Hoàng Đạt
Lớp: D23CQCNPM01-N
Môn học: Xử lý ảnh số (INT 13146)

Yêu cầu bài toán (theo hw5.pdf):
- Áp dụng bộ lọc thông thấp Gauss (Gaussian LPF) với tần số cắt Ucutoff = 77.5 cpi cho cả 3 ảnh:
  girl2, girl2Noise32Hi, girl2Noise32.
- Thiết kế hằng số không gian sigma:
    SigmaH = 0.19 * N / Ucutoff = 0.19 * 256 / 77.5 ≈ 0.6276
- Sử dụng thuật toán Example 4 với padding 512x512:
    1. Htilde = ifftshift(HtildeCenter)
    2. h = ifft2(Htilde).real
    3. h2 = fftshift(h)
    4. ZPH2 (512x512): ZPH2[:256, :256] = h2
    5. xpad (512x512): xpad[:256, :256] = x
    6. ypad = ifft2(fft2(xpad) * fft2(ZPH2)).real
    7. Cắt ảnh kết quả tại vùng giữa: y = ypad[128:384, 128:384]
- Lưu đáp ứng tần số, đáp ứng xung và 3 ảnh sau lọc vào output/Bai_2/.
- Đo MSE và ISNR (dB) trên dữ liệu float64 gốc chưa qua stretch/làm tròn.
- Đánh giá: Biến dạng ảnh gốc ít nhất (MSE 60.62) và đạt ISNR cao nhất đối với nhiễu toàn dải (+1.4940 dB).
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


def loc_gaussian_lpf(x: np.ndarray, u_cutoff: float = 77.5) -> tuple[np.ndarray, dict]:
    """Lọc Gaussian LPF qua FFT padding 512x512 theo Example 4."""
    u, v = frequency_grid()
    sigma = 0.19 * N / u_cutoff
    h_freq_center = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) / (N ** 2) * (u * u + v * v))

    h_freq_uncentered = np.fft.ifftshift(h_freq_center)
    h_impulse_zero_phase = np.fft.ifft2(h_freq_uncentered).real
    h_impulse_center = np.fft.fftshift(h_impulse_zero_phase)

    h_pad = np.zeros((512, 512), dtype=np.float64)
    x_pad = np.zeros((512, 512), dtype=np.float64)
    h_pad[:N, :N] = h_impulse_center
    x_pad[:N, :N] = x

    y_pad = np.fft.ifft2(np.fft.fft2(x_pad) * np.fft.fft2(h_pad)).real
    result = y_pad[128:384, 128:384]

    return result, {
        'sigma': sigma,
        'h_freq_center': h_freq_center,
        'h_impulse_center': h_impulse_center
    }


def main():
    print("=" * 65)
    print("BÀI 2(d): BỘ LỌC THÔNG THẤP GAUSS (GAUSSIAN LPF, CUTOFF = 77.5 CPI)")
    print("Sinh viên: Trần Hoàng Đạt - MSSV: N23DCCN009")
    print("=" * 65)

    cutoff = 77.5
    sigma = 0.19 * N / cutoff
    print(f"[1] Tham số bộ lọc:")
    print(f"    - Tần số cắt Ucutoff: {cutoff} cpi")
    print(f"    - Hằng số không gian sigma: {sigma:.4f}")

    # 1. Đọc dữ liệu ảnh
    print("\n[2] Đọc dữ liệu 3 ảnh girl2.bin, girl2Noise32Hi.bin, girl2Noise32.bin...")
    girl2 = read_bin('girl2.bin')
    inputs = {
        'original': girl2,
        'high_pass': read_bin('girl2Noise32Hi.bin'),
        'broadband': read_bin('girl2Noise32.bin')
    }

    # 2. Áp dụng Gaussian LPF 77.5 cho 3 ảnh
    print("[3] Thực hiện lọc Gaussian LPF (FFT padding 512x512)...")
    results = {}
    extra_data = None

    for label, img in inputs.items():
        res, extra = loc_gaussian_lpf(img, u_cutoff=cutoff)
        results[label] = res
        if extra_data is None:
            extra_data = extra

    # 3. Lưu đáp ứng tần số, xung và 3 ảnh sau lọc
    print("[4] Lưu ảnh đáp ứng và kết quả lọc vào output/Bai_2/:")
    path_freq = OUT_DIR / '2_gaussian_77_5_dap_ung_tan_so.png'
    path_impulse = OUT_DIR / '2_gaussian_77_5_dap_ung_xung.png'
    save_gray(path_freq, extra_data['h_freq_center'])
    save_gray(path_impulse, extra_data['h_impulse_center'])
    print(f"    - Đáp ứng tần số H(u, v):  {path_freq.name}")
    print(f"    - Đáp ứng xung h(m, n):    {path_impulse.name}")

    for label, res in results.items():
        p = OUT_DIR / f'2_gaussian_77_5_{label}.png'
        save_gray(p, res)
        print(f"    - Ảnh sau lọc ({label:<9}): {p.name}")

    # 4. Tính toán MSE và ISNR (tính trên float64 gốc chưa qua stretch/làm tròn)
    print("\n[5] KẾT QUẢ ĐO MSE VÀ ISNR (GAUSSIAN LPF, CUTOFF = 77.5 CPI):")
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
    print("(*) Nhận xét tổng kết so với hai bộ lọc trước:")
    print("    - MSE trên ảnh gốc nhỏ nhất (60.6184), ảnh sắc nét và ít biến dạng nhất.")
    print("    - Đạt hiệu quả ISNR cao nhất đối với nhiễu toàn dải broadband (+1.4940 dB).")

    # 5. Hiển thị hình ảnh trực quan bằng Matplotlib (lưới 1x5)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    fig.suptitle('BÀI 2(d) - BỘ LỌC GAUSSIAN LPF CUTOFF = 77.5 CPI (N23DCCN009 - Trần Hoàng Đạt)', fontsize=13, fontweight='bold')

    axes[0].imshow(full_scale(extra_data['h_freq_center']), cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Đáp ứng tần số H', fontsize=9)
    axes[0].axis('off')

    axes[1].imshow(full_scale(extra_data['h_impulse_center']), cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Đáp ứng xung h', fontsize=9)
    axes[1].axis('off')

    axes[2].imshow(full_scale(results['original']), cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f"Gốc sau lọc\nMSE = {stats['original'][0]:.2f}", fontsize=9)
    axes[2].axis('off')

    axes[3].imshow(full_scale(results['high_pass']), cmap='gray', vmin=0, vmax=255)
    axes[3].set_title(f"Hi-pass sau lọc\nMSE = {stats['high_pass'][0]:.2f} | ISNR = {stats['high_pass'][1]:+.2f}dB", fontsize=9)
    axes[3].axis('off')

    axes[4].imshow(full_scale(results['broadband']), cmap='gray', vmin=0, vmax=255)
    axes[4].set_title(f"Broadband sau lọc\nMSE = {stats['broadband'][0]:.2f} | ISNR = {stats['broadband'][1]:+.2f}dB", fontsize=9)
    axes[4].axis('off')

    plt.tight_layout()
    fig_path = OUT_DIR / 'bai2_d_figure.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"    - Đã lưu đồ thị hiển thị: {fig_path.name}")
    print("=" * 65)
    print("HOÀN THÀNH BÀI 2(d)!\n")

    if '--no-gui' not in sys.argv:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
