# Homework 5 - Digital Image Processing

**MSSV:** N23DCCN009  
**Họ tên:** Trần Hoàng Đạt  
**Lớp:** D23CQCNPM01-N  
**Môn học:** Xử lý ảnh số (INT 13146) - Học viện Công nghệ Bưu chính Viễn thông

Mỗi bài tập được tổ chức thành **từng file `.py` độc lập hoàn toàn (standalone & self-contained)**. Bạn có thể mang từng file đi chạy ở bất cứ đâu mà không cần phụ thuộc vào file trung gian (`common.py` hay module ngoài). Mỗi file tự đọc dữ liệu trong `data/`, tính toán, lưu ảnh kết quả và bật cửa sổ Matplotlib trực quan.

---

## 1. Cấu trúc thư mục

```text
HW5_N23DCCN009_TranHoangDat/
├── bai1_a.py             <- Bài 1(a): Lọc 7x7 miền không gian, zero padding 262x262
├── bai1_b.py             <- Bài 1(b): Lọc nhân DFT 384x384 (xuất 8 ảnh, verify 1a: diff = 0)
├── bai1_c.py             <- Bài 1(c): Lọc zero-phase DFT 512x512 (xuất 4 ảnh, verify 1a: diff = 0)
│
├── bai2_a.py             <- Bài 2(a): Đọc 3 ảnh girl2 & tính MSE ban đầu
├── bai2_b.py             <- Bài 2(b): Bộ lọc Ideal LPF (cutoff = 64 cpi, circular convolution)
├── bai2_c.py             <- Bài 2(c): Bộ lọc Gaussian LPF (cutoff = 64 cpi, linear FFT 512x512)
├── bai2_d.py             <- Bài 2(d): Bộ lọc Gaussian LPF (cutoff = 77.5 cpi, linear FFT 512x512)
│
├── data/                 <- 4 file ảnh thô 256x256 8-bit
│   ├── salesman.bin
│   ├── girl2.bin
│   ├── girl2Noise32Hi.bin
│   └── girl2Noise32.bin
│
└── output/               <- Nơi lưu trữ toàn bộ ảnh kết quả & đồ thị Matplotlib
    ├── Bai_1/            <- Ảnh kết quả và figure của Bài 1(a), 1(b), 1(c)
    └── Bai_2/            <- Ảnh kết quả và figure của Bài 2(a), 2(b), 2(c), 2(d)
```

---

## 2. Cách chạy trực tiếp từng file (Chủ động & Độc lập)

Mở terminal tại thư mục này và chạy trực tiếp file bạn muốn:

### Nhóm Bài 1 (`salesman.bin`):
```powershell
# Chạy Bài 1(a): Lọc trung bình 7x7 trực tiếp trong miền không gian
python bai1_a.py

# Chạy Bài 1(b): Lọc bằng phép nhân DFT 384x384 (hiển thị đủ 8 ảnh và đối chiếu)
python bai1_b.py

# Chạy Bài 1(c): Lọc bằng đáp ứng xung zero-phase 512x512
python bai1_c.py
```

### Nhóm Bài 2 (`girl2.bin` và 2 ảnh nhiễu):
```powershell
# Chạy Bài 2(a): Đọc 3 ảnh đầu vào và in MSE trước lọc
python bai2_a.py

# Chạy Bài 2(b): Bộ lọc Ideal LPF (Ucutoff = 64 cpi)
python bai2_b.py

# Chạy Bài 2(c): Bộ lọc Gaussian LPF (Ucutoff = 64 cpi)
python bai2_c.py

# Chạy Bài 2(d): Bộ lọc Gaussian LPF (Ucutoff = 77.5 cpi)
python bai2_d.py
```

*Mẹo: Thêm tham số `--no-gui` nếu chỉ muốn xuất file ảnh mà không bật cửa sổ pop-up hiển thị (ví dụ: `python bai1_a.py --no-gui`).*

---

## 3. Bảng đối chiếu số liệu định lượng

### Bài 1 - Sai khác pixel sau khi full-scale contrast & làm tròn số nguyên:
- **1(b) so với 1(a):** `0 pixel` (sai khác số thực float64: `8.527e-14`)
- **1(c) so với 1(a):** `0 pixel` (sai khác số thực float64: `8.527e-14`)

### Bài 2 - Bảng đo MSE và ISNR (tính trên float64 nguyên bản chưa làm tròn):
| Phương pháp | $U_{cutoff}$ | Ảnh đầu vào | MSE trước lọc | MSE sau lọc | ISNR (dB) |
|:---|:---:|:---|---:|---:|---:|
| **Bài 2(a)** (Gốc) | - | Nhiễu cao tần (`high_pass`) | 692.5050 | - | - |
| **Bài 2(a)** (Gốc) | - | Nhiễu toàn dải (`broadband`) | 744.4679 | - | - |
| **Ideal LPF** | 64 | Ảnh chuẩn (`original`) | 0.0000 | 127.7481 | N/A |
| **Ideal LPF** | 64 | Nhiễu cao tần (`high_pass`) | 692.5050 | 398.9978 | **+2.3945 dB** |
| **Ideal LPF** | 64 | Nhiễu toàn dải (`broadband`) | 744.4679 | 550.8787 | **+1.3079 dB** |
| **Gaussian LPF** | 64 | Ảnh chuẩn (`original`) | 0.0000 | 91.2585 | N/A |
| **Gaussian LPF** | 64 | Nhiễu cao tần (`high_pass`) | 692.5050 | 415.8219 | **+2.2152 dB** |
| **Gaussian LPF** | 64 | Nhiễu toàn dải (`broadband`) | 744.4679 | 534.7741 | **+1.4368 dB** |
| **Gaussian LPF** | 77.5 | Ảnh chuẩn (`original`) | 0.0000 | **60.6184** | N/A |
| **Gaussian LPF** | 77.5 | Nhiễu cao tần (`high_pass`) | 692.5050 | 406.7896 | **+2.3105 dB** |
| **Gaussian LPF** | 77.5 | Nhiễu toàn dải (`broadband`) | 744.4679 | 527.7710 | **+1.4940 dB** |
