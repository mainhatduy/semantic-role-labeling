# MySRLGraph: Edge-only Discrete Diffusion for SRL Graph Generation

Dự án này triển khai **MySRLGraph**, một mô hình khuếch tán rời rạc có điều kiện (conditional, edge-only discrete denoising diffusion model) nhằm sinh ra đồ thị SRL (Semantic Role Labeling) từ câu văn bản đầu vào.

Mô hình được xây dựng dựa trên kiến trúc DiGress nhưng tối ưu hóa riêng cho tác vụ SRL:
- **Cố định các node ($X$):** Biểu diễn các từ dưới dạng embedding liên tục từ mô hình XLM-RoBERTa (768d), bỏ qua việc thêm nhiễu lên node.
- **Khuếch tán trên cạnh ($E$):** Chỉ khuếch tán ma trận nhãn cạnh rời rạc (gồm 58 lớp: `No-Edge` + 57 nhãn quan hệ ngữ nghĩa).
- **Đồ thị có hướng (Asymmetric / Directed Edges):** Các quan hệ ngữ nghĩa đi một chiều từ Vị ngữ (Predicate) đến Tham tố (Argument).

---

## 1. Cài đặt môi trường (Environment Setup)

Dự án yêu cầu hệ điều hành Linux (khuyên dùng Ubuntu) có hỗ trợ GPU CUDA để tăng tốc độ huấn luyện.

### Bước 1: Tạo môi trường ảo (Khuyên dùng Conda)
Khuyên dùng Python 3.9 hoặc 3.10 để đảm bảo tính tương thích tốt nhất với PyTorch Geometric.

```bash
conda create -n srl_env python=3.9 -y
conda activate srl_env
```

### Bước 2: Cài đặt PyTorch và PyTorch Geometric
Cài đặt phiên bản PyTorch phù hợp với phiên bản CUDA trên máy của bạn. Ví dụ đối với CUDA 11.8:

```bash
# Cài đặt PyTorch
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Cài đặt PyTorch Geometric (PyG) và các thư viện phụ thuộc
pip install torch_geometric==2.3.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

### Bước 3: Cài đặt các thư viện bổ sung từ `requirements.txt`
Di chuyển vào thư mục `models/MySRLGraph` và cài đặt:

```bash
cd models/MySRLGraph
pip install -r requirements.txt
```

---

## 2. Chuẩn bị Dữ liệu (Data Preparation)

Dữ liệu huấn luyện PropBank được lưu trữ tại thư mục `dataset/propbank/` ở gốc của repository:
- `dataset/propbank/train.jsonl`: Chứa các câu văn bản cùng với danh sách các vị ngữ (predicates) và tham tố (arguments).
- `dataset/propbank/unique_roles.json`: Chứa danh sách các nhãn quan hệ ngữ nghĩa (ví dụ: `ARG0`, `ARG1`, `ARGM-TMP`,...).

Cấu hình đường dẫn dữ liệu và mô hình ngôn ngữ đã được thiết lập sẵn trong file [configs/dataset/srl_propbank.yaml](configs/dataset/srl_propbank.yaml):
```yaml
name: 'srl_propbank'
datadir: '../../dataset/propbank'
train_file: 'train.jsonl'
roles_file: 'unique_roles.json'
embedding_model: 'FacebookAI/xlm-roberta-base'
embedding_dim: 768
```

---

## 3. Huấn luyện Mô hình (Training)

Quá trình huấn luyện sử dụng **Hydra** để cấu hình và **PyTorch Lightning** để huấn luyện mô hình. Hãy đảm bảo bạn đang đứng ở thư mục `models/MySRLGraph`.

### Huấn luyện thông thường (GPU)
Để huấn luyện với cấu hình thí nghiệm SRL mặc định (500 epochs, batch size 32, lưu checkpoint tự động):
```bash
python src/main.py +experiment=srl
```

### Huấn luyện trên CPU (Không dùng GPU)
Nếu hệ thống của bạn không hỗ trợ GPU, chạy lệnh sau:
```bash
python src/main.py +experiment=srl general.gpus=0
```

### Chạy thử nghiệm nhanh (Debug Mode)
Để kiểm tra lỗi runtime nhanh chóng trước khi chạy thật, bạn có thể chạy chế độ debug (chỉ chạy 1 epoch huấn luyện và validate):
```bash
python src/main.py +experiment=srl general.name=debug
```

### Tiếp tục huấn luyện từ checkpoint (Resume Training)
Nếu quá trình huấn luyện bị gián đoạn, bạn có thể tiếp tục bằng cách chỉ định đường dẫn checkpoint:
```bash
python src/main.py +experiment=srl general.resume=checkpoints/srl_edge_diffusion/epoch=X.ckpt
```
*(Thay thế `epoch=X.ckpt` bằng checkpoint thực tế trong thư mục `checkpoints/srl_edge_diffusion/`)*

---

## 4. Kiểm tra Kết quả (Evaluation)

Sau khi huấn luyện hoàn tất, mô hình sẽ tự động chạy thử nghiệm (testing) và ghi lại kết quả F1-Score, Precision, Recall và độ chính xác khớp hoàn chỉnh (Exact Match Accuracy).

Để chỉ chạy đánh giá (Testing/Inference) dựa trên một checkpoint đã lưu:
```bash
python src/main.py +experiment=srl general.test_only=checkpoints/srl_edge_diffusion/epoch=X.ckpt
```

---

## 5. Đầu ra của mô hình (Outputs)
- **Checkpoints:** Lưu tại thư mục `checkpoints/srl_edge_diffusion/`.
- **Logs & TensorBoard/Wandb:** Được quản lý theo cấu hình chung, mặc định log sẽ được đẩy lên Wandb (nếu sử dụng online) hoặc lưu local trong thư mục `outputs/`.
