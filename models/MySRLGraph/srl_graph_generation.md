# Đây là ý tưởng nghiên cứu của tôi

Việc bạn quyết định **bỏ qua việc dự đoán kiểu vị ngữ (như `kill.01`, `cause.01`) là một hướng đi rất chính xác và thực tế**. Trong bài toán SRL, việc xác định nghĩa của từ (Word Sense Disambiguation) nên để cho các mô hình ngôn ngữ mạnh như DeBERTa tự xử lý thông qua Embedding. Việc bắt mô hình Diffusion vừa phải học cấu trúc đồ thị, vừa phải học phân loại nghĩa từ sẽ làm bùng nổ số lượng nhãn không cần thiết.

Khi bạn chỉ tập trung vào việc **dự đoán quan hệ (roles/relations) giữa các nút**, định dạng Input của DiGress sẽ trở nên đơn giản và tối ưu hơn rất nhiều.

Để bạn dễ hình dung, chúng ta hãy cùng đi vào một ví dụ thực tế với một câu ngắn gồm 4 từ (4 nodes) được trích từ ảnh của bạn:

> **Câu văn:** `"killing cleric causing trouble"` (việc giết giáo sĩ gây ra rắc rối)

---

## 1. Định nghĩa tập nhãn Cạnh (Edge Labels)

Vì chúng ta chỉ dự đoán quan hệ giữa các từ, ta định nghĩa số lượng nhãn cạnh là $d_e = 3$:

* **Chỉ mục 0:** `No-Edge` (Không có quan hệ ngữ nghĩa giữa 2 từ)
* **Chỉ mục 1:** `ARG0` (Chủ thể thực hiện hành động)
* **Chỉ mục 2:** `ARG1` (Đối tượng bị tác động)

---

## 2. Chi tiết các Tensor Input trong Ma trận DiGress

Giả sử `batch_size = 1` và chiều dài câu $N = 4$.

### A. Tensor Cạnh $E$ — Kích thước `(1, 4, 4, 3)`

Đây là tensor quan trọng nhất. Nó là một ma trận 2D kích thước $4 \times 4$ (tương ứng với 4 từ $\times$ 4 từ). Tại **mỗi ô** của ma trận, thay vì chỉ lưu một con số, nó lưu một **One-hot Vector chiều dài bằng 3** để biểu diễn nhãn quan hệ.

Mối quan hệ thực tế trong câu:

1. `killing` tác động lên `cleric` $\rightarrow$ `ARG1`
2. `causing` bị tác động bởi hành vi `killing` $\rightarrow$ `ARG0`
3. `causing` gây ra `trouble` $\rightarrow$ `ARG1`

Ma trận $E$ ở trạng thái sạch ($t=0$) sẽ trông như sau:

| Từ gốc (Hàng) | [0] killing | [1] cleric | [2] causing | [3] trouble |
| --- | --- | --- | --- | --- |
| **[0] killing** | `[1, 0, 0]` *(No-Edge)* | **`[0, 0, 1]` *(ARG1)*** | `[1, 0, 0]` *(No-Edge)* | `[1, 0, 0]` *(No-Edge)* |
| **[1] cleric** | `[1, 0, 0]` *(No-Edge)* | `[1, 0, 0]` *(No-Edge)* | `[1, 0, 0]` *(No-Edge)* | `[1, 0, 0]` *(No-Edge)* |
| **[2] causing** | **`[0, 1, 0]` *(ARG0)*** | `[1, 0, 0]` *(No-Edge)* | `[1, 0, 0]` *(No-Edge)* | **`[0, 0, 1]` *(ARG1)*** |
| **[3] trouble** | `[1, 0, 0]` *(No-Edge)* | `[1, 0, 0]` *(No-Edge)* | `[1, 0, 0]` *(No-Edge)* | `[1, 0, 0]` *(No-Edge)* |

> **Cơ chế thêm nhiễu (Forward Process):** Khi ở bước $t > 0$, các ô có nhãn `[0, 0, 1]` (`ARG1`) hoặc `[0, 1, 0]` (`ARG0`) sẽ bị ma trận chuyển trạng thái phá hủy ngẫu nhiên biến thành `[1, 0, 0]` (`No-Edge`) hoặc ngược lại. Nhiệm vụ của mô hình ở bước Reverse là từ các vector bị xáo trộn đó, đoán lại chính xác bảng một-nóng (one-hot) như trên.

---

### B. Tensor Nút $X$ — Kích thước `(1, 4, d_x)`

Vì bạn đã loại bỏ việc dự đoán kiểu vị ngữ (`kill.01`), cấu trúc của $X$ sẽ thay đổi theo hướng **tối ưu hơn**:

* **Cách 1: Biến $X$ thành thuộc tính cố định (Static Features) không thêm nhiễu.**
Thay vì để DiGress khuếch tán $X$, bạn giữ nguyên $X$ là các vector Embedding từ DeBERTa (kích thước `(1, 4, 768)`). Suốt quá trình T bước, $X$ không bị thêm nhiễu, nó đóng vai trò làm kim chỉ nam ngữ cảnh để mô hình khôi phục ma trận cạnh $E$.
* **Cách 2: Nếu vẫn muốn giữ nguyên pipeline của DiGress ($X$ là biến rời rạc), bạn chỉ cần đặt $d_x = 2$ để phân biệt đâu là động từ/vị ngữ gốc:**
* Chỉ mục 0: `Normal Token`
* Chỉ mục 1: `Predicate Token` (Từ đóng vai trò trung tâm ngữ nghĩa)



Khi đó Tensor $X$ dạng one-hot của câu trên sẽ là:

* `[0] killing` (Là vị ngữ): `[0, 1]`
* `[1] cleric` (Từ thường): `[1, 0]`
* `[2] causing` (Là vị ngữ): `[0, 1]`
* `[3] trouble` (Từ thường): `[1, 0]`

---

### C. Tensor `node_mask` — Kích thước `(1, 4)`

Tensor này rất đơn giản, nó dùng để xử lý các câu dài ngắn khác nhau trong cùng một Batch.

Giả sử trong Batch đó, câu dài nhất có 6 từ ($N_{max} = 6$), nhưng câu ví dụ của chúng ta chỉ có 4 từ. Mô hình sẽ tự động thêm 2 nút đệm (Padding). Khi đó `node_mask` sẽ là:

```python
node_mask = [True, True, True, True, False, False]

```

Mô hình sẽ nhìn vào đây và bỏ qua hoàn toàn không tính toán Loss hay cập nhật trọng số cho 2 nút `False` cuối cùng.

---

### D. Tensor Toàn cục $y$ — Kích thước `(1, d_y)`

Trong quá trình huấn luyện, tensor này chủ yếu chứa bước thời gian $t$ hiện tại (ví dụ: $t = 25$ trên tổng số $T = 100$ bước khuếch tán) được nhúng dưới dạng vector (Time Embedding) để mô hình biết nó đang ở giai đoạn nhiễu nặng hay nhiễu nhẹ mà đưa ra dự đoán phù hợp.


Hướng tiếp cận này trong nghiên cứu được gọi là **Edge-only Diffusion (Khuếch tán chỉ áp dụng lên cạnh)** hoặc **Node-conditioned Graph Diffusion**.

Để hiểu rõ tại sao lại có sự khác biệt này và nó mang lại lợi ích gì, chúng ta hãy cùng đặt hai bài toán lên bàn cân:

---

## 1. So sánh DiGress gốc vs. Bài toán SRL của bạn

| Tiêu chí | DiGress Gốc (Sinh phân tử hóa học) | Bài toán SRL của bạn (Sinh đồ thị ngữ nghĩa) |
| --- | --- | --- |
| **Bản chất bài toán** | **Sinh vô điều kiện (Unconditional):** Tạo ra một phân tử hoàn toàn mới từ hư không. | **Sinh có điều kiện (Conditional):** Bắt buộc phải dựa vào câu văn bản đầu vào. |
| **Trạng thái của Nút ($X$)** | **Chưa biết:** Mô hình không biết đồ thị mới sẽ chứa nguyên tử Carbon, Oxy hay Nitơ. | **Đã biết cố định:** Nút chính là các từ trong câu (`killing`, `cleric`,...). Danh tính và thứ tự của nút không thay đổi. |
| **Mục tiêu khuếch tán** | Phải thêm nhiễu và học cách sinh **đồng thời** cả Nút lẫn Cạnh để chúng khớp nhau. | **Chỉ cần thêm nhiễu và sinh Ma trận Cạnh ($E$)**. Nút $X$ được giữ nguyên làm "neo" ngữ cảnh. |

---

## 2. Quá trình "Edge-only Diffusion" sẽ hoạt động như thế nào?

Khi bạn chỉ áp dụng nhiễu lên quan hệ (cạnh), luồng đi của mô hình sẽ thay đổi trực tiếp trong code như sau:

### Pha thêm nhiễu (Forward Process)

* **Nút $X$:** Bạn giữ nguyên vector Embedding của DeBERTa ($X_0 = H$) xuyên suốt từ bước $t = 0$ đến bước $t = T$. **Không gọi hàm thêm nhiễu lên $X$.**
* **Cạnh $E$:** Tiến hành thêm nhiễu rời rạc thông qua ma trận chuyển trạng thái $Q_t^E$. Ở bước $T$, ma trận cạnh $E_T$ sẽ biến thành một ma trận ngẫu nhiên (phần lớn các ô sẽ bị xáo trộn ngẫu nhiên thành `No-Edge`, `ARG0`, `ARG1` dựa trên phân phối biên).

[Diagram of edge-only graph diffusion process]

### Pha khử nhiễu (Reverse Process)

Mạng Graph Transformer tại mỗi bước $t$ sẽ nhận vào:

1. Ma trận đặc trưng nút **sạch hoàn toàn** $X_0$ (Text Embeddings).
2. Ma trận cạnh **đang bị nhiễu** $E_t$.
3. Bước thời gian $t$.

Nhiệm vụ của mạng lúc này cực kỳ tập trung: **"Dựa vào ý nghĩa của các từ đã biết ở $X_0$, hãy khôi phục lại các ô quan hệ bị mờ/nhiễu ở $E_t$ để trả về ma trận cạnh sạch $\hat{E}_0$".**

### Hàm Loss khi huấn luyện

Bạn loại bỏ hoàn toàn thành phần Loss của nút. Hàm Loss tổng lúc này chỉ còn:


$$\mathcal{L} = \mathcal{L}_{CE}(E_0, \hat{E}_0)$$


Mô hình chỉ bị phạt nếu dự đoán sai nhãn quan hệ (`ARG0`, `ARG1`,...) giữa các từ.


## Dataset

lấy từ folder: dataset/propbank