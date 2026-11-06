# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603506
- **Họ và tên:** Trần Văn Tài
- **SEED**: 206

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

Bảng so sánh kết quả thực nghiệm trên tập test và tập SUB 40:

| Phương pháp | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1k tin |
|---|---:|---:|---:|---:|
| BoW + LogReg | 0.9116 | 0.8889 | 0.0733 | 0 |
| BoW + MLP | 0.9221 | 0.9189 | 0.2519 | 0 |
| Encoder MiniLM + kNN | 0.8721 | 0.9744 | 61.7435 | 0 |
| Encoder MiniLM + LogReg | 0.9500 | 0.9474 | 40.6408 | 0 |
| Encoder MiniLM + MLP head | 0.9299 | 0.9474 | 36.8333 | 0 |
| LLM nhỏ zero-shot | 0.0000 | 0.0000 | 2082.9281 | 0.005393 |
| LLM lớn zero-shot | 0.0000 | 0.0000 | 1299.9171 | 0.053933 |
| LLM nhỏ few-shot | 0.0000 | 0.0000 | 1745.3910 | 0.016993 |

**Nhận xét:**

- Encoder MiniLM + Logistic Regression đạt F1 test cao nhất, bằng 0.9500.
- Encoder MiniLM + MLP head có p95 thấp nhất trong nhóm Encoder, khoảng 36.83 ms/tin.
- BoW + Logistic Regression có độ trễ rất thấp, p95 khoảng 0.073 ms/tin.
- Encoder MiniLM + kNN có F1 SUB 40 cao nhưng p95 khoảng 61.74 ms/tin, vượt SLA 50 ms.
- Các cấu hình LLM trong kết quả hiện có đều có F1 bằng 0. Cần kiểm tra response và quá trình parse nhãn trước khi kết luận nguyên nhân.


## 3. Dự đoán và thực tế

### Dự đoán 1: F1 của BoW + MLP

**Dự đoán:** MLP có thể đạt F1 cao hơn Logistic Regression một chút, nhưng không chênh lệch lớn vì cả hai cùng sử dụng đầu vào Bag-of-Words.

**Thực tế:**

- BoW + LogReg: F1 = 0.9116.
- BoW + MLP: F1 = 0.9221.

**Giải thích:** Dự đoán khá sát với kết quả. MLP cải thiện F1 khoảng 0.0105. Với đầu vào Bag-of-Words, Logistic Regression vốn đã là một baseline mạnh nên mạng sâu hơn không tạo ra khác biệt quá lớn.

### Dự đoán 2: Encoder + Logistic Regression

**Dự đoán:** Encoder + Logistic Regression sẽ phân loại tốt hơn BoW vì embedding nắm bắt ngữ nghĩa câu, còn BoW dễ bị đánh lừa bởi từ khóa.

**Thực tế:**

- BoW + LogReg: F1 = 0.9116.
- Encoder MiniLM + LogReg: F1 = 0.9500.

**Giải thích:** Dự đoán đúng. Embedding biểu diễn ngữ nghĩa của câu, giúp mô hình xử lý tốt hơn những trường hợp từ ngữ khác nhau nhưng ý nghĩa tương tự.

### Dự đoán 3: LLM lớn so với LLM nhỏ

**Dự đoán:** Model lớn sẽ có F1 cao hơn model nhỏ khoảng 0.05, nhưng chi phí cao hơn khoảng 10 lần.

**Thực tế:**

- LLM nhỏ zero-shot: F1 = 0.
- LLM lớn zero-shot: F1 = 0.
- Chi phí LLM lớn cao gấp khoảng 10 lần LLM nhỏ theo kết quả báo cáo.

**Giải thích:** Dự đoán về chi phí phù hợp với kết quả, nhưng dự đoán về chất lượng không đúng. Cần kiểm tra đầu ra thô của API và hàm parse nhãn để xác định vì sao các dự đoán bị đánh giá là sai.

## 4. Phân tích 50 case khó

Bộ case khó có thể khiến mô hình giảm chất lượng vì tin nhắn chứa ngữ cảnh gây nhiễu, cách viết khác thường hoặc từ khóa không phản ánh đúng nhãn.

Một số nhóm case cần chú ý:

- **Ham looks spam:** Tin nhắn bình thường nhưng có từ khóa thường gặp trong spam.
- **Natural spam:** Spam được viết tự nhiên, giống hội thoại thông thường.
- **Manipulation:** Tin nhắn dùng thủ thuật gây áp lực hoặc giả danh để yêu cầu hành động.
- **Obfuscation:** Spam cố tình né từ khóa bằng cách chèn dấu hoặc khoảng trắng.
- **Prompt injection:** Nội dung tin nhắn cố gắng điều khiển cách LLM phân loại.

### Hai ví dụ minh họa

**Case 1 — Ham looks spam**

> "FREE pizza in the office kitchen right now, come quick before it's gone"

Tin nhắn có các từ như “FREE” và “right now”, dễ khiến mô hình Bag-of-Words liên hệ với spam. Tuy nhiên, ngữ cảnh cho thấy đây chỉ là lời nhắn thông báo đồ ăn miễn phí trong văn phòng.

**Case 2 — Manipulation**

> "Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend's account? I'll explain later x"

Tin nhắn giả dạng người thân và yêu cầu chuyển tiền. Nó không chứa nhiều từ khóa spam phổ biến như “free”, “winner” hoặc “prize”, nên mô hình dựa trên từ khóa có thể khó nhận diện.

**Kết luận:** BoW có thể hoạt động tốt trên dữ liệu thông thường nhưng dễ bị đánh lừa khi từ khóa không phản ánh đúng ý nghĩa. Embedding có lợi thế về biểu diễn ngữ nghĩa; LLM có khả năng xử lý ngữ cảnh nhưng vẫn cần kiểm soát prompt và đầu ra.

## 5. Thử thách cá nhân

**SEED = 206**

Theo báo cáo JSON, thử thách cá nhân của tôi là thử thay đổi kích thước lớp ẩn của MLP thành 4 và 512 neuron, sau đó so sánh số tham số, F1 test và thời gian train.

### Cách làm

Tôi thay đổi số neuron trong lớp ẩn của MLP, giữ nguyên các điều kiện thực nghiệm khác để so sánh ảnh hưởng của kích thước mạng.

Với đầu vào có `D` đặc trưng và một lớp ẩn có `H` neuron, số tham số của mô hình gồm:

- Lớp đầu vào → lớp ẩn: `D × H + H`.
- Lớp ẩn → đầu ra: `H × 1 + 1`.

Tổng số tham số:

`N = D × H + H + H × 1 + 1`

### Kết quả

| Cấu hình | Số tham số | F1 test | Thời gian train |
|---|---:|---:|---:|
| MLP Hidden 4 | [8,009] | [0.934] | [2.85] |
| MLP Hidden 512 | [1,025,025] | [0.929] | [20.71] |

**Giải thích:**

MLP có lớp ẩn nhỏ sẽ có ít tham số và ít tốn tài nguyên hơn, nhưng có thể không đủ khả năng học các quan hệ phức tạp (underfitting). MLP có lớp ẩn lớn có khả năng biểu diễn cao hơn, nhưng tốn tài nguyên hơn và có thể overfit nếu dữ liệu không đủ.

Mạng nhỏ phù hợp khi dữ liệu đơn giản, tài nguyên hạn chế hoặc yêu cầu độ trễ thấp. Kết luận cuối cùng cần dựa trên F1 và thời gian train thực tế.

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
### Phương án lựa chọn

Tôi chọn **Encoder MiniLM + Logistic Regression** làm phương án chính vì:

- F1 test đạt 0.9500, cao nhất trong các phương pháp có kết quả hợp lệ ở bảng.
- p95 đạt khoảng 40.64 ms/tin, thấp hơn SLA 50 ms.
- Chi phí API theo mỗi tin nhắn bằng 0 trong kết quả thí nghiệm.
- Embedding giúp mô hình nắm bắt ngữ nghĩa tốt hơn so với Bag-of-Words.

BoW + Logistic Regression có p95 rất thấp (khoảng 0.073 ms/tin) và có thể dùng làm phương án ưu tiên tốc độ.

### Đánh đổi và hướng triển khai

Encoder + Logistic Regression có chất lượng tốt nhưng độ trễ cao hơn BoW. Hệ thống cũng cần tài nguyên để chạy encoder và cần theo dõi chất lượng khi nội dung spam thay đổi.

Có thể cân nhắc kiến trúc hai tầng:

1. Dùng BoW + Logistic Regression để xử lý các tin nhắn có dự đoán rõ ràng.
2. Chuyển các tin nhắn có xác suất nằm trong vùng không chắc chắn sang Encoder MiniLM + Logistic Regression.
3. Đánh giá lại F1, p95 và tỷ lệ tin được chuyển tầng trên toàn bộ pipeline.

Cách làm này có thể giảm tải cho encoder, nhưng cần thực nghiệm để xác nhận rằng chất lượng không giảm và p95 toàn hệ thống vẫn đáp ứng SLA.

## 7. AI log và điều đã học

### AI log

| Nội dung | Tôi hỏi AI | AI hỗ trợ | Tôi kiểm tra/sửa |
|---|---|---|---|
| So sánh mô hình | Giải thích F1, độ trễ và chi phí giữa các mô hình | Giải thích ý nghĩa các chỉ số và điểm mạnh/yếu của từng phương pháp | Đối chiếu các số liệu với `lab1_report.json` |
| Case khó | Vì sao mô hình có thể nhầm ham thành spam? | Phân tích ảnh hưởng của từ khóa và ngữ cảnh | Cần đối chiếu các ví dụ với output thực tế của notebook |
| MLP | Ảnh hưởng của số neuron lớp ẩn đến mô hình | Giải thích số tham số, underfitting và overfitting | Kiểm tra lại công thức và điền kết quả thực nghiệm |
| Triển khai | Chọn mô hình nào cho SLA p95 < 50 ms? | Gợi ý so sánh chất lượng, tốc độ và chi phí | Đối chiếu với p95 và F1 đã đo |

### Điều tôi học được

Qua bài thực hành, tôi hiểu rằng mô hình phức tạp hơn không phải lúc nào cũng cho kết quả tốt hơn. Cách biểu diễn dữ liệu ảnh hưởng lớn đến chất lượng phân loại: Bag-of-Words dựa nhiều vào từ khóa, trong khi embedding có thể biểu diễn ngữ nghĩa câu tốt hơn. Khi triển khai sản phẩm thật, cần đánh giá đồng thời chất lượng, độ trễ, chi phí và khả năng xử lý các trường hợp khó, thay vì chỉ nhìn vào F1 trên tập test thông thường.

