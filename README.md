# Sepsis Monitoring - Big Data Pipeline

Hệ thống phân tích và giám sát thời gian thực bệnh nhân nhiễm khuẩn huyết (Sepsis) sử dụng các công nghệ Big Data hiện đại. Dự án này giả lập luồng dữ liệu y tế từ các thiết bị đo đạc trong phòng ICU, dự đoán và cảnh báo Sepsis ngay trong thời gian thực.

## 🏗 Kiến trúc Hệ thống (Architecture)

Hệ thống được thiết kế theo kiến trúc streaming, bao gồm các thành phần chính:

1. **Apache NiFi:** Đóng vai trò là Ingestion Layer. Đọc dữ liệu thô (các file `.psv`) từ thư mục `data/` (Data-Set-A, Data-Set-B), trích xuất ID bệnh nhân, chuyển đổi sang định dạng JSON và đẩy vào Kafka.
2. **Apache Kafka & Zookeeper:** Đóng vai trò là Message Broker. Nhận luồng dữ liệu streaming tốc độ cao từ NiFi (topic `icu_data`) và phân phối cho Spark.
3. **Apache Spark (Structured Streaming):** Đóng vai trò là Processing Layer. Nhận luồng dữ liệu từ Kafka, thực hiện làm sạch dữ liệu, áp dụng các mô hình Machine Learning phân tán thông qua **Pandas UDF** để đánh giá trạng thái Sepsis, và lưu trữ kết quả.
4. **Apache Cassandra:** Đóng vai trò là Storage Layer (NoSQL Database). Lưu trữ dữ liệu chuỗi thời gian (time-series) của từng bệnh nhân với tốc độ ghi/đọc cực cao.
5. **MLflow:** Đóng vai trò Model Registry & Tracking, quản lý vòng đời của mô hình Machine Learning.
6. **Flask API & Nginx (Load Balancer):** Backend Server đóng vai trò cân bằng tải và cung cấp các REST API cho Frontend truy xuất dữ liệu từ Cassandra.
7. **Web Dashboard:** Giao diện trực quan hóa dữ liệu (HTML/JS + Highcharts) cho phép các bác sĩ theo dõi chỉ số sinh tồn của bệnh nhân, chia các nhóm chỉ số sinh tồn theo Scale hợp lý và có hệ thống cảnh báo Sepsis Alert (dạng Step/Block) theo thời gian thực.

## 🚀 Yêu cầu hệ thống (Prerequisites)

- **Docker:** Phiên bản mới nhất (20.10+)
- **Docker Compose:** Phiên bản V2
- **Tài nguyên phần cứng:** Khuyến nghị ít nhất 8GB RAM (Tốt nhất là 16GB) và 4 Core CPU để cụm Big Data có thể hoạt động trơn tru.

## 🛠 Hướng dẫn Cài đặt & Khởi chạy (Running Instructions)

Toàn bộ hệ thống đã được đóng gói bằng Docker Compose, giúp việc triển khai vô cùng dễ dàng. Thực hiện các bước sau để khởi chạy:

### Bước 1: Khởi động hệ thống
Mở terminal tại thư mục gốc của dự án và chạy lệnh:
```bash
docker compose up -d --build
```
Lệnh này sẽ tải/build các Image và khởi chạy toàn bộ 18 container (Zookeeper, Kafka, Cassandra, NiFi, Spark Master/Workers, MLflow, API, Nginx...). Quá trình build lần đầu có thể mất vài phút.

### Bước 2: Kiểm tra trạng thái
Sử dụng lệnh sau để xem trạng thái các container:
```bash
docker compose ps
```
Hãy đảm bảo các container cấu hình một lần (như `cassandra-init`, `kafka-init`, và `nifi-setup`) báo trạng thái `Exited (0)` (setup thành công) và các dịch vụ khác báo trạng thái `Up`/`Healthy`.

### Bước 3: Truy cập các Dashboard & Services
Sau khi hệ thống khởi động hoàn tất (đợi khoảng 1-2 phút để cụm NiFi và Spark sẵn sàng khởi động pipeline), bạn có thể truy cập các giao diện sau trên trình duyệt:

- 📊 **Sepsis Web Dashboard:** [http://localhost:5000/dashboard](http://localhost:5000/dashboard) (Cổng Nginx Load Balancer)
- 💧 **Apache NiFi UI:** [https://localhost:8443/nifi](https://localhost:8443/nifi) (Bỏ qua cảnh báo SSL. User: `admin` / Pass: `adminpassword123`)
- ⚙️ **Spark Master UI:** [http://localhost:4040](http://localhost:4040) (Giao diện Spark theo dõi các Batch Streaming)
- 🧠 **MLflow UI:** [http://localhost:5001](http://localhost:5001)

### Bước 4: Tương tác với Dashboard
1. Truy cập vào **Sepsis Web Dashboard**.
2. Tại ô **Patient ID**, nhập mã bệnh nhân (Ví dụ: `p000001` - Thuộc `Data-Set-A`).
3. Nhấn **Apply** để tải dữ liệu, sau đó nhấn nút **▶ Demo Real-time** để giả lập biểu đồ chạy theo thời gian thực mô phỏng monitor tại bệnh viện.

## 🛑 Cách Dừng & Xóa hệ thống

Để dừng các container (Pause) mà không làm mất dữ liệu database/checkpoint (Lần sau bật lại sẽ tiếp tục xử lý):
```bash
docker compose stop
```

Để dừng và **xóa sạch** toàn bộ container, mạng lưới, và cả Volume dữ liệu (Cassandra, Checkpoints):
```bash
docker compose down -v
```
*(Lưu ý: Chạy lệnh này nếu bạn muốn reset lại hệ thống từ đầu sạch sẽ, NiFi sẽ tự quét lại và đọc lại dữ liệu từ file đầu tiên).*

## 📌 Một số lưu ý quan trọng (Troubleshooting & Notes)

- **Thứ tự nạp dữ liệu của NiFi (Queueing):** Thư mục `/data` bao gồm hai tập `Data-Set-A` (20.000 bệnh nhân) và `Data-Set-B` (20.000 bệnh nhân). Do tính chất đọc tuần tự, NiFi ListFile sẽ quét và đẩy toàn bộ 20.000 file của `Data-Set-A` (`p0...`) vào Queue trước. Spark Streaming sẽ mất hàng giờ để xử lý xong khối dữ liệu này trước khi tới lượt `Data-Set-B` (`p1...`).
- Vì vậy, nếu bạn tra cứu một ID từ `Data-Set-B` nhưng không thấy dữ liệu thì là do file đó đang "xếp hàng" ở hệ thống. Để ưu tiên test các ID này, hãy mang file đó ra đổi tên bắt đầu bằng `p00...` hoặc cho nó vào một thư mục ưu tiên đọc riêng.
