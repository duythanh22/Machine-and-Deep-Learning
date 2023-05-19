
### **Một số vấn đề về học sâu**

**1. Học sâu, học máy, trí tuệ nhân tạo và mạng nơ-ron**

Lịch sử phát triển của trí tuệ nhân tạo và các lĩnh vực liên quan:
- Để dự đoán tương lai, cần phải có "kinh nghiệm sâu sắc" trong một lĩnh vực cụ thể.
- Kinh nghiệm và kiến thức không thể được đạt được qua một đêm.
- Dựa vào các phép tính xác suất, phù hợp với mô hình (pattern fit), các nhà toán học đã tạo ra nhiều mô hình toán học (hồi quy tuyến tính, mô hình Bayes, v.v.), các mô hình này có thể học được một số quy luật từ dữ liệu lịch sử và dựa vào những quy luật này để dự đoán.
- Các bước tính toán hoặc phương pháp tính toán có khả năng dự đoán hoặc đạt được một kết quả cụ thể dựa trên dữ liệu lịch sử được gọi là thuật toán.
- Các nhà khoa học máy tính kết hợp máy tính với toán học để thực hiện thuật toán theo quy mô, nâng cao hiệu suất "học" và "dự đoán".
- Các nhà triết học và nhà tâm lý học đã tái tạo cơ chế suy nghĩ của bộ não, tạo ra mạng nơ-ron nhân tạo (ANN, năm 1943), để tạo ra trí tuệ giống con người.
- Năm 1950, ý tưởng "Phép thử Turing" được đề xuất.
- Năm 1956, ý tưởng "Trí tuệ nhân tạo" và "Học máy" được đề xuất.
- Do hiệu suất dự đoán kém, yêu cầu dữ liệu lớn, thời gian tính toán lâu và không thể đạt được trình độ trí tuệ của con người, mạng nơ-ron đã gặp một thời kì khó khăn.
- Vào thế kỷ mới, điều kiện ứng dụng quy mô lớn của các thuật toán mạng nơ-ron dần phát triển hơn:
  - Thuật toán mạng nơ-ron đã có những bước tiến lớn.
  - Lượng dữ liệu toàn cầu tăng mạnh - bùng nổ dữ liệu.
  - Công nghệ chip, công nghệ đám mây phát triển nhanh chóng, nâng cao khả năng tính toán.
- Năm 2016, ý tưởng "Học sâu" được đề xuất.

  Tóm lại, trí tuệ nhân tạo bao gồm học máy, học máy bao gồm học sâu.
![image](https://github.com/duythanh22/Machine-and-Deep-Learning/assets/84120300/28324c8c-bda9-4969-a57d-db5ac33350f5)

**2. Các ứng dụng phổ biến của trí tuệ nhân tạo, học sâu:**
  - Khôi phục màu, hình ảnh.
  - Nhận diện đối tượng
  - Tạo hình với trí tuệ nhân tạo
  - Dịch máy
  - Hiệu ứng động với trí tuệ nhân tạo
  - Công nghệ tăng cường video bằng trí tuệ nhân tạo
  - Công nghệ tăng cường hình ảnh bằng trí tuệ nhân tạo
  - Mô hình ngôn ngữ

**3. Các khái niệm cơ bản:**
  
  Các khái niệm trong học sâu khác biệt so với học máy truyền thống:
  
  - *Mẫu, đặc trưng và nhãn (Samples, features and labels):*
    + Mẫu: Trong các phép toán với tensor nhiều chiều, thường không phân biệt "hàng" và "cột" nữa, mà xem mỗi chỉ số tương ứng là một mẫu.
    + Đặc trưng: Thông tin chứa trong tensor chính là đặc trưng, và tensor chứa đặc trưng thường được gọi là tensor đặc trưng.
    + Nhãn: Thường được tách riêng khỏi tập dữ liệu.
  - *Phân loại và hồi quy:*
  
    Nhãn là một khái niệm rất quan trọng trong học máy. Các nhãn khác nhau chỉ ra các vấn đề khác nhau, với hai loại phổ biến:
      + Phân loại: Nhãn có số lượng hữu hạn và loại trừ lẫn nhau, biểu thị bằng biến rời rạc (Categorical).
      + Hồi quy: Kết quả đầu ra của mô hình là một con số cụ thể, nhãn là một số thực, biểu thị bằng biến liên tục (Continuous).
    
  - *Học có giám sát và học không giám sát:*
    + Học có giám sát: Các tác vụ được gắn nhãn, học từ dữ liệu  đã biết, sau đó dự đoán những gì bạn muốn biết, chẳng hạn như: KNN, cây quyết định, máy vectơ hỗ trợ, hồi quy tuyến tính, hồi quy logistic và hầu hết các mạng nơ-ron, ...
    + Học không giám sát: các tác vụ không được gắn nhãn, thường được sử dụng làm thuật toán hỗ trợ để nâng cao hiệu quả học của các thuật toán có giám sát, bao gồm phân cụm, lọc cộng tác, ...
    + Học bán giám sát
    + Học tăng cường
 
  - Tiêu chí đánh giá mô hình:
    + Hiệu quả dự đoán của mô hình: Hiệu quả dự đoán/đánh giá của mô hình là mục tiêu chính, đối với các thuật toán khác nhau, có các chỉ số đánh giá mô hình là khác nhau, chúng ta sử dụng các chỉ số đánh giá này để đo lường hiệu quả dự đoán của mô hình.
    
    + Tốc độ tính toán: Có khả năng xử lý đồng thời lượng lớn dữ liệu, học nhanh chóng trong thời gian ngắn và thực hiện dự đoán thời gian thực là một ưu điểm quan trọng của học máy. Nếu tốc độ tính toán của thuật toán quá chậm, nó cũng không thuận lợi cho việc điều chỉnh và thử nghiệm, đồng thời có thể yêu cầu nhiều tài nguyên tính toán và lưu trữ hơn, tạo ra chi phí cao hơn. Trong trường hợp mô hình cho kết quả tốt, đảm bảo tốc độ tính toán nhanh là một yếu tố quan trọng trong học máy.
   
    + Khả năng giải thích: Cần giải thích kết quả dự đoán của thuật toán cho mọi người, nếu không, các bên liên quan sẽ không chấp nhận. Yêu cầu về khả năng giải thích của mô hình đối với học sâu thấp hơn.
    
    + Phục vụ cho doanh nghiệp: Chỉ khi phục vụ cho doanh nghiệp hoặc nghiên cứu đẩy mạnh sự nhận thức của con người, thuật toán mới có giá trị thương mại.

**4. Pytorch framework:**
  - Ưu điểm:
    - Hỗ trợ tính toán nhanh cho dữ liệu lớn và mạng nơ-ron lớn.
    - Tính linh hoạt cao, có khả năng khai thác tiềm năng của mạng nơ-ron. Kết hợp sự linh hoạt với cú pháp Python đơn giản và dễ học.
    - Hỗ trợ chuyển đổi mượt mà giữa môi trường nghiên cứu và môi trường sản xuất với chi phí gỡ lỗi thấp.
  - Kiến trúc:
  
    *Thư viện gốc Torch - Cung cấp các mô-đun để xây dựng mạng nơ-ron linh hoạt.*

    - Các thành phần cơ bản trong quá trình chạy:
      + Tensor
      + Autograd
    - Tiện ích xử lý trước dữ liệu:
      + Xử lý và nhập liệu dữ liệu (data, datasets)
      + Hiển thị trực quan bằng TensorBoard
      + Mô hình được huấn luyện trước (model_zoo)
    - Các yếu tố cơ bản của mạng nơ-ron:
      + nn: Mô-đun mạng nơ-ron
      + Các lớp đa dạng (Module)
      + Hàm mất mát và hàm kích hoạt (functional)
    - Thuật toán tối ưu (optim)
    - Hiệu suất tính toán:
      + Huấn luyện phân tán (torchelastic)
      + Huấn luyện GPU (cuda)
    - Triển khai môi trường sản xuất (JIT)

    *Các mô-đun hỗ trợ ứng dụng cụ thể trong lĩnh vực Trí tuệ Nhân tạo đã được phát triển:*
    - Thị giác máy tính (torchvision)
        + Các tập dữ liệu phổ biến (datasets)
        +  Các mô hình phổ biến (models)
        +  Xử lý trước dữ liệu hình ảnh (transform)
    - Xử lý ngôn ngữ tự nhiên (torchtext)
        + Xử lý trước dữ liệu (data)
        + Các tập dữ liệu phổ biến (datasets)
    - Xử lý âm thanh (torchaudio)
        + Các tập dữ liệu phổ biến (datasets)
        + Xử lý trước dữ liệu âm thanh (transform)
       +  Các mô hình phổ biến (models)
        + Các hàm phổ biến (functional)


