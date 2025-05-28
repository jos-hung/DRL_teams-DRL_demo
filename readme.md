
# 🎮 Game Falling Square (Kết hợp với Deep Q-Network)

Trò chơi này mô phỏng một hình vuông rơi tự do, người chơi điều khiển thanh đỡ (paddle) để đỡ hình vuông. Dự án cũng hỗ trợ mô phỏng hành động tự động từ AI sử dụng thuật toán Deep Q-Network (DQN).

---

## 📦 Yêu cầu cài đặt

Tạo môi trường Python và cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
pygame==2.6.0
numpy
torch
```

---

## 🚀 Chạy trò chơi

### Chạy với bàn phím (người chơi điều khiển):
```bash
python game_manual.py
```

### Chạy mô phỏng với DQN (AI điều khiển):
```bash
python trainer.py
```

---

## 🧪 Cấu trúc thư mục đề xuất

```
📂 game_falling_pad/
│
├── game_manual.py         # Trò chơi điều khiển bằng tay (bàn phím)
├── game_env.py            # Môi trường trò chơi để huấn luyện AI (dạng Gym-like)
├── dqn_agent.py           # Định nghĩa agent DQN và logic huấn luyện
├── trainer.py             # Tập tin chính để huấn luyện DQN
├── requirements.txt       # Danh sách thư viện cần cài
└── README.md              # Hướng dẫn và mô tả dự án
```

---

## 📄 Mô tả các tập tin

- `game_manual.py`: Phiên bản người chơi điều khiển thanh đỡ bằng phím trái/phải.
- `game_env.py`: Định nghĩa môi trường game dưới dạng tương tự như OpenAI Gym để AI có thể tương tác.
- `dqn_agent.py`: Cài đặt agent sử dụng Deep Q-Network, bao gồm replay buffer và cập nhật tham số.
- `trainer.py`: Tập tin chính dùng để khởi tạo môi trường, agent và tiến hành huấn luyện.
- `requirements.txt`: Chứa các thư viện Python cần thiết.
- `README.md`: Tài liệu mô tả dự án.

---

## 🤖 Huấn luyện AI

### Bước 1: Cài thư viện

```bash
pip install -r requirements.txt
```

### Bước 2: Chạy huấn luyện
```bash
python trainer.py
```

AI sẽ được huấn luyện để đỡ hình vuông bằng cách tối đa hóa điểm số.

### Bước 3: Theo dõi quá trình
- Kết quả huấn luyện sẽ được in ra terminal.
- Bạn có thể chỉnh sửa tham số như `epsilon`, `batch_size`, `gamma`, `learning_rate` trong `dqn_agent.py`.

---

## 🧠 Ghi chú

- Mô hình DQN sử dụng mạng neural đơn giản với `torch.nn`.
- Replay buffer và epsilon-greedy được tích hợp để đảm bảo sự ổn định khi học.
- Trò chơi không dùng đến bàn phím trong chế độ huấn luyện, thay vào đó mô phỏng các phím thông qua các danh sách `keys`.
- Mô hình này chưa được lưu nếu bạn thích bạn có thể lưu nó lại sau khi train nhằm cho các mục đích test sau đó của bạn.
- Nếu bạn muốn test thử độ chính xác của game, hãy mở comment # game_loop() ở file game.py và chạy python game.py để thử.

---

## 📫 Liên hệ

Nếu bạn có bất kỳ câu hỏi hay góp ý nào, đừng ngại mở issue hoặc liên hệ trực tiếp!
