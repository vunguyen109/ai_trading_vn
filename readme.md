# AI Trading VN — Project Blueprint

Một bản tóm tắt và roadmap cho dự án AI hỗ trợ quyết định giao dịch (decision support system) dành cho nhà đầu tư cá nhân.

## Mục lục

- [Tóm tắt](#tóm-tắt)
- [Tính năng chính](#tính-năng-chính)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Roadmap theo tuần](#roadmap-theo-tuần)
- [Bắt đầu nhanh](#bắt-đầu-nhanh)
- [Cấu hình mẫu (config.yaml)](#cấu-hình-mẫu-configyaml)
- [Mẫu STATUS.txt](#mẫu-statustxt)
- [Góp phần & Liên hệ](#góp-phần--liên-hệ)
- [License](#license)

## Tóm tắt

Mục tiêu: xây dựng hệ thống hỗ trợ quyết định cho nhà đầu tư cá nhân — dự đoán xu hướng giá ngắn hạn (1–5 ngày), gửi cảnh báo (Telegram), cung cấp công cụ backtest và quản lý rủi ro. Hệ thống không thực hiện giao dịch tự động.

## Tính năng chính

- Dự đoán hướng giá ngắn hạn (1–5 ngày)
- Pipeline thu thập và làm sạch dữ liệu
- Tạo feature kỹ thuật và sentiment
- Mô-đun huấn luyện và lưu model
- Backtest với chi phí giao dịch và slippage
- Gửi cảnh báo qua Telegram, dashboard đơn giản

## Cấu trúc dự án

```
ai_trading_vn/
│
├── 📂 data/                    # MODULE 1
│   ├── collector.py           # Thu thập dữ liệu
│   ├── cleaner.py             # Xử lý missing data
│   └── database.db            # SQLite database
│
├── 📂 features/               # MODULE 2  
│   ├── technical.py           # Chỉ báo kỹ thuật
│   ├── fundamental.py         # Dữ liệu cơ bản (nếu có)
│   └── sentiment.py           # Phân tích tin tức
│
├── 📂 models/                 # MODULE 3
│   ├── trainer.py             # Training pipeline
│   ├── predictor.py           # Dự đoán
│   └── trained_models/        # Lưu model
│
├── 📂 backtest/               # MODULE 4
│   ├── engine.py              # Backtesting engine
│   ├── strategies.py          # Các chiến lược
│   └── results/               # Kết quả backtest
│
├── 📂 alerts/                 # MODULE 5
│   ├── telegram_bot.py        # Gửi cảnh báo
│   └── dashboard.py           # Giao diện web đơn giản
│
├── config.yaml               # Cấu hình hệ thống
├── requirements.txt          # Thư viện cần cài
├── main.py                  # Chạy toàn bộ hệ thống
└── STATUS.txt               # FILE QUAN TRỌNG: Trạng thái hiện tại
```

## Roadmap theo tuần

- Tuần 1 — Data Foundation
  - Thiết lập môi trường, viết `data/collector.py`, `data/cleaner.py`, lưu vào SQLite
- Tuần 2 — Feature Engineering
  - Xây các chỉ báo kỹ thuật (RSI, MACD, SMA...), transformations và lag-features
- Tuần 3 — Machine Learning Core
  - Xác định target, walk-forward split, huấn luyện RandomForest và đánh giá
- Tuần 4 — Backtesting System
  - Tích hợp tín hiệu ML vào backtest, thêm commission/slippage, tính Sharpe/MDD
- Tuần 5 — Alerts & Deployment
  - Tạo Telegram bot, kết nối dự đoán, lên lịch chạy định kỳ, dashboard

## Bắt đầu nhanh

1. Chuẩn bị Python 3.10+ và virtual environment

2. Cài dependencies (ví dụ cơ bản):

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

3. Tạo `STATUS.txt` từ mẫu và cập nhật `config.yaml`

4. Chạy thử collector (ví dụ):

```bash
python data/collector.py --symbols VNINDEX,VIC
```

## Cấu hình mẫu (config.yaml)

Ví dụ cấu hình (rút gọn):

```yaml
system:
  version: "1.0"
  mode: "development"

data:
  symbols: ["VNINDEX","VIC","VHM"]
  source: "yfinance"
  update_frequency: "daily"

ml:
  model_type: "random_forest"
  target: "price_up_next_3_days"

backtest:
  initial_capital: 1000000000
  commission: 0.0015
  slippage: 0.001

alerts:
  telegram:
    enabled: false
    bot_token: "YOUR_TOKEN"
    chat_id: "YOUR_CHAT_ID"
```

## Mẫu STATUS.txt

Sao chép nội dung sau vào `STATUS.txt` và cập nhật khi làm việc:

```
=== AI TRADING SYSTEM STATUS ===
PROJECT: ai_trading_vn
LAST UPDATED: [YYYY-MM-DD]

CURRENT MODULE: [1/2/3/4/5]
MODULE STATUS: [NOT STARTED / IN PROGRESS / COMPLETED]

RECENT WORK:
- [Công việc vừa làm xong]

NEXT TASK:
- [Công việc tiếp theo]

ERRORS/ISSUES:
- [Lỗi đang gặp]

CONFIG SNAPSHOT:
- Python: 3.10
- OS: Windows
```

## Góp phần & Liên hệ

- Góp phần: mở Pull Request với mô tả chi tiết và cập nhật `STATUS.txt`.
- Liên hệ: tạo issue hoặc liên lạc trong README (nếu muốn thêm thông tin liên hệ).

## License

Tùy chọn license — nếu chưa quyết định, thêm file `LICENSE` sau.

---

Nếu bạn muốn, tôi có thể:

- Tạo `requirements.txt` mẫu
- Viết `data/collector.py` mẫu để lấy dữ liệu VN30
- Tạo `STATUS.txt` tự động

Cho tôi biết bước bạn muốn tiếp theo.