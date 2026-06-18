import os
import shutil
import glob

DATA_DIR = "/home/leduc1009/BIGDATA_FINAL/data"
TRAIN_DIR = os.path.join(DATA_DIR, "train_data")
LIVE_DIR = os.path.join(DATA_DIR, "live_data")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(LIVE_DIR, exist_ok=True)

# Lấy toàn bộ file từ Data-Set-A và Data-Set-B
all_files = []
all_files.extend(glob.glob(os.path.join(DATA_DIR, "Data-Set-A", "*.psv")))
all_files.extend(glob.glob(os.path.join(DATA_DIR, "Data-Set-B", "*.psv")))

# Sắp xếp để có tính ổn định
all_files.sort()

total_files = len(all_files)
print(f"Tổng số files tìm thấy: {total_files}")

# Đưa 10,000 files cuối vào live_data, phần còn lại vào train_data
train_files = all_files[:-10000] if total_files > 10000 else []
live_files = all_files[-10000:] if total_files > 10000 else all_files

print(f"Chuyển {len(train_files)} files sang train_data...")
for f in train_files:
    shutil.move(f, os.path.join(TRAIN_DIR, os.path.basename(f)))

print(f"Chuyển {len(live_files)} files sang live_data...")
for f in live_files:
    shutil.move(f, os.path.join(LIVE_DIR, os.path.basename(f)))

print("Hoàn tất mô phỏng dữ liệu!")
