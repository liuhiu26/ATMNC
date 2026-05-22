import torch
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from model import SimpleCNN

print("--- KHỞI ĐỘNG KIỂM TRA LỖI ẨN ---")

try:
    # 1. Kiểm tra mô hình và file trọng số
    print("1. Đang kiểm tra cấu trúc model và file baseline.pth...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('baseline.pth', map_location=device))
    model.eval()
    print("=> Khởi tạo Model THÀNH CÔNG!")

    # 2. Kiểm tra Camera
    print("\n2. Đang kiểm tra kết nối Webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("=> LỖI: Hệ thống không mở được Webcam (Có thể bị app khác chiếm)!")
    else:
        ret, frame = cap.read()
        if ret:
            print("=> Kết nối và đọc Frame từ Webcam THÀNH CÔNG!")
            cv2.imshow('Test Camera', frame)
            print("Đang hiện cửa sổ test. Nhấn một phím bất kỳ tại cửa sổ ảnh để tắt...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("=> LỖI: Camera bật được nhưng không đọc được dữ liệu ảnh!")
        cap.release()

except Exception as e:
    print("\n[PHÁT HIỆN LỖI CHÍNH LÀM SẬP LUỒNG]:")
    print(f"Xảy ra lỗi: {e}")