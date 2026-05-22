import torch
import cv2
import torch.nn.functional as F
from model import SimpleCNN
import torchvision.transforms as transforms
from PIL import Image

# 1. Cấu hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ('may bay', 'o to', 'chim', 'meo', 'huou', 'cho', 'ech', 'ngua', 'tau thuy', 'xe tai')

# NGƯỠNG TỰ TIN (Điều chỉnh con số này từ 0.0 đến 1.0)
# 0.6 nghĩa là AI phải chắc chắn > 60% thì mới dám chốt kết quả
CONFIDENCE_THRESHOLD = 0.6 

# 2. Load mô hình đã train
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('baseline.pth', map_location=device))
model.eval() 

# 3. Định nghĩa tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 4. Bật Webcam
cap = cv2.VideoCapture(0)
print("Dang bat Camera... Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Tiền xử lý
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tensor = transform(img).unsqueeze(0).to(device)

    # AI dự đoán
    with torch.no_grad():
        outputs = model(img_tensor)
        
        # DÙNG SOFTMAX ĐỂ CHUYỂN ĐẦU RA THÀNH PHẦN TRĂM (0 -> 1)
        probabilities = F.softmax(outputs[0], dim=0)
        
        # Lấy giá trị % cao nhất và vị trí của nó
        max_prob, predicted = torch.max(probabilities, 0)
        
        # Kiểm tra xem có vượt qua ngưỡng tự tin không
        if max_prob.item() >= CONFIDENCE_THRESHOLD:
            # Nếu chắc chắn, in ra tên + số %
            label = f"{classes[predicted.item()]} ({max_prob.item()*100:.1f}%)"
            color = (0, 255, 0) # Màu xanh lá
        else:
            # Nếu không chắc chắn
            label = "AI k doan duoc"
            color = (0, 0, 255) # Màu đỏ

    # Hiển thị kết quả lên màn hình
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('He thong nhan dien ATM - Hieu & Manh', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()