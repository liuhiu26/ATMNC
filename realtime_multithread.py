import cv2
import torch
import threading
import queue
import time
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from model import SimpleCNN
from attacks import fgsm_attack

# Định nghĩa các lớp phân loại toàn cục
classes = ('may bay', 'o to', 'chim', 'meo', 'huou', 'cho', 'ech', 'ngua', 'tau thuy', 'xe tai')

# ================= THREAD 1: BẮT LUỒNG FRAME TỪ CAMERA =================
def thread_capture(frame_queue):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("LỖI: Không mở được Webcam!")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.full():
            frame_queue.get() 
        frame_queue.put(frame)
    cap.release()

# ================= THREAD 3: PHÂN LOẠI & SINH NHIỄU ĐỐI KHÁNG =================
def thread_classify(frame_queue, result_queue, model, device, transform, state_dict):
    # Đảm bảo trạng thái điều khiển tấn công qua phím bấm
    global ATTACK_MODE, EPSILON, CONFIDENCE_THRESHOLD
    
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            
            # Tiền xử lý dữ liệu ảnh
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tensor = transform(img).unsqueeze(0).to(device)
            img_tensor.requires_grad = True 
            
            start_time = time.time()
            
            # Dự đoán ảnh sạch ban đầu
            outputs = model(img_tensor)
            probs = F.softmax(outputs[0], dim=0)
            max_prob, predicted_label = torch.max(probs, 0)
            
            # Kiểm tra trạng thái kích hoạt tấn công
            if ATTACK_MODE:
                model.zero_grad()
                loss = F.cross_entropy(outputs, predicted_label.unsqueeze(0))
                loss.backward()
                
                data_grad = img_tensor.grad.data
                img_tensor_adv = fgsm_attack(img_tensor, EPSILON, data_grad)
                
                outputs_adv = model(img_tensor_adv)
                probs_adv = F.softmax(outputs_adv[0], dim=0)
                max_prob, predicted_label = torch.max(probs_adv, 0)
                
                status_text = f"ATTACKED (Eps={EPSILON})"
                color = (0, 0, 255) 
            else:
                status_text = "CLEAN"
                color = (0, 255, 0) 

            latency = (time.time() - start_time) * 1000

            if max_prob.item() >= CONFIDENCE_THRESHOLD:
                label_text = f"{classes[predicted_label.item()]} ({max_prob.item()*100:.1f}%)"
            else:
                label_text = "AI k doan duoc"

            if result_queue.full():
                result_queue.get()
            result_queue.put((frame, label_text, latency, status_text, color))
        else:
            time.sleep(0.01) # Chờ một chút nếu hàng đợi trống để giảm tải CPU

# ================= LUỒNG CHÍNH (MAIN) =================
if __name__ == '__main__':
    print("Khởi động hệ thống Pipeline Đa Luồng bảo mật...")
    
    # Cấu hình biến toàn cục kiểm soát trạng thái
    global ATTACK_MODE, EPSILON, CONFIDENCE_THRESHOLD
    ATTACK_MODE = False
    EPSILON = 0.1
    CONFIDENCE_THRESHOLD = 0.5

    # Đưa việc khởi tạo tài nguyên phần cứng vào hẳn bên trong main chống sập luồng
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform_setup = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    model_instance = SimpleCNN().to(device)
    model_instance.load_state_dict(torch.load('baseline.pth', map_location=device))
    model_instance.eval()

    # Tạo Hàng đợi cục bộ
    frame_queue = queue.Queue(maxsize=2)   
    result_queue = queue.Queue(maxsize=2)  

    # Khởi chạy luồng và truyền tham số an toàn
    t1 = threading.Thread(target=thread_capture, args=(frame_queue,), daemon=True)
    t3 = threading.Thread(target=thread_classify, args=(frame_queue, result_queue, model_instance, device, transform_setup, classes), daemon=True)
    
    t1.start()
    t3.start()

    print("Hệ thống đa luồng đang chạy ổn định. Đang mở Camera màn hình...")

    while True:
        if not result_queue.empty():
            frame, label, latency, status_text, color = result_queue.get()
            
            # Ghi thông số lên giao diện camera
            cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Latency: {latency:.1f} ms", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Mode: {status_text}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow('ATMNC - Pipeline Da Luong', frame)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('a'): 
            ATTACK_MODE = not ATTACK_MODE
            print(f"Trạng thái tấn công đối kháng: {ATTACK_MODE}")

    cv2.destroyAllWindows()