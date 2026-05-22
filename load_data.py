import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. Định nghĩa tiền xử lý ảnh (Transform)
# Chuyển ảnh PIL về dạng Tensor mà PyTorch hiểu được, đồng thời Normalize (chuẩn hóa) dữ liệu.
# Bước chuẩn hóa này rất quan trọng để model hội tụ nhanh khi train.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

# 2. Tải tập Huấn luyện (Train set)
# root='./data': Tạo thư mục tên 'data' ở cùng nơi để code và lưu data vào đó.
# Khi chạy lần 2, nó sẽ tự quét thấy file và không tải lại nữa.
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# DataLoader: Giúp băm dữ liệu thành các cục nhỏ (batch) thay vì ném cả 50.000 ảnh vào RAM cùng lúc.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

# 3. Tải tập Kiểm thử (Test set)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

# 10 nhãn mặc định của CIFAR-10
classes = ('máy bay', 'ô tô', 'chim', 'mèo',
           'hươu', 'chó', 'ếch', 'ngựa', 'tàu thủy', 'xe tải')

print(f"Đã tải xong! Tập train có {len(trainset)} ảnh, tập test có {len(testset)} ảnh.")

# --- ĐOẠN CODE TEST THỬ XEM ẢNH CÓ LÊN KHÔNG ---
def imshow(img):
    img = img / 2 + 0.5     # de-normalize để hiển thị
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    # Lấy ngẫu nhiên 1 batch ảnh train
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Hiển thị ảnh và nhãn
    imshow(torchvision.utils.make_grid(images[:4])) # Show 4 ảnh đầu tiên
    print('Nhãn tương ứng: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))