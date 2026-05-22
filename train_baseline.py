import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import SimpleCNN 

# Chuyển phần này vào trong hàm main để tránh lỗi đa tiến trình trên Windows
def main():
    # 1. Cấu hình thiết bị
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Dang chay tren: {device}")

    # 2. Tiền xử lý
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 3. Load dữ liệu
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    # Giảm num_workers xuống 0 nếu máy bạn vẫn báo lỗi này, nhưng thường bọc main là hết.
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    # 4. Khởi tạo mô hình
    net = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 5. Huấn luyện
    print("Bat dau Train 50 vong...")
    for epoch in range(50):  
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[Vong {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
        
        scheduler.step()

    print('Huan luyen xong!')
    torch.save(net.state_dict(), 'baseline.pth')
    print("Da luu bo nao vao baseline.pth")

if __name__ == '__main__':
    main()