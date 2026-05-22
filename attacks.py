import torch
import torch.nn.functional as F

def fgsm_attack(image, epsilon, data_grad):
    # Lấy dấu của gradient (hướng làm tăng sai số của AI)
    sign_data_grad = data_grad.sign()
    
    # Tạo ảnh nhiễu: Ảnh gốc + (epsilon * hướng sai số)
    perturbed_image = image + epsilon * sign_data_grad
    
    # Ép giá trị pixel về lại khoảng [-1, 1] theo chuẩn hóa của CIFAR-10
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    
    return perturbed_image

def pgd_attack(model, image, label, epsilon, alpha=0.01, iters=10):
    original_image = image.clone().detach()
    perturbed_image = image.clone().detach()
    
    for _ in range(iters):
        perturbed_image.requires_grad = True
        outputs = model(perturbed_image)
        model.zero_grad()
        
        loss = F.cross_entropy(outputs, label)
        loss.backward()
        
        data_grad = perturbed_image.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_image = perturbed_image + alpha * sign_data_grad
        
        # Giới hạn nhiễu không vượt quá khoảng epsilon so với ảnh gốc
        eta = torch.clamp(perturbed_image - original_image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(original_image + eta, -1, 1)
        perturbed_image = perturbed_image.detach()
        
    return perturbed_image