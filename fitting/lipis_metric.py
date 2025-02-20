
import torch
import lpips
from PIL import Image
from torchvision import transforms

# 设置设备 (GPU 或 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 LPIPS 模型并移动到设备
loss_fn = lpips.LPIPS(net='alex').to(device)

# 定义图像预处理函数
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小以匹配LPIPS模型要求
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # 添加batch维度并移动到设备
    return image

# 加载三张图片
image1 = load_image("final_result3/160/0.png")
image2 = load_image("final_result2/160/0.png")
image3 = load_image("final_result1/160/0.png")

# 计算两两之间的LPIPS差异
with torch.no_grad():
    dist12 = loss_fn(image1, image2).item()
    dist13 = loss_fn(image1, image3).item()
    dist23 = loss_fn(image2, image3).item()

# 输出结果
print(f"LPIPS Difference between Image 1 and Image 2: {dist12:.4f}")
print(f"LPIPS Difference between Image 1 and Image 3: {dist13:.4f}")
print(f"LPIPS Difference between Image 2 and Image 3: {dist23:.4f}")

print("mean", dist12+dist13+dist23)
