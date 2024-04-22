import Diffusion_Model

from PIL import Image
import torchvision.transforms as transforms
import torch

def load_image(image_path):
    image = Image.open(image_path)
    return image

image_path = 'BG.jpg'  # 图像文件路径
image = load_image(image_path)

def preprocess_image(image, size=(2488, 1400)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image)

preprocessed_image = preprocess_image(image)

print(preprocessed_image.shape)
exit(0)

device = "cpu"
state = torch.randn(256, 11).to(device)
model = Diffusion_Model.Diffusion(loss_type="l2", obs_dim=11, act_dim=256, hidden_dim=256, T=100, device=device)

# 假设模型和损失函数已经定义好
# 这里使用一个简化的训练循环示例
def train_model(model, image, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model.loss(image, state)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


# 调用训练函数
train_model(model, preprocessed_image)

def sample_image(model, state=None):
    model.eval()
    with torch.no_grad():
        sampled_image = model.sample(state)
    return sampled_image

sampled_image = sample_image(model)
