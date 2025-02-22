import os
from PIL import Image
import numpy as np
import torch
import clip
import matplotlib.pyplot as plt

def CLIP_EVAlUATION(your_image_folder, your_texts):

    # 加载 CLIP 模型
    model, preprocess = clip.load("ViT-B/32")
    # print(model)
    model.cuda().eval()

    # 准备你的图像和文本
    # your_image_folder = "final_result3/0"  # 替换为你的图像文件夹路径
    # your_texts = ["A samoyed puppy in the box"]  # 替换为你的文本列表

    images = []
    for filename in os.listdir(your_image_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            path = os.path.join(your_image_folder, filename)
            image = Image.open(path).convert("RGB")
            images.append(preprocess(image))

    # 图像和文本预处理
    image_input = torch.tensor(np.stack(images)).cuda()
    text_tokens = clip.tokenize(your_texts).cuda()

    # 计算特征
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    # 计算相似度
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (text_features.cpu().numpy() @ image_features.cpu().numpy().T)

    return similarity


#print('similarity:',similarity)

# # 可视化相似度
# plt.imshow(similarity, cmap="hot", interpolation="nearest")
# plt.colorbar()
# plt.xlabel("Images")
# plt.ylabel("Texts")
# plt.title("Similarity between Texts and Images")
# plt.xticks(range(len(images)), range(len(images)), rotation=90)
# plt.yticks(range(len(your_texts)), your_texts, rotation='vertical', va='center')  # 设置标签竖向显示并居中
#
# # 保存图像到文件
# plt.savefig("similarity.png", bbox_inches='tight')  # 确保所有内容都在保存的图片里
#
# # 显示图像
# plt.show()
