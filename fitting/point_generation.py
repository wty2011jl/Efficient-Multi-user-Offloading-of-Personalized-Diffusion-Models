import torch
import clip
import lpips
from PIL import Image
from torchvision import transforms
from clip_metric import CLIP_EVAlUATION
#from image_generation import prompts
from lipis_metric import LIPIS_EVAL
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math


import numpy as np

prompt1 = "A photo of sks dog in the box"  #0-20
prompt2 = "A photo of sks dog running on the grass"
prompt3 = "A photo of sks dog on the sofa"
prompt4 = "A photo of sks dog next to the car"
prompt5 = "A photo of sks dog playing the ball"
prompt6 = "A photo of sks dog lying on a chair"
prompt7 = "A photo of two sks dogs chasing each other"
prompt8 = "A photo of sks dog sleeping on the bed"
prompt9 = "A photo of sks dog drinking water"
prompt10 = "A photo of sks dog crossing the road"

prompts = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8, prompt9, prompt10]


# generated_data = np.zeros((11, 10))
# similaritys = np.zeros((11, 10))
# differences = np.zeros((11, 10))
# #
# # pre_generated_data = np.load("generated_point.npy")
# #
# # generated_data[:10,:] = generated_data[:10,:] + pre_generated_data
# #
# for i in range(11):  # different spilt point
#
#     split_point = i * 20
#
#     for j in range(10): #different prompt
#
#         # CLIP
#         texts = prompts[j]  # 替换为你的文本列表
#
#         jj = j + 1
#
#         image_folder1 = "final_result1/%d/prompt%d" % (split_point, jj)  # 替换为你的图像文件夹路径
#         prompt1  = texts.replace("dog", "corgi puppy")
#         similarity  = CLIP_EVAlUATION(image_folder1,prompt1)
#         c1 = np.mean(similarity)
#
#         image_folder1 = "final_result2/%d/prompt%d" % (split_point, jj)  # 替换为你的图像文件夹路径
#         prompt1 = texts.replace("dog", "goldenretriever puppy")
#         similarity = CLIP_EVAlUATION(image_folder1, prompt1)
#         c2 = np.mean(similarity)
#
#         image_folder1 = "final_result3/%d/prompt%d" % (split_point, jj)  # 替换为你的图像文件夹路径
#         prompt1 = texts.replace("dog", "samoyed puppy")
#         similarity = CLIP_EVAlUATION(image_folder1, prompt1)
#         c3 = np.mean(similarity)
#
#         Simi =  np.mean([c1, c2, c3])
#
#
#
#         # LIPIS
#
#         for k in range(20):  # different generation
#
#             path1 = r"final_result1/%d/prompt%d/%d.png" % (split_point, (j+1), k)
#             path2 = r"final_result2/%d/prompt%d/%d.png" % (split_point, (j+1), k)
#             path3 = r"final_result3/%d/prompt%d/%d.png" % (split_point, (j+1), k)
#
#             diff =np.mean(LIPIS_EVAL(path1, path2, path3))
#
#         weighted_metric = Simi * diff
#
#         similaritys[i,j] = Simi
#         differences[i,j] = diff
#         generated_data[i,j] = weighted_metric
#
# np.save("generated_data", generated_data)
# np.save("difference", differences)
# np.save("similarity", similaritys)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 9,
}
difference = np.load("difference.npy")
similarity = np.load("similarity.npy")
print("difference", difference)
print("similarity", similarity)
generated_data =3*similarity * 1/(1 + np.exp(-30*(difference-0.1)) ) #similarity *
#generated_data = difference
reverse_generated_data = generated_data[::-1]  #offloaded denoising steps -> split point

def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))



xxx = np.arange(0, 201, 20)  #split point

vvv = np.mean(similarity, axis=1)

yyy =  np.mean(reverse_generated_data, axis=1)

popt, pcov = curve_fit(sigmoid, xxx, yyy, p0=[0.8, 40])

# 获取拟合的参数
a_fit, b_fit = popt
print(f"Fitted parameters: a = {a_fit}, b = {b_fit}")

y_fit = sigmoid(xxx, a_fit, b_fit)

x_ticks = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])

colors = plt.cm.get_cmap('tab20b', 10)  # 使用'tab10'颜色映射
fig=plt.figure(dpi=300,figsize=(5,4))
# 对每个x刻度，绘制对应的10个点，每个点不同颜色
for i in range(11):
    scatter = plt.scatter([x_ticks[i]]*10, reverse_generated_data[i], c=colors(np.arange(10)),s=5, alpha=0.7)

# 添加图例
# 为了在图例中显示不同颜色，使用空点生成图例
for j in range(10):
    plt.scatter([], [], c=colors(j), label=f'Prompt {j+1}',s=7, alpha=0.7)
plt.scatter(xxx, yyy, label='Mean value', color='blue', alpha=0.5)
plt.plot(xxx, y_fit, label='Fitted curve', color='#653126',linewidth =0.8, linestyle = "-.")
# 添加标签和标题
equation = r'$\frac{1}{{1 + \exp \left( { - a\left( {x - b} \right)} \right)}}$'
plt.text(120, 0.6, equation, fontsize=12, color='#653126',fontdict={'family': 'Times New Roman'})

plt.xlabel(r'Split point  $n^*$', fontproperties='Times New Roman',fontsize=11)
plt.ylabel('Personalized accuracy index', fontproperties='Times New Roman',fontsize=11)
#plt.title('Scatter Plot with Different Colors in Each Category')
plt.xticks(x_ticks,fontproperties='Times New Roman',fontsize= 11)  # 设置x轴刻度
plt.yticks(fontproperties='Times New Roman',fontsize=11)
plt.grid(True,  which='major', color = "lightgray", linestyle='--')
ax = plt.gca()
ax.spines['top'].set_visible(False)    # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框
ax.set_facecolor('whitesmoke')
plt.legend(loc='lower left', prop=font1)  # 图例右上角 bbox_to_anchor=(1.15, 1), title="Point Colors"
plt.show()









