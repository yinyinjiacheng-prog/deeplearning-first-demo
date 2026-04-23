#------------------------步骤1：导入需要的库---------------------------------
import torch #深度学习框架
import torch.nn as nn #神经网络层
import numpy as np #数值计算
from PIL import Image #图片读取/保存
import matplotlib.pyplot as plt #画图显示结果

#------------------------步骤2：设置基本参数---------------------------------
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100

#-----------------------步骤3：创建一张"目标图片”给模型学习-------------------
#我们造一张渐变图：红随x变，绿随y变，蓝固定
def create_gradient_image():
    # def = 定义一个功能，打包一段代码
    # 创建一张全黑的图(高，宽，RGB3通道)
    img = torch.zeros(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    #我们来解析一下这段代码:(高度，宽度， 颜色通道)
    #IMAGE_HEIGHT:图片有多少行(纵向); IMAGE_WIDTH:图片有多少列(横向)
    #RGB三个颜色通道:
    #0:红色(RED); 绿色(Green); 蓝色(Blue)
    #在图片中:y = 第几行(纵向，从上到下)；x = 第几列(横向，从左到右)

    # 遍历每一个像素(x=横向 y=纵向)
    for y in range(IMAGE_HEIGHT):
        for x in range(IMAGE_WIDTH):
            red = x / IMAGE_WIDTH #越往右越红
            green = y / IMAGE_HEIGHT #越往下越绿
            blue = 0.7 #蓝色固定
            img[y, x] = torch.tensor([red, green, blue])
    return img

target_image = create_gradient_image()

#-----------------------步骤4: 准备"坐标"和"颜色"给AI学----------------------
coordinates = []  # 存放所有像素坐标(x,y)
colors = []  # 存放每个坐标对应的颜色(R,G,B)

for y in range(IMAGE_HEIGHT):
    for x  in range(IMAGE_WIDTH):
        # 把坐标缩成0-1之间(AI喜欢这种格式)
        # 位置归一化，和颜色深浅无关
        coordinates.append([x / IMAGE_WIDTH, y / IMAGE_HEIGHT])
        # 取出这个坐标的颜色
        colors.append(target_image[y, x].tolist())
        # .tolist() = 转成普通列表,把一个“数组格式的颜色值”-->转成普通的python列表
        # target_image[y, x]
        # 拿到这个像素的颜色，但它是numpy数组
        # .tolist()
        # 把它转成普通列表，变成我们熟悉的RGB颜色格式
        # colors.append()
        # 把这个颜色存进列表里，转成列表 = 取出了干净、标准、可用的RGB颜色值
        # coordinates是输入， colors是输出；
        # 整个模型，就是在学什么位置-->应该对应什么颜色

# 把数据转成AI能看懂的格式(tensor)
# float32:32位浮点数
coordinates = torch.tensor(coordinates, dtype=torch.float32)
colors = torch.tensor(colors, dtype=torch.float32)

#----------------------------步骤5：搭建AI神经网络----------------------------
# class = 造一个AI大脑模板
class ColorLearningAI(nn.Module):
    def __init__(self):
        super().__init__()


        # Sequential = 按顺序搭建大脑层
        self.brain = nn.Sequential(
            #Linear = 全连接层(最简单的大脑细胞层)
            #输入2个数，坐标[x, y]
            #输出64个数:让大脑记住64个特征
            #把坐标传给64个神经元，开始学习
            nn.Linear(2, 64),
            nn.ReLU(),
            #激活函数，让AI能学复杂的东西
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            #输出3个数字，这三个数是:[R, G, B]颜色
            nn.Sigmoid()


        )
    # forward = AI思考的过程
    def forward(self, x):
        return self.brain(x)
    
# 创建AI模型
ai_model = ColorLearningAI()
    
#----------------------步骤6:设置训练工具---------------------
optimizer = torch.optim.Adam(ai_model.parameters(), lr=0.01)
    #torch.optim.Adam 深度学习优化器，全称Adaptive Moment Estimation(自适应矩估计)
    #1. 计算参数的梯度(知道该往哪个方向改参数)
    #2. 用Adam算法自适应调整更新步长
    #3. 更新模型参数
    # Adam:它是SGD（随机梯度下降）的升级版，结合了两个优秀优化器的优点：
    #AdaGrad：自适应学习率
    #RMSProp：平滑梯度、处理非平稳目标
    #Adam 的核心优势（新手必记）
    #自动调整学习率 → 不用像 SGD 那样反复调参
    #收敛速度极快 → 训练更快达到好效果
    #对超参数不敏感 → 就算 lr 设得不太完美，也能训得动
    #适用于绝大多数任务：分类、检测、NLP、生成模型…
loss_function = nn.MSELoss()

#-----------------------步骤7:开始训练AI(深度学习)--------------
print("===== AI开始学习图片 =====")

    # step:步数/迭代次数
for step in range(8000):
    optimizer.zero_grad()  #清空上一次的错题
    predict = ai_model(coordinates)  # AI看图答题
    loss = loss_function(predict, colors)  #打分
    loss.backward()
    #把最后的误差(loss)往回算，算出模型里每个参数"应该往哪个方向调、调多少"
    optimizer.step()

    # 每500步输出一次分数
    if step % 500 == 0:
        print(f"训练步数：{step}   |    AI错误分数:{loss.item():.6f}")

print("===== AI学习完成 =====")


# ----------------------步骤8:让AI画出它学会的图----------------
# with = 开启一段"特殊模式"
# [[1,2,3,4], [5,6,7,8]]是2行4列的矩阵/表格
with torch.no_grad():
    generated_image = ai_model(coordinates).view(IMAGE_HEIGHT, IMAGE_WIDTH, 3)

# --------------------  步骤9:对比原图 VS AI画的图--------------
# plt = 画图工具(专门用来显示、保存图片的)，全称matplotlib.pyplot
#figure: 开画布
#subplot: 切位置
#title: 加标题
#imshow: 显示图片
#savefig：保存图片
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("原图(标准答案)")
plt.imshow(target_image.numpy())

plt.subplot(1, 2, 2)
plt.title("AI学会画的图")
plt.imshow(generated_image.numpy())

#保存图片
plt.savefig("ai_learned_image.png")
plt.close()

print("✅ 运行完成!图片已保存:ai_learned_image.png")












