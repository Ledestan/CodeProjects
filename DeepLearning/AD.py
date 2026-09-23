import torch
from torchviz import make_dot

# 定义变量
a = torch.tensor(2.0)

# 待求梯度的参数
w1 = torch.tensor(1.0, requires_grad=True)
w2 = torch.tensor(2.0, requires_grad=True)
w3 = torch.tensor(3.0, requires_grad=True)
w4 = torch.tensor(4.0, requires_grad=True)

# 前向传播，构建计算图
b = w1 * a
c = w2 * a
d = w3 * b + w4 * c
L = 10 - d

print("前向传播中间结果")
print(f"a = {a.item()}")
print(f"b = w1 * a = {b.item()}")
print(f"c = w2 * a = {c.item()}")
print(f"d = w3 * b + w4 * c = {d.item()}")
print(f"L = 10 - d = {L.item()}")

# 反向传播，自动求导
L.backward()

print("\n反向传播梯度结果")
print(f"w1.grad = {w1.grad}")
print(f"w2.grad = {w2.grad}")
print(f"w3.grad = {w3.grad}")
print(f"w4.grad = {w4.grad}")

lr = 0.01  # 学习率
epoch_num = 15  # 迭代轮数

# 构造优化器，管理参数
optimizer = torch.optim.SGD([w1, w2, w3, w4], lr=lr)

# 梯度下降训练循环
print("\n开始训练")
for epoch in range(epoch_num):
    # 前向传播
    b = w1 * a
    c = w2 * a
    d = w3 * b + w4 * c
    L = 10 - d

    # 梯度手动清零
    if w1.grad is not None:
        w1.grad.zero_()
        w2.grad.zero_()
        w3.grad.zero_()
        w4.grad.zero_()

    L.backward()

    # 禁用计算图追踪，原地更新权重
    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad
        w3 -= lr * w3.grad
        w4 -= lr * w4.grad

    if epoch % 3 == 0:
        print(f"迭代 {epoch}: Loss L = {L.item():.4f}")
print("训练结束")

# 绘制计算图
dot = make_dot(
    L,
    params={
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "w4": w4,
    },
)

# 展示计算图
dot.view()
