"""
分类器图形测试界面
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

sys.dont_write_bytecode = True
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ImageClassifier import ImageClassifier


class Window:
    """图形界面应用程序"""
    def __init__(self, root, classifier):
        self.root = root
        self.clf = classifier
        self.image_path = None

        root.title("图片分类器")
        root.geometry("500x300")
        root.resizable(False, False)

        # 标题
        title_label = tk.Label(root, text="图片分类器", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # 图片选择按钮
        self.btn_select = tk.Button(root, text="选择图片", command=self.select_image, width=15)
        self.btn_select.pack(pady=5)

        # 显示选中的文件名
        self.lbl_path = tk.Label(root, text="尚未选择图片", fg="gray", wraplength=400)
        self.lbl_path.pack(pady=5)

        # 预测按钮
        self.btn_predict = tk.Button(root, text="预测", command=self.predict, width=15, state=tk.DISABLED)
        self.btn_predict.pack(pady=5)

        # 结果显示区域
        self.lbl_result = tk.Label(root, text="", font=("Arial", 12))
        self.lbl_result.pack(pady=10)

        # 提示信息
        info_label = tk.Label(root, text="模型已加载，请选择一张图片进行预测。", fg="green")
        info_label.pack(pady=10)

    def select_image(self):
        """打开文件对话框选择图片"""
        file_path = filedialog.askopenfilename(
            title="选择一张图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png"), ("所有文件", "*.*")]
        )
        if file_path:
            self.image_path = file_path
            # 只显示文件名，避免路径过长
            self.lbl_path.config(text=os.path.basename(file_path), fg="black")
            self.btn_predict.config(state=tk.NORMAL)
            # 清空上次结果
            self.lbl_result.config(text="")

    def predict(self):
        """调用分类器预测选中图片"""
        if not self.image_path:
            return

        result = self.clf.predict_single(self.image_path)

        if not result['success']:
            messagebox.showerror("错误", result.get('error', '预测失败'))
            return

        prob = result['probability']
        class_name = result['class_name']

        # 格式化显示结果
        self.lbl_result.config(
            text=f"预测结果: {class_name}\n概率 (猫=1, 狗=0): {prob:.4f}",
            fg="blue"
        )


def main():
    # 加载模型参数
    param_path = "data/models/params.npz"
    if not os.path.exists(param_path):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("错误", f"未找到参数文件 '{param_path}'。\n请先运行 train.py 进行训练。")
        return

    # 创建分类器实例并加载参数
    clf = ImageClassifier()
    clf.load_params(param_path)

    # 创建 GUI 窗口
    root = tk.Tk()
    app = Window(root, clf)
    root.mainloop()


if __name__ == '__main__':
    main()