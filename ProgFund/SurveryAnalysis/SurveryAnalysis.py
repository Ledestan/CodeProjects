"""
项目名称: 调研分析系统
创建日期: 2025-11-29
"""

import csv
import math
import tkinter as tk
from tkinter import filedialog


class Window:
    def __init__(self):
        self.data_list = None
        self.pie_window = None

        self.root = tk.Tk()
        self.root.title("数据统计系统")
        self.root.geometry("500x500")

        tk.Label(self.root, text="数据统计系统", font=("SimSun", 20)).pack(pady=15)

        # 文本框和滚动条
        text_frame = tk.Frame(self.root)
        text_frame.pack(pady=10, padx=20, fill=tk.X)
        self.text_box = tk.Text(text_frame, width=48, height=12, font=("SimSun", 12))
        self.text_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(text_frame, command=self.text_box.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_box.config(yscrollcommand=scrollbar.set)
        self.text_box.insert(tk.END, "欢迎使用数据统计系统！\n")

        # 按钮区域
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)
        tk.Button(btn_frame, text="导入数据", bg="lightblue", width=12, height=2,
                  font=("SimSun", 14), command=self.read_csv_to_list).grid(row=0, column=0, padx=10, pady=5)
        tk.Button(btn_frame, text="学习时间", bg="lightgreen", width=12, height=2,
                  font=("SimSun", 14), command=lambda: self.list_count_to_dict(0, 1)).grid(row=0, column=1, padx=10, pady=5)
        tk.Button(btn_frame, text="活动激励", bg="lightyellow", width=12, height=2,
                  font=("SimSun", 14), command=lambda: self.list_count_to_dict(1, 4)).grid(row=0, column=2, padx=10, pady=5)
        tk.Button(btn_frame, text="活动问题", bg="lightpink", width=12, height=2,
                  font=("SimSun", 14), command=lambda: self.list_count_to_dict(4, 6)).grid(row=1, column=0, padx=10, pady=5)
        tk.Button(btn_frame, text="满意度", bg="lightcyan", width=12, height=2,
                  font=("SimSun", 14), command=lambda: self.list_count_to_dict(6, 7)).grid(row=1, column=1, padx=10, pady=5)
        tk.Button(btn_frame, text="退出", bg="red", fg="white", width=12, height=2,
                  font=("SimSun", 14), command=self.root.quit).grid(row=1, column=2, padx=10, pady=5)

    def read_csv_to_list(self):
        """导入CSV文件"""
        file_path = filedialog.askopenfilename(title="选择CSV文件", filetypes=[("CSV文件", "*.csv")])
        if not file_path:
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.data_list = list(csv.reader(file))
                self.text_box.insert(tk.END, "数据导入成功。\n")
        except:
            self.text_box.insert(tk.END, "导入失败，请检查文件格式！\n请导入编码类型为 utf-8 的文件。\n")

    def draw_pie_chart(self, data_dict):
        """绘制饼图到独立窗口"""
        if not data_dict:
            self.text_box.insert(tk.END, "没有有效数据！\n")
            return

        # 创建或更新饼图窗口
        if self.pie_window and self.pie_window.winfo_exists():
            self.pie_window.canvas.delete("all")
        else:
            self.pie_window = tk.Toplevel()
            self.pie_window.title("统计饼图")
            self.pie_window.geometry("1000x700")
            self.pie_window.canvas = tk.Canvas(self.pie_window, width=1000, height=700, bg="white")
            self.pie_window.canvas.pack()

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
                  "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9", "#F8C471"]
        total = sum(data_dict.values())
        center_x, center_y, radius = 350, 350, 180
        start_angle = 0

        for i, (label, value) in enumerate(data_dict.items()):
            percentage = (value / total) * 100
            angle = (value / total) * 360

            points = [center_x, center_y]
            num_segments = max(int(angle / 5), 5)
            for j in range(num_segments + 1):
                current_angle = math.radians(start_angle + (angle * j / num_segments))
                px = center_x + radius * math.cos(current_angle)
                py = center_y - radius * math.sin(current_angle)
                points.extend([px, py])

            color = colors[i % len(colors)]
            self.pie_window.canvas.create_polygon(points, fill=color, outline="black", width=1)
            mid_angle = math.radians(start_angle + angle / 2)
            text_x = center_x + 0.7 * radius * math.cos(mid_angle)
            text_y = center_y - 0.7 * radius * math.sin(mid_angle)
            self.pie_window.canvas.create_text(text_x, text_y, text=f"{percentage:.1f}%",
                                               fill="white", font=("SimSun", 12))
            start_angle += angle

        legend_x, legend_y = 700, 50
        for i, (label, value) in enumerate(data_dict.items()):
            percentage = (value / total) * 100
            color = colors[i % len(colors)]
            self.pie_window.canvas.create_rectangle(legend_x, legend_y + i * 40,
                                                    legend_x + 20, legend_y + 20 + i * 40,
                                                    fill=color, outline="black")
            self.pie_window.canvas.create_text(legend_x + 30, legend_y + 10 + i * 40,
                                               text=f"{label[:15]}: {value} ({percentage:.1f}%)",
                                               anchor="w", font=("SimSun", 12))

    def list_count_to_dict(self, start, end):
        """统计指定列的数据并绘制饼图"""
        if self.data_list is None:
            self.text_box.insert(tk.END, "请先导入数据。\n")
            return

        try:
            data_dict = {}
            for row in self.data_list:
                for item in row[start:end]:
                    if item != '无':
                        data_dict[item] = data_dict.get(item, 0) + 1
            self.draw_pie_chart(data_dict)
        except:
            self.text_box.insert(tk.END, "统计失败，请检查数据！\n")

    def run(self):
        """启动主界面消息循环"""
        self.root.mainloop()


if __name__ == "__main__":
    window = Window()
    window.run()