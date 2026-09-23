"""
项目名称: 班级点名系统
创建日期: 2025-12-05
"""

import random
import turtle as tk
import winsound


class Classmates:
    def __init__(self, raw_list):
        # 按行切分，去掉空行，再按逗号拆成 [学号, 姓名]
        self.students = [
            [field.strip() for field in line.split(",")] for line in raw_list
        ]

    def draw_name(self, x, y):
        tk.clear()
        tk.up()
        student = random.choice(self.students)

        colors = [
            "#000080",
            "#006400",
            "#8B0000",
            "#4B0082",
            "#008B8B",
            "#800080",
            "#8B4513",
            "#FF8C00",
            "#FF1493",
            "#000080",
        ]
        color = random.choice(colors)
        tk.color(color)

        winsound.Beep(500, 200)

        tk.goto(0, 25)
        tk.write(student[0], align="center", font=("SimSun", 24, "bold"))
        tk.goto(0, -25)
        tk.write(student[1], align="center", font=("SimSun", 24, "bold"))

    def start(self):
        tk.hideturtle()
        screen = tk.Screen()
        screen.setup(500, 500)
        screen.onclick(self.draw_name)
        tk.done()


if __name__ == "__main__":
    # 班级名单
    Students_List = [
        "25001252001,张三",
        "25001252002,李四",
        "25001252003,王五",
        "25001252004,赵六",
        "25001252005,孙七",
        "25001252006,周八",
        "25001252007,吴九",
        "25001252008,郑十",
    ]
    classmate = Classmates(Students_List)
    classmate.start()
