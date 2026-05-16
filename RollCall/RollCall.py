"""
项目名称: 班级点名系统
创建日期: 2025-12-05
"""

import random
import turtle as tk
import winsound


class Classmates():
    def __init__(self, file_path):
        file = open(file_path, encoding="utf-8")
        self.lines = file.readlines()
        file.close()
    
    def draw_name(self, x, y):
        tk.clear()
        tk.up()
        student = random.choice(self.lines).strip().split()
        
        colors = ["#000080", "#006400", "#8B0000", "#4B0082", "#008B8B", "#800080", "#8B4513", "#FF8C00", "#FF1493", "#000080"]
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
    classmate = Classmates("Data/Students_List.txt")
    classmate.start()