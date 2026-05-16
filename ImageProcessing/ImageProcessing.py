"""
图像矩阵处理工具

创建日期: 2026-03-30
需求文件: Data\image.png

依赖库:
numpy>=2.2.6
matplotlib>=3.10.8
"""

import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


class ImageProcessing:
    """图像处理核心类：负责图像读写、灰度转换、几何变换和 SVD 压缩"""

    def __init__(self, path: str = None):
        """读取图像数据转换为 NumPy 数组，并进行预处理翻转"""
        if path is not None:
            self.image_orig = plt.imread(path)
            # 归一化到 uint8
            if self.image_orig.dtype == np.float32 or self.image_orig.dtype == np.float64:
                self.image_orig = (self.image_orig * 255).astype(np.uint8)
            # 垂直翻转，使数组第0行对应图像底部，与坐标系 origin="lower" 匹配
            self.image_orig = np.flipud(self.image_orig)
            self.image_gray = None
            self.transform()  # 自动转为灰度图
        else:
            self.image_orig = None
            self.image_gray = None

    def load_image(self, path: str):
        """重新加载图像（供外部调用）"""
        self.image_orig = plt.imread(path)
        if self.image_orig.dtype == np.float32 or self.image_orig.dtype == np.float64:
            self.image_orig = (self.image_orig * 255).astype(np.uint8)
        self.image_orig = np.flipud(self.image_orig)   # 翻转存储
        self.transform()   # 自动生成灰度图

    def transform(self):
        """彩色图转换灰度图（加权平均法）"""
        if self.image_orig is None:
            return
        if len(self.image_orig.shape) == 2:
            self.image_gray = self.image_orig.copy()
            return
        R = self.image_orig[:, :, 0].astype(np.float32)
        G = self.image_orig[:, :, 1].astype(np.float32)
        B = self.image_orig[:, :, 2].astype(np.float32)
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        self.image_gray = np.clip(gray, 0, 255).astype(np.uint8)

    def preview(self):
        """预览图像矩阵信息并弹出 matplotlib 窗口显示"""
        if self.image_orig is None:
            print("未加载图像")
            return
        # 彩色图像信息
        print("彩色图，左上角 2 * 3 像素信息：")
        print(self.image_orig[:2, :3])
        print(f"矩阵信息：{self.image_orig.shape}")
        print(f"数据类型：{self.image_orig.dtype}")

        # 灰度图像信息
        print("灰度图，左上角 5 * 5 像素信息：")
        print(self.image_gray[:5, :5])
        print(f"矩阵信息：{self.image_gray.shape}")
        print(f"数据类型：{self.image_gray.dtype}")

        # 显示图像
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(self.image_orig)
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(self.image_gray, cmap="gray")
        plt.title("Grayscale Image")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # ---------- 通用几何变换（反向映射 + 双线性插值） ----------
    def _apply_transform(self, img, matrix, is_homogeneous=False):
        """
        对图像应用 2x2 或 3x3 变换矩阵，返回变换后图像及对应的 extent
        输出图像尺寸根据变换后包围盒自动计算，避免拉伸变形
        - matrix: 2x2 或 3x3 ndarray
        - is_homogeneous: 若为 True，matrix 为 3x3 齐次矩阵；否则为 2x2 需自动扩展
        """
        if img is None:
            return None, None

        h, w = img.shape[:2]

        # 1. 构造 3x3 齐次矩阵
        if not is_homogeneous:
            M = np.eye(3)
            M[:2, :2] = matrix
        else:
            M = matrix

        # 2. 计算输入图像四个角点变换后的位置，确定包围盒
        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]])
        new_corners = (M @ corners.T).T
        xmin, xmax = np.min(new_corners[:, 0]), np.max(new_corners[:, 0])
        ymin, ymax = np.min(new_corners[:, 1]), np.max(new_corners[:, 1])

        # 3. 确定输出图像的像素尺寸（保证每个像素大致对应 1 个单位）
        out_w = int(np.ceil(xmax - xmin))
        out_h = int(np.ceil(ymax - ymin))
        out_w = max(out_w, 1)
        out_h = max(out_h, 1)

        # 4. 生成输出图像的网格坐标（齐次），并映射到世界坐标系
        y_out, x_out = np.indices((out_h, out_w))
        world_x = xmin + (x_out / out_w) * (xmax - xmin)
        world_y = ymin + (y_out / out_h) * (ymax - ymin)
        coords_out = np.stack([world_x, world_y, np.ones_like(world_x)], axis=-1)

        # 5. 计算逆矩阵，将世界坐标映射回输入图像坐标
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            print("警告：变换矩阵不可逆")
            return img, [xmin, xmax, ymin, ymax]

        coords_flat = coords_out.reshape(-1, 3)
        src_coords = (M_inv @ coords_flat.T).T
        src_x = src_coords[:, 0].reshape(out_h, out_w)
        src_y = src_coords[:, 1].reshape(out_h, out_w)

        # 6. 双线性插值生成新图像
        if len(img.shape) == 3:
            channels = img.shape[2]
            new_img = np.zeros((out_h, out_w, channels), dtype=np.uint8)
            for c in range(channels):
                self._interpolate(img[:, :, c], src_x, src_y, new_img[:, :, c])
        else:
            new_img = np.zeros((out_h, out_w), dtype=np.uint8)
            self._interpolate(img, src_x, src_y, new_img)

        extent = [xmin, xmax, ymin, ymax]
        return new_img, extent

    def _interpolate(self, src, x, y, dst):
        """双线性插值辅助函数"""
        h, w = src.shape
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        x0 = np.floor(x).astype(int)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y0 = np.floor(y).astype(int)
        y1 = np.clip(y0 + 1, 0, h - 1)

        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)

        dst[:] = (wa * src[y0, x0] + wb * src[y0, x1] +
                  wc * src[y1, x0] + wd * src[y1, x1]).astype(np.uint8)

    # ---------- SVD 压缩 ----------
    def svd_compress(self, k, use_gray=False):
        """
        对当前图像进行 SVD 压缩（灰度图或彩色图逐通道）
        k: 保留的奇异值个数
        """
        img = self.image_gray if use_gray else self.image_orig
        if img is None:
            return None

        if len(img.shape) == 2:
            # 灰度图
            U, s, Vt = np.linalg.svd(img.astype(np.float64), full_matrices=False)
            S = np.diag(s[:k])
            compressed = U[:, :k] @ S @ Vt[:k, :]
            compressed = np.clip(compressed, 0, 255).astype(np.uint8)
            orig_size = img.size
            comp_size = k * (U.shape[0] + Vt.shape[1] + 1)
            print(f"\n[SVD 灰度压缩] k={k}, 压缩比 ≈ {comp_size/orig_size:.2%}")
            return compressed
        else:
            # 彩色图逐通道处理
            h, w, c = img.shape
            compressed = np.zeros_like(img)
            total_orig = img.size
            total_comp = 0
            for ch in range(c):
                channel = img[:, :, ch].astype(np.float64)
                U, s, Vt = np.linalg.svd(channel, full_matrices=False)
                S = np.diag(s[:k])
                compressed[:, :, ch] = np.clip(U[:, :k] @ S @ Vt[:k, :], 0, 255)
                total_comp += k * (U.shape[0] + Vt.shape[1] + 1)
            print(f"\n[SVD 彩色压缩] k={k}, 压缩比 ≈ {total_comp/total_orig:.2%}")
            return compressed.astype(np.uint8)


class Window:
    """主 GUI 窗口类"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("图像几何变换工具")
        self.root.geometry("1200x900")

        self.img_proc = ImageProcessing()          # 图像处理实例
        self.total_matrix = np.eye(3)              # 累积变换矩阵（齐次坐标）
        self.current_extent = [0, 200, 0, 200]     # 当前图像的显示范围
        self.use_gray = tk.BooleanVar(value=False) # 是否显示灰度图

        self._setup_ui()
        self._setup_plot()

    # ---------- UI 初始化 ----------
    def _setup_ui(self):
        """构建功能区"""
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # 第一行：选择图片、显示模式
        row1 = tk.Frame(control_frame)
        row1.pack(fill=tk.X, pady=2)
        tk.Button(row1, text="选择图片", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(row1, text="彩色图", variable=self.use_gray, value=False,
                       command=self._update_display).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(row1, text="灰度图", variable=self.use_gray, value=True,
                       command=self._update_display).pack(side=tk.LEFT, padx=5)

        # 第二行：单步变换选择及参数
        row2 = tk.Frame(control_frame)
        row2.pack(fill=tk.X, pady=2)
        tk.Label(row2, text="单步变换:").pack(side=tk.LEFT, padx=5)

        self.transform_var = tk.StringVar()
        self.transform_combo = ttk.Combobox(row2, textvariable=self.transform_var,
                                            state="readonly", width=20)
        self.transform_combo["values"] = [
            "关于x轴对称", "关于y轴对称", "关于y=x对称", "关于y=-x对称", "关于原点对称",
            "缩放(等比例)", "水平收缩与拉伸", "垂直收缩与拉伸", "水平剪切", "垂直剪切",
            "旋转", "平移"
        ]
        self.transform_combo.pack(side=tk.LEFT, padx=5)
        self.transform_combo.bind("<<ComboboxSelected>>", self._on_transform_select)

        # 参数输入框（默认禁用）
        self.entry1 = tk.Entry(row2, width=8, state="disabled")
        self.entry1.pack(side=tk.LEFT, padx=2)
        self.entry2 = tk.Entry(row2, width=8, state="disabled")
        self.entry2.pack(side=tk.LEFT, padx=2)

        tk.Button(row2, text="确认变换", command=self.apply_single_transform).pack(side=tk.LEFT, padx=10)

        # 第三行：复合变换、SVD 压缩、重置
        row3 = tk.Frame(control_frame)
        row3.pack(fill=tk.X, pady=5)
        tk.Button(row3, text="复合变换", command=self.open_composite_dialog).pack(side=tk.LEFT, padx=10)
        tk.Button(row3, text="SVD压缩", command=self.open_svd_dialog).pack(side=tk.LEFT, padx=10)
        tk.Button(row3, text="重置变换", command=self.reset_transform).pack(side=tk.LEFT, padx=10)

    def _setup_plot(self):
        """设置画布（800x800 坐标系，原点在中心）"""
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(-400, 400)
        self.ax.set_ylim(-400, 400)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle="--", alpha=0.7)
        self.ax.axhline(0, color="black", linewidth=0.5)
        self.ax.axvline(0, color="black", linewidth=0.5)
        self.ax.set_title("图像显示区 (第一象限初始位置)")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    # ---------- 辅助方法 ----------
    def _on_transform_select(self, event=None):
        """根据选中的变换启用对应的参数输入框"""
        name = self.transform_var.get()
        self.entry1.config(state="disabled")
        self.entry2.config(state="disabled")
        self.entry1.delete(0, tk.END)
        self.entry2.delete(0, tk.END)

        if name in ["缩放(等比例)", "水平收缩与拉伸", "垂直收缩与拉伸", "水平剪切", "垂直剪切", "旋转"]:
            self.entry1.config(state="normal")
        elif name == "平移":
            self.entry1.config(state="normal")
            self.entry2.config(state="normal")

    def _get_transform_matrix(self, name, param1=0.0, param2=0.0):
        """
        根据变换名称和参数返回 3x3 齐次变换矩阵
        - name: 变换名称（与下拉框选项一致）
        - param1, param2: 参数（如缩放系数、角度、平移量等）
        """
        M = np.eye(3)
        if name == "关于x轴对称":
            M[:2, :2] = [[1, 0], [0, -1]]
        elif name == "关于y轴对称":
            M[:2, :2] = [[-1, 0], [0, 1]]
        elif name == "关于y=x对称":
            M[:2, :2] = [[0, 1], [1, 0]]
        elif name == "关于y=-x对称":
            M[:2, :2] = [[0, -1], [-1, 0]]
        elif name == "关于原点对称":
            M[:2, :2] = [[-1, 0], [0, -1]]
        elif name == "缩放(等比例)":
            M[:2, :2] = [[param1, 0], [0, param1]]
        elif name == "水平收缩与拉伸":
            M[:2, :2] = [[param1, 0], [0, 1]]
        elif name == "垂直收缩与拉伸":
            M[:2, :2] = [[1, 0], [0, param1]]
        elif name == "水平剪切":
            M[:2, :2] = [[1, param1], [0, 1]]
        elif name == "垂直剪切":
            M[:2, :2] = [[1, 0], [param1, 1]]
        elif name == "旋转":
            rad = np.radians(param1)
            c, s = np.cos(rad), np.sin(rad)
            M[:2, :2] = [[c, -s], [s, c]]
        elif name == "平移":
            M = np.array([[1, 0, param1], [0, 1, param2], [0, 0, 1]])
        return M

    def load_image(self):
        """选择图片并重置变换状态"""
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return
        self.img_proc.load_image(path)
        self.total_matrix = np.eye(3)
        h, w = self.img_proc.image_orig.shape[:2]
        self.current_extent = [0, w, 0, h]
        self._update_display()
        print(f"已加载图像: {path}")

    def reset_transform(self):
        """重置变换矩阵到初始状态"""
        if self.img_proc.image_orig is None:
            return
        self.total_matrix = np.eye(3)
        h, w = self.img_proc.image_orig.shape[:2]
        self.current_extent = [0, w, 0, h]
        self._update_display()
        print("变换已重置")

    def _update_display(self):
        """根据当前总矩阵和显示模式刷新画面"""
        if self.img_proc.image_orig is None:
            return
        # 根据模式选择原始图像
        original = self.img_proc.image_gray if self.use_gray.get() else self.img_proc.image_orig
        # 应用累积变换矩阵
        transformed, extent = self.img_proc._apply_transform(original, self.total_matrix, is_homogeneous=True)
        if transformed is not None:
            self.current_extent = extent
            self._draw_image(transformed, extent)

    def _draw_image(self, img, extent=None):
        """在坐标系中绘制图像"""
        self.ax.clear()
        self.ax.set_xlim(-400, 400)
        self.ax.set_ylim(-400, 400)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle="--", alpha=0.7)
        self.ax.axhline(0, color="black", linewidth=0.5)
        self.ax.axvline(0, color="black", linewidth=0.5)
        self.ax.set_title("图像显示区")

        if extent is None:
            h, w = img.shape[:2]
            extent = [0, w, 0, h]

        cmap = "gray" if len(img.shape) == 2 else None
        self.ax.imshow(img, extent=extent, cmap=cmap, origin="lower")
        self.canvas.draw()

    # ---------- 单步变换 ----------
    def apply_single_transform(self):
        """执行单步变换，累积到总矩阵中"""
        if self.img_proc.image_orig is None:
            print("请先选择图片")
            return

        name = self.transform_var.get()
        if not name:
            print("请选择变换类型")
            return

        try:
            p1 = float(self.entry1.get()) if self.entry1["state"] == "normal" else 0.0
            p2 = float(self.entry2.get()) if self.entry2["state"] == "normal" else 0.0
        except ValueError:
            print("参数输入错误，请输入数字")
            return

        M_new = self._get_transform_matrix(name, p1, p2)
        # 累积变换：新矩阵左乘总矩阵（先施加新变换，再施加原有变换）
        self.total_matrix = M_new @ self.total_matrix
        print("当前总变换矩阵（齐次坐标）:")
        print(self.total_matrix)
        self._update_display()
        print(f"执行单步变换: {name}")

    # ---------- 复合变换 ----------
    def open_composite_dialog(self):
        """弹出复合变换设置窗口"""
        if self.img_proc.image_orig is None:
            print("请先选择图片")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("复合变换设置")
        dialog.geometry("400x300")

        # 界面布局
        tk.Label(dialog, text="第一/二类变换:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        combo_var = tk.StringVar()
        combo = ttk.Combobox(dialog, textvariable=combo_var, state="readonly", width=20)
        combo["values"] = [
            "关于x轴对称", "关于y轴对称", "关于y=x对称", "关于y=-x对称", "关于原点对称",
            "缩放(等比例)", "水平收缩与拉伸", "垂直收缩与拉伸", "水平剪切", "垂直剪切"
        ]
        combo.grid(row=0, column=1, padx=5, pady=5)
        combo.current(0)

        tk.Label(dialog, text="参数 k (如需要):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        entry_k = tk.Entry(dialog, width=10)
        entry_k.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        tk.Label(dialog, text="旋转角度 (°):").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        entry_angle = tk.Entry(dialog, width=10)
        entry_angle.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        tk.Label(dialog, text="平移 dx:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        entry_dx = tk.Entry(dialog, width=10)
        entry_dx.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        tk.Label(dialog, text="平移 dy:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        entry_dy = tk.Entry(dialog, width=10)
        entry_dy.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        def apply_composite():
            try:
                t1 = combo_var.get()
                # 对称变换不需要参数
                if t1 in ["关于x轴对称", "关于y轴对称", "关于y=x对称", "关于y=-x对称", "关于原点对称"]:
                    p1 = 0.0
                else:
                    p1 = float(entry_k.get())
                angle = float(entry_angle.get())
                dx = float(entry_dx.get())
                dy = float(entry_dy.get())
            except ValueError:
                print("请输入有效的数字")
                return

            # 使用统一的矩阵生成函数
            M1 = self._get_transform_matrix(t1, p1, 0)
            M_rot = self._get_transform_matrix("旋转", angle, 0)
            M_trans = self._get_transform_matrix("平移", dx, dy)
            M_total = M_trans @ M_rot @ M1

            print("\n===== 复合变换矩阵计算 =====")
            print(f"第一/二类变换 ({t1}):")
            print(M1[:2, :2])
            print(f"\n旋转矩阵 (角度 {angle:.2f}°):")
            print(M_rot[:2, :2])
            print(f"\n平移矩阵 (dx={dx:.2f}, dy={dy:.2f}):")
            print(M_trans)
            print("\n复合顺序: M_total = M_trans @ M_rot @ M1")
            print("总变换矩阵 (齐次坐标):")
            print(M_total)
            print("=============================\n")

            self.total_matrix = M_total
            self._update_display()
            print("复合变换完成")
            dialog.destroy()

        tk.Button(dialog, text="确认", command=apply_composite).grid(row=5, column=0, columnspan=2, pady=20)

    # ---------- SVD 压缩 ----------
    def open_svd_dialog(self):
        """弹出 SVD 压缩设置窗口"""
        if self.img_proc.image_orig is None:
            print("请先选择图片")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("SVD 压缩")
        dialog.geometry("400x150")

        # 获取当前显示图像以确定最大 k
        original = self.img_proc.image_gray if self.use_gray.get() else self.img_proc.image_orig
        current_img, _ = self.img_proc._apply_transform(original, self.total_matrix, is_homogeneous=True)
        max_k = min(current_img.shape[:2])
        k_var = tk.IntVar(value=min(50, max_k))

        tk.Label(dialog, text="保留奇异值个数 k:").pack(pady=10)
        scale = tk.Scale(dialog, from_=1, to=max_k, orient=tk.HORIZONTAL,
                         variable=k_var, length=300)
        scale.pack()

        def apply_svd():
            k = k_var.get()
            # 对当前显示的图像进行压缩
            if len(current_img.shape) == 2:
                U, s, Vt = np.linalg.svd(current_img.astype(np.float64), full_matrices=False)
                S = np.diag(s[:k])
                compressed = np.clip(U[:, :k] @ S @ Vt[:k, :], 0, 255).astype(np.uint8)
            else:
                h, w, c = current_img.shape
                compressed = np.zeros_like(current_img)
                for ch in range(c):
                    channel = current_img[:, :, ch].astype(np.float64)
                    U, s, Vt = np.linalg.svd(channel, full_matrices=False)
                    S = np.diag(s[:k])
                    compressed[:, :, ch] = np.clip(U[:, :k] @ S @ Vt[:k, :], 0, 255)
                compressed = compressed.astype(np.uint8)

            self._draw_image(compressed, self.current_extent)
            print(f"SVD 压缩完成，k={k}")
            dialog.destroy()

        tk.Button(dialog, text="确认", command=apply_svd).pack(pady=10)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = Window()
    app.run()