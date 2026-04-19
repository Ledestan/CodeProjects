"""
图像处理工具

创建日期：2026-03-30
需求文件：static\image.png

依赖库：
numpy>=2.2.6
matplotlib>=3.10.8
"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class ImageProcessing:
    def __init__(self, path: str = None):
        """读取图像数据转换为 NumPy 数组"""
        if path is not None:
            self.image_orig = plt.imread(path)
            self.image_gray = None
            self.transform()  # 自动转为灰度图
        else:
            self.image_orig = None
            self.image_gray = None

    def load_image(self, path: str):
        """重新加载图像（供外部调用）"""
        self.image_orig = plt.imread(path)
        self.transform()

    def transform(self):
        """彩色图转换灰度图"""
        if self.image_orig is None:
            return
        # 如果已经是 uint8 且是彩色图，跳过重复转换
        if self.image_orig.dtype == np.float32 or self.image_orig.dtype == np.float64:
            self.image_orig = (self.image_orig * 255).astype(np.uint8)

        # 如果是灰度图（二维数组），直接复制
        if len(self.image_orig.shape) == 2:
            self.image_gray = self.image_orig.copy()
            return

        # 提取 RGB 通道
        R = self.image_orig[:, :, 0].astype(np.float32)
        G = self.image_orig[:, :, 1].astype(np.float32)
        B = self.image_orig[:, :, 2].astype(np.float32)

        # 加权平均并转为 uint8
        image_gray = 0.299 * R + 0.587 * G + 0.114 * B
        self.image_gray = np.clip(image_gray, 0, 255).astype(np.uint8)

    def preview(self):
        """预览图像矩阵信息"""
        if self.image_orig is None:
            print("未加载图像")
            return
        # 彩色图像
        print('彩色图，左上角 2 * 3 像素信息：')
        print(self.image_orig[:2, :3])
        print(f'矩阵信息：{self.image_orig.shape}')
        print(f'数据类型：{self.image_orig.dtype}')

        # 灰度图像
        print('灰度图，左上角 5 * 5 像素信息：')
        print(self.image_gray[:5, :5])
        print(f'矩阵信息：{self.image_gray.shape}')
        print(f'数据类型：{self.image_gray.dtype}')

        # 显示图像
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(self.image_orig)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(self.image_gray, cmap='gray')
        plt.title('Grayscale Image')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # ---------- 辅助函数：通用图像变换（反向映射 + 双线性插值） ----------
    def _apply_transform(self, img, matrix, is_homogeneous=False):
        """
        对图像应用 2x2 或 3x3 变换矩阵，返回变换后图像 (uint8)
        - matrix: 2x2 或 3x3 ndarray
        - is_homogeneous: 若为 True，matrix 为 3x3 齐次矩阵；否则为 2x2 需自动扩展
        """
        if img is None:
            return None
        h, w = img.shape[:2]

        # 生成网格坐标 (x, y, 1) 齐次形式
        y, x = np.indices((h, w))
        coords = np.stack([x, y, np.ones_like(x)], axis=-1)  # (h, w, 3)

        if not is_homogeneous:
            M = np.eye(3)
            M[:2, :2] = matrix
        else:
            M = matrix

        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            print("警告：变换矩阵不可逆，无法进行反向映射")
            return img

        # 计算新坐标 (反向映射)
        coords_flat = coords.reshape(-1, 3)
        new_coords = (M_inv @ coords_flat.T).T
        new_x = new_coords[:, 0].reshape(h, w)
        new_y = new_coords[:, 1].reshape(h, w)

        # 双线性插值
        if len(img.shape) == 3:
            channels = img.shape[2]
            new_img = np.zeros_like(img)
            for c in range(channels):
                self._interpolate(img[:, :, c], new_x, new_y, new_img[:, :, c])
        else:
            new_img = np.zeros_like(img)
            self._interpolate(img, new_x, new_y, new_img)

        return new_img.astype(np.uint8)

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

    def _get_image(self, use_gray):
        """根据显示模式返回当前要处理的图像副本"""
        if use_gray:
            return self.image_gray.copy() if self.image_gray is not None else None
        else:
            return self.image_orig.copy() if self.image_orig is not None else None

    # ---------- 第一类变换：对称变换 ----------
    def symmetry_x(self, use_gray=False):
        img = self._get_image(use_gray)
        M = np.array([[1, 0], [0, -1]])
        return self._apply_transform(img, M)

    def symmetry_y(self, use_gray=False):
        img = self._get_image(use_gray)
        M = np.array([[-1, 0], [0, 1]])
        return self._apply_transform(img, M)

    def symmetry_yx(self, use_gray=False):
        img = self._get_image(use_gray)
        M = np.array([[0, 1], [1, 0]])
        return self._apply_transform(img, M)

    def symmetry_y_minus_x(self, use_gray=False):
        img = self._get_image(use_gray)
        M = np.array([[0, -1], [-1, 0]])
        return self._apply_transform(img, M)

    def symmetry_origin(self, use_gray=False):
        img = self._get_image(use_gray)
        M = np.array([[-1, 0], [0, -1]])
        return self._apply_transform(img, M)

    # ---------- 第二类变换：缩放与剪切 ----------
    def scale_uniform(self, k, use_gray=False):
        img = self._get_image(use_gray)
        M = np.array([[k, 0], [0, k]])
        return self._apply_transform(img, M)

    def scale_horizontal(self, k, use_gray=False):
        img = self._get_image(use_gray)
        M = np.array([[k, 0], [0, 1]])
        return self._apply_transform(img, M)

    def scale_vertical(self, k, use_gray=False):
        img = self._get_image(use_gray)
        M = np.array([[1, 0], [0, k]])
        return self._apply_transform(img, M)

    def shear_horizontal(self, k, use_gray=False):
        img = self._get_image(use_gray)
        M = np.array([[1, k], [0, 1]])
        return self._apply_transform(img, M)

    def shear_vertical(self, k, use_gray=False):
        img = self._get_image(use_gray)
        M = np.array([[1, 0], [k, 1]])
        return self._apply_transform(img, M)

    # ---------- 第三类变换：旋转 ----------
    def rotate(self, angle_deg, use_gray=False):
        img = self._get_image(use_gray)
        rad = np.radians(angle_deg)
        cos = np.cos(rad)
        sin = np.sin(rad)
        M = np.array([[cos, -sin], [sin, cos]])
        return self._apply_transform(img, M)

    # ---------- 第四类变换：平移（齐次坐标） ----------
    def translate(self, dx, dy, use_gray=False):
        img = self._get_image(use_gray)
        M = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
        return self._apply_transform(img, M, is_homogeneous=True)

    # ---------- 复合变换 ----------
    def composite_transform(self, t1_name, param1, angle_deg, dx, dy, use_gray=False):
        """
        执行复合变换：第一/二类变换 + 旋转 + 平移
        在控制台输出详细计算过程
        """
        img = self._get_image(use_gray)
        if img is None:
            return None

        # 1. 构造第一/二类变换矩阵
        if t1_name == "关于x轴对称":
            M1 = np.array([[1, 0], [0, -1]])
        elif t1_name == "关于y轴对称":
            M1 = np.array([[-1, 0], [0, 1]])
        elif t1_name == "关于y=x对称":
            M1 = np.array([[0, 1], [1, 0]])
        elif t1_name == "关于y=-x对称":
            M1 = np.array([[0, -1], [-1, 0]])
        elif t1_name == "关于原点对称":
            M1 = np.array([[-1, 0], [0, -1]])
        elif t1_name == "缩放(等比例)":
            M1 = np.array([[param1, 0], [0, param1]])
        elif t1_name == "水平收缩与拉伸":
            M1 = np.array([[param1, 0], [0, 1]])
        elif t1_name == "垂直收缩与拉伸":
            M1 = np.array([[1, 0], [0, param1]])
        elif t1_name == "水平剪切":
            M1 = np.array([[1, param1], [0, 1]])
        elif t1_name == "垂直剪切":
            M1 = np.array([[1, 0], [param1, 1]])
        else:
            raise ValueError(f"未知变换名称: {t1_name}")

        # 2. 旋转矩阵
        rad = np.radians(angle_deg)
        cos, sin = np.cos(rad), np.sin(rad)
        M_rot = np.array([[cos, -sin], [sin, cos]])

        # 3. 平移矩阵
        M_trans = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

        # 4. 扩展为齐次坐标
        M1_h = np.eye(3)
        M1_h[:2, :2] = M1
        M_rot_h = np.eye(3)
        M_rot_h[:2, :2] = M_rot

        # 5. 复合顺序: M_total = M_trans @ M_rot_h @ M1_h
        M_total = M_trans @ M_rot_h @ M1_h

        # 控制台输出
        print("\n===== 复合变换矩阵计算 =====")
        print(f"第一/二类变换 ({t1_name}):")
        print(M1)
        print(f"\n旋转矩阵 (角度 {angle_deg:.2f}°):")
        print(M_rot)
        print(f"\n平移矩阵 (dx={dx:.2f}, dy={dy:.2f}):")
        print(M_trans)
        print("\n复合顺序: M_total = M_trans @ M_rot @ M1")
        print("总变换矩阵 (齐次坐标):")
        print(M_total)
        print("=============================\n")

        return self._apply_transform(img, M_total, is_homogeneous=True)

    # ---------- SVD 压缩 ----------
    def svd_compress(self, k, use_gray=False):
        """
        对当前图像进行 SVD 压缩
        k: 保留的奇异值个数
        """
        img = self._get_image(use_gray)
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
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("图像几何变换工具")
        self.root.geometry("1200x900")

        self.img_proc = ImageProcessing()  # 初始空对象
        self.current_image = None          # 当前显示的图像 (uint8)
        self.current_extent = [0, 200, 0, 200]  # 当前图像的显示范围，初始化默认值
        self.use_gray = tk.BooleanVar(value=False)  # 是否显示灰度图

        self._setup_ui()
        self._setup_plot()

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
        self.transform_combo['values'] = [
            "关于x轴对称", "关于y轴对称", "关于y=x对称", "关于y=-x对称", "关于原点对称",
            "缩放(等比例)", "水平收缩与拉伸", "垂直收缩与拉伸", "水平剪切", "垂直剪切",
            "旋转", "平移"
        ]
        self.transform_combo.pack(side=tk.LEFT, padx=5)
        self.transform_combo.bind('<<ComboboxSelected>>', self._on_transform_select)

        # 参数输入框
        self.entry1 = tk.Entry(row2, width=8, state='disabled')
        self.entry1.pack(side=tk.LEFT, padx=2)
        self.entry2 = tk.Entry(row2, width=8, state='disabled')
        self.entry2.pack(side=tk.LEFT, padx=2)

        tk.Button(row2, text="确认变换", command=self.apply_single_transform).pack(side=tk.LEFT, padx=10)

        # 第三行：复合变换与压缩
        row3 = tk.Frame(control_frame)
        row3.pack(fill=tk.X, pady=5)
        tk.Button(row3, text="复合变换", command=self.open_composite_dialog).pack(side=tk.LEFT, padx=10)
        tk.Button(row3, text="SVD压缩", command=self.open_svd_dialog).pack(side=tk.LEFT, padx=10)

    def _setup_plot(self):
        """设置画布（800x800 坐标系）"""
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(-400, 400)
        self.ax.set_ylim(-400, 400)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.axvline(0, color='black', linewidth=0.5)
        self.ax.set_title("图像显示区 (第一象限初始位置)")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def _on_transform_select(self, event=None):
        """根据选中的变换启用对应的参数输入框"""
        name = self.transform_var.get()
        self.entry1.config(state='disabled')
        self.entry2.config(state='disabled')
        self.entry1.delete(0, tk.END)
        self.entry2.delete(0, tk.END)

        if name in ["缩放(等比例)", "水平收缩与拉伸", "垂直收缩与拉伸", "水平剪切", "垂直剪切", "旋转"]:
            self.entry1.config(state='normal')
        elif name == "平移":
            self.entry1.config(state='normal')
            self.entry2.config(state='normal')

    def load_image(self):
        """选择图片并显示"""
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return
        self.img_proc.load_image(path)
        self.current_image = self.img_proc.image_orig.copy()
        # 初始 extent 设为图像原始范围（第一象限）
        h, w = self.current_image.shape[:2]
        self.current_extent = [0, w, 0, h]
        self._update_display()
        print(f"已加载图像: {path}")

    def _update_display(self):
        """根据显示模式刷新画布，复用当前 extent"""
        if self.img_proc.image_orig is None:
            return
        if self.use_gray.get():
            img = self.img_proc.image_gray
        else:
            img = self.img_proc.image_orig
        self.current_image = img.copy()
        self._draw_image(img, self.current_extent)

    def _get_image_extent(self, img, transform_matrix=None):
        """
        计算图像四个角点经过变换后的边界框 extent
        extent = [xmin, xmax, ymin, ymax]
        """
        h, w = img.shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]])  # 齐次坐标

        if transform_matrix is not None:
            if transform_matrix.shape == (2, 2):
                M = np.eye(3)
                M[:2, :2] = transform_matrix
            else:
                M = transform_matrix
            new_corners = (M @ corners.T).T
            xmin = np.min(new_corners[:, 0])
            xmax = np.max(new_corners[:, 0])
            ymin = np.min(new_corners[:, 1])
            ymax = np.max(new_corners[:, 1])
        else:
            xmin, xmax = 0, w
            ymin, ymax = 0, h
        return [xmin, xmax, ymin, ymax]

    def _draw_image(self, img, extent=None):
        """
        在坐标系中绘制图像
        注意：由于 matplotlib imshow 的 origin 默认行为，图像数组第0行在顶部。
        为了与我们的坐标系（左下角为原点）一致，需要对图像数组垂直翻转。
        """
        # 垂直翻转图像，使其在 origin='lower' 模式下正向显示
        img_flipped = np.flipud(img)

        self.ax.clear()
        self.ax.set_xlim(-400, 400)
        self.ax.set_ylim(-400, 400)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.axvline(0, color='black', linewidth=0.5)
        self.ax.set_title("图像显示区")

        if extent is None:
            h, w = img.shape[:2]
            extent = [0, w, 0, h]

        cmap = 'gray' if len(img_flipped.shape) == 2 else None
        self.ax.imshow(img_flipped, extent=extent, cmap=cmap, origin='lower')
        self.canvas.draw()

    def apply_single_transform(self):
        """执行单步变换"""
        if self.img_proc.image_orig is None:
            print("请先选择图片")
            return

        name = self.transform_var.get()
        if not name:
            print("请选择变换类型")
            return

        use_gray = self.use_gray.get()
        img_transformed = None
        M = None  # 用于计算 extent

        try:
            if name == "关于x轴对称":
                img_transformed = self.img_proc.symmetry_x(use_gray)
                M = np.array([[1, 0], [0, -1]])
            elif name == "关于y轴对称":
                img_transformed = self.img_proc.symmetry_y(use_gray)
                M = np.array([[-1, 0], [0, 1]])
            elif name == "关于y=x对称":
                img_transformed = self.img_proc.symmetry_yx(use_gray)
                M = np.array([[0, 1], [1, 0]])
            elif name == "关于y=-x对称":
                img_transformed = self.img_proc.symmetry_y_minus_x(use_gray)
                M = np.array([[0, -1], [-1, 0]])
            elif name == "关于原点对称":
                img_transformed = self.img_proc.symmetry_origin(use_gray)
                M = np.array([[-1, 0], [0, -1]])
            elif name in ["缩放(等比例)", "水平收缩与拉伸", "垂直收缩与拉伸", "水平剪切", "垂直剪切"]:
                k = float(self.entry1.get())
                if name == "缩放(等比例)":
                    img_transformed = self.img_proc.scale_uniform(k, use_gray)
                    M = np.array([[k, 0], [0, k]])
                elif name == "水平收缩与拉伸":
                    img_transformed = self.img_proc.scale_horizontal(k, use_gray)
                    M = np.array([[k, 0], [0, 1]])
                elif name == "垂直收缩与拉伸":
                    img_transformed = self.img_proc.scale_vertical(k, use_gray)
                    M = np.array([[1, 0], [0, k]])
                elif name == "水平剪切":
                    img_transformed = self.img_proc.shear_horizontal(k, use_gray)
                    M = np.array([[1, k], [0, 1]])
                elif name == "垂直剪切":
                    img_transformed = self.img_proc.shear_vertical(k, use_gray)
                    M = np.array([[1, 0], [k, 1]])
            elif name == "旋转":
                angle = float(self.entry1.get())
                img_transformed = self.img_proc.rotate(angle, use_gray)
                rad = np.radians(angle)
                cos, sin = np.cos(rad), np.sin(rad)
                M = np.array([[cos, -sin], [sin, cos]])
            elif name == "平移":
                dx = float(self.entry1.get())
                dy = float(self.entry2.get())
                img_transformed = self.img_proc.translate(dx, dy, use_gray)
                M = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
            else:
                print("未知变换")
                return
        except ValueError:
            print("参数输入错误，请输入数字")
            return

        if img_transformed is not None:
            self.current_image = img_transformed
            self.current_extent = self._get_image_extent(img_transformed, M)
            self._draw_image(img_transformed, self.current_extent)
            print(f"执行单步变换: {name}")

    def open_composite_dialog(self):
        """弹出复合变换设置窗口"""
        if self.img_proc.image_orig is None:
            print("请先选择图片")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("复合变换设置")
        dialog.geometry("400x300")

        tk.Label(dialog, text="第一/二类变换:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        combo_var = tk.StringVar()
        combo = ttk.Combobox(dialog, textvariable=combo_var, state="readonly", width=20)
        combo['values'] = [
            "关于x轴对称", "关于y轴对称", "关于y=x对称", "关于y=-x对称", "关于原点对称",
            "缩放(等比例)", "水平收缩与拉伸", "垂直收缩与拉伸", "水平剪切", "垂直剪切"
        ]
        combo.grid(row=0, column=1, padx=5, pady=5)
        combo.current(0)

        tk.Label(dialog, text="参数 k (如需要):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        entry_k = tk.Entry(dialog, width=10)
        entry_k.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        tk.Label(dialog, text="旋转角度 (°):").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        entry_angle = tk.Entry(dialog, width=10)
        entry_angle.grid(row=2, column=1, padx=5, pady=5, sticky='w')

        tk.Label(dialog, text="平移 dx:").grid(row=3, column=0, padx=5, pady=5, sticky='e')
        entry_dx = tk.Entry(dialog, width=10)
        entry_dx.grid(row=3, column=1, padx=5, pady=5, sticky='w')

        tk.Label(dialog, text="平移 dy:").grid(row=4, column=0, padx=5, pady=5, sticky='e')
        entry_dy = tk.Entry(dialog, width=10)
        entry_dy.grid(row=4, column=1, padx=5, pady=5, sticky='w')

        def apply_composite():
            try:
                t1 = combo_var.get()
                # 获取参数（对称变换不需要参数时设为0）
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

            use_gray = self.use_gray.get()
            result = self.img_proc.composite_transform(t1, p1, angle, dx, dy, use_gray)
            if result is not None:
                self.current_image = result
                # 重新计算总矩阵用于 extent（与 composite_transform 内部一致）
                if t1 == "关于x轴对称":
                    M1 = np.array([[1, 0], [0, -1]])
                elif t1 == "关于y轴对称":
                    M1 = np.array([[-1, 0], [0, 1]])
                elif t1 == "关于y=x对称":
                    M1 = np.array([[0, 1], [1, 0]])
                elif t1 == "关于y=-x对称":
                    M1 = np.array([[0, -1], [-1, 0]])
                elif t1 == "关于原点对称":
                    M1 = np.array([[-1, 0], [0, -1]])
                elif t1 == "缩放(等比例)":
                    M1 = np.array([[p1, 0], [0, p1]])
                elif t1 == "水平收缩与拉伸":
                    M1 = np.array([[p1, 0], [0, 1]])
                elif t1 == "垂直收缩与拉伸":
                    M1 = np.array([[1, 0], [0, p1]])
                elif t1 == "水平剪切":
                    M1 = np.array([[1, p1], [0, 1]])
                elif t1 == "垂直剪切":
                    M1 = np.array([[1, 0], [p1, 1]])
                rad = np.radians(angle)
                cos, sin = np.cos(rad), np.sin(rad)
                M_rot_h = np.eye(3)
                M_rot_h[:2, :2] = [[cos, -sin], [sin, cos]]
                M1_h = np.eye(3)
                M1_h[:2, :2] = M1
                M_trans = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
                M_total = M_trans @ M_rot_h @ M1_h
                self.current_extent = self._get_image_extent(result, M_total)
                self._draw_image(result, self.current_extent)
                print("复合变换完成")
            dialog.destroy()

        tk.Button(dialog, text="确认", command=apply_composite).grid(row=5, column=0, columnspan=2, pady=20)

    def open_svd_dialog(self):
        """弹出 SVD 压缩设置窗口"""
        if self.img_proc.image_orig is None:
            print("请先选择图片")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("SVD 压缩")
        dialog.geometry("400x150")

        tk.Label(dialog, text="保留奇异值个数 k:").pack(pady=10)
        max_k = min(self.img_proc.image_orig.shape[:2])
        k_var = tk.IntVar(value=min(50, max_k))
        scale = tk.Scale(dialog, from_=1, to=max_k, orient=tk.HORIZONTAL,
                         variable=k_var, length=300)
        scale.pack()

        def apply_svd():
            k = k_var.get()
            use_gray = self.use_gray.get()
            compressed = self.img_proc.svd_compress(k, use_gray)
            if compressed is not None:
                self.current_image = compressed
                # SVD 压缩不改变几何位置，extent 保持原样
                self.current_extent = self._get_image_extent(compressed)
                self._draw_image(compressed, self.current_extent)
                print(f"SVD 压缩完成，k={k}")
            dialog.destroy()

        tk.Button(dialog, text="确认", command=apply_svd).pack(pady=10)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = Window()
    app.run()