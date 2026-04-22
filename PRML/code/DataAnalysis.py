"""
数据处理工具

创建日期：2026-03-20
需求文件：data/lagou_data.csv

依赖库：
pandas>=3.0.1
numpy>=2.2.6
seaborn>=0.13.2
matplotlib>=3.10.8
scikit-learn>=1.8.0
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def classify_salary(salary):
    """根据薪资金额分类薪资区间"""
    if salary < 15000:
        return '15k以下'
    elif salary <= 25000:
        return '15-25k'
    elif salary <= 35000:
        return '25-35k'
    else:
        return '35k以上'


class DataAnalysis:
    def __init__(self, path: str):
        """加载数据 DataFrame"""
        self.path = path
        self.data = pd.read_csv(os.path.join(path, 'lagou_data.csv'), encoding='gbk').reset_index()
        self.df = None

    def data_preview(self):
        """数据预览"""
        print('\n' + '=' * 50)
        print('列名数组：', ', '.join(self.data.columns.values))
        print('\n' + '=' * 50)
        print('数据结构预览：\n', self.data.head(2))
        print('\n' + '=' * 50)
        print(f'数据维度：{self.data.shape}')
        print(f'数据基本信息：{self.data.info()}')
        print(f'数据描述：{self.data.describe()}')

        # 直方图
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(self.data['salary'], bins=20)
        plt.hist(self.data['salary'], bins=20)

        # 密度图
        plt.subplot(1, 2, 2)
        sns.histplot(self.data['salary'], bins=20, kde=True, edgecolor=None)
        sns.histplot(self.data['salary'], bins=20, kde=True, edgecolor=None)
        plt.show()

    def duplicate(self):
        """数据去重"""
        duplicate_count = self.data.duplicated().sum()
        self.data = self.data.drop_duplicates(keep='first')

        print('\n' + '=' * 50)
        print(f'数据去重处理前：{duplicate_count}')
        print(f'数据去重处理后：{self.data.shape[0]}')

    def missing_process(self):
        """缺失值处理（数据清洗）"""
        print('\n' + '=' * 50)
        print('缺失值检查：\n', self.data.isnull())

        print('\n' + '=' * 50)
        missing_stats = pd.DataFrame({'缺失值数量': self.data.isnull().sum(),
                                      '缺失值比例': (self.data.isnull().sum() / self.data.shape[0] * 100).round(2)})
        print('缺失数据：\n', missing_stats)
        self.df = self.data.drop(columns=['industryLables'])
        print(f'清洗后：{self.df.shape}')

        total_empty_count = self.df['label'].isnull().sum()
        print(f'缺失值处理前label字段的缺失数量：{total_empty_count}')

        self.df['label'] = self.df['label'].replace('\'""\'', np.nan)
        total_empty_count = self.df['label'].isnull().sum()
        print(f'缺失值处理后label字段的缺失数量：{total_empty_count}')

        # 保存
        # self.df.to_csv(os.path.join(self.path, 'lagou_clean.csv'), encoding ='gbk')

    def feature_extraction(self):
        """特征提取"""
        # 盒图
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.boxplot(self.df['salary'])

        plt.subplot(1, 2, 2)
        plt.boxplot(self.df['salary'], vert=True)
        plt.show()

        # 连续型特征、离散型（分类）特征的提取
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        catgorical_cols = self.df.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
        print('\n' + '=' * 50)
        print(f'连续型特征数量：{len(numerical_cols)}')
        print(numerical_cols)
        print(f'离散型特征数量：{len(catgorical_cols)}')
        print(catgorical_cols)

        # 离散点、异常点
        if numerical_cols:
            numerical_stats = self.df[numerical_cols].describe().round(2)
            print('\n' + '=' * 50)
            print('统计摘要表：\n', numerical_stats)

            # 检查异常、利用IQR（四分位数极差）
            for col in numerical_cols:
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                ouliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                oulier_ratio = len(ouliers) / self.df.shape[0] * 100 if len(ouliers) else 0

                print('\n' + '=' * 50)
                print(f'{col}')
                print(f'正常范围：{lower_bound:.2f} - {upper_bound:.2f}')
                print(f'异常值数量：{len(ouliers)}')
                print(f'异常值比例:{oulier_ratio}')

        # 离散型数据分析
        sns.boxplot(x=self.df['city'], y=self.df['salary'])
        plt.show()

        # 处理 city 特征
        city_counts_df = self.df['city'].value_counts().reset_index()
        city_counts_df.columns = ['city', 'count']
        cities = city_counts_df[city_counts_df['count'] >= 30]['city'].tolist()
        print('\n' + '=' * 50)
        print(f'城市：{cities}')

        for i, j in enumerate(self.df['city']):
            if j not in cities:
                self.df.loc[i, 'city'] = '其它'

        # 整合预览
        sns.boxplot(x=self.df['city'], y=self.df['salary'])
        plt.show()

    def mutual_info_analysis(self, k=10):
        """互信息分析：计算特征与薪资区间之间的互信息得分，并绘制TopK特征柱状图"""
        # 生成薪资区间
        if 'salary' in self.df.columns:
            salary_col = 'salary'
        else:
            salary_col = self.df.columns[self.df.columns.str.contains('salary|薪资')][0]
        self.df['salary_range'] = self.df[salary_col].apply(classify_salary)

        print('\n' + '=' * 50)
        print(f'新增薪资区间列 salary_range，取值分布:')
        print(self.df['salary_range'].value_counts())

        # 特征编码
        # 有序特征（LabelEncoder）
        ordinal_features = ['work_year', 'education', 'size_standard']
        ordinal_features = [f for f in ordinal_features if f in self.df.columns]
        for col in ordinal_features:
            self.df[col + '_encoded'] = LabelEncoder().fit_transform(self.df[col])

        # 无序特征（OneHotEncoder）
        nominal_features = ['industry', 'position_name', 'city', 'extracted_skill']
        nominal_features = [f for f in nominal_features if f in self.df.columns]
        onehot = OneHotEncoder(sparse_output=False, drop='first')
        nominal_encoded = onehot.fit_transform(self.df[nominal_features])

        # 合并特征矩阵
        ordinal_encoded = self.df[[col + '_encoded' for col in ordinal_features]].values
        x = np.hstack([nominal_encoded, ordinal_encoded])
        y = LabelEncoder().fit_transform(self.df['salary_range'])

        # 计算互信息得分
        mi_scores = mutual_info_classif(x, y, random_state=42)

        # 构造特征名称列表
        ohe_feature_names = onehot.get_feature_names_out(nominal_features).tolist()
        ordinal_feature_names = [col + '_encoded' for col in ordinal_features]
        all_feature_names = ohe_feature_names + ordinal_feature_names

        # 整理得分并排序（取前k个）
        mi_series = pd.Series(mi_scores, index=all_feature_names)
        mi_series_topk = mi_series.sort_values(ascending=False).head(k)

        # 绘图
        mi_df = mi_series_topk.reset_index()
        mi_df.columns = ['特征名称', '互信息得分']

        max_score = mi_df['互信息得分'].max()
        step = 0.02
        xmax = np.ceil(max_score / step) * step

        plt.figure(figsize=(12, 6))
        bars = plt.barh(mi_df['特征名称'], mi_df['互信息得分'])

        # 数值标签（跳过最高分）
        max_score = mi_df['互信息得分'].max()
        for i, bar in enumerate(bars):
            score = mi_df['互信息得分'].iloc[i]
            if score != max_score:
                plt.text(score + 0.001, bar.get_y() + bar.get_height() / 2,
                         f'{score:.3f}', va='center', fontsize=10)

        plt.title(f'特征与工资水平（salary_range）的互信息得分（Top{k}）')
        plt.xlabel('互信息得分（越高影响力越强）')
        plt.ylabel('特征名称')
        plt.xticks(np.arange(0, xmax + step, step))
        plt.tight_layout()
        plt.show()

        # 保存
        # plt.savefig(f'data/lagou_Top{k}.png', dpi=300, bbox_inches='tight')

        print('\n' + '=' * 50)
        print(f'特征与工资水平互信息得分（Top{k}）')
        print(mi_df.round(3))


if __name__ == "__main__":
    path = 'data'
    anlys = DataAnalysis(path)
    # anlys.data_preview()
    anlys.duplicate()
    anlys.missing_process()
    # anlys.feature_extraction()
    anlys.mutual_info_analysis()