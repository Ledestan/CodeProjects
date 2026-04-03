import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class DataAnalysis:
    def __init__(self, path:str):
        # 加载数据 DataFrame
        self.path = path
        self.data = pd.read_csv(os.path.join(path, 'lagou_data.csv'), encoding ='gbk').reset_index()
        self.df = None

    def data_preview(self):
        """数据预览"""
        print('\n' + '=' * 50 )
        print('列名数组：', ', '.join(self.data.columns.values))
        print('\n' + '=' * 50 )
        print('数据结构预览：\n', self.data.head(2)) 
        print('\n' + '=' * 50 )
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

        print('\n' + '=' * 50 )
        print(f'数据去重处理前：{duplicate_count}')
        print(f'数据去重处理后：{self.data.shape[0]}')

    def missing_process(self):
        """缺失值处理（数据清洗）"""
        print('\n' + '=' * 50 )
        print('缺失值检查：\n', self.data.isnull())

        print('\n' + '=' * 50 )
        missing_stats = pd.DataFrame({'缺失值数量':self.data.isnull().sum(),
                                      '缺失值比例':(self.data.isnull().sum() / self.data.shape[0] * 100).round(2)})
        print('缺失数据：\n', missing_stats)
        self.df = self.data.drop(columns=['industryLables'])
        print(f'清洗后：{self.df.shape}')

        total_empty_count = self.df['label'].isnull().sum()
        print(f'缺失值处理前label字段的缺失数量：{total_empty_count}')

        self.df['label'] = self.df['label'].replace('\'""\'', np.nan)
        total_empty_count = self.df['label'].isnull().sum()
        print(f'缺失值处理后label字段的缺失数量：{total_empty_count}')

        # 保存
        self.df.to_csv(os.path.join(self.path, 'lagou_clean.csv'), encoding ='gbk')

    def feature_extraction(self):
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
        print('\n' + '=' * 50 )
        print(f'连续型特征数量：{len(numerical_cols)}')
        print(numerical_cols)
        print(f'离散型特征数量：{len(catgorical_cols)}')
        print(catgorical_cols)

        # 离散点、异常点
        if numerical_cols:
            numerical_stats = self.df[numerical_cols].describe().round(2)
            print('\n' + '=' * 50 )
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

                print('\n' + '=' * 50 )
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
        print('\n' + '=' * 50 )
        print(f'城市：{cities}')

        for i, j in enumerate(self.df['city']):
            if j not in cities:
                self.df.loc[i, 'city'] = '其它'
        
        # 整合预览
        sns.boxplot(x=self.df['city'], y=self.df['salary'])
        plt.show()

if __name__ == "__main__":
    path = 'data'
    anlys = DataAnalysis(path)
    anlys.data_preview()
    anlys.duplicate()
    anlys.missing_process() 
    anlys.feature_extraction()