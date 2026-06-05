# Healthcare Appointment Dataset

## 元数据

| 项目 | 内容 |
|------|------|
| 主题（Topic） | 医疗健康 |
| 领域（Field） | 数据挖掘、机器学习/分类 |
| 许可证（License） | Apache 2.0 - 作者保留版权，他人有使用权 |
| 文件格式（Ext） | .csv |

## 数据来源

[Healthcare Appointment Dataset](https://www.kaggle.com/datasets/wajahat1064/healthcare-appointment-dataset)

## 数据集简介

该数据集包含有关某人是否会出现在医疗预约中的数据。  
数据集共有 **107K 行**以及 **15 列**，可用于预测患者是否会前往就诊。

## 字段说明

| 字段名 | 说明 |
|--------|------|
| PatientId | 患者 ID |
| AppointmentID | 预约 ID |
| Gender | 性别 |
| ScheduledDay | 预约日期 |
| AppointmentDay | 就诊日期 |
| Age | 年龄 |
| Neighbourhood | 医院位置 |
| Scholarship | 是否参加巴西福利项目/家庭津贴 |
| Hipertension | 是否高血压 |
| Diabetes | 是否糖尿病 |
| Alcoholism | 是否酗酒 |
| Handcap | 是否残障 |
| SMS_received | 患者是否收到短信通知 |
| Date.diff | 就诊日期与预约日期的时间差 |
| Showed_up | **目标变量**：`no` 表示病人如约就诊，`yes` 表示病人没有前往就诊 |

## 使用场景

我们可以使用这些数据来预测是否有人会去看医生。