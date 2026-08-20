import sys

sys.dont_write_bytecode = True

import time

import plotly.express as px
import streamlit as st

from db_helper import (count_by_category, count_by_status, get_all_tickets,
                       get_recent_tickets)

st.set_page_config(layout="wide")  # 大屏全屏
st.title("村民生诉求实时看板")

# 手动刷新按钮
if st.button("刷新数据"):
    st.rerun()
st.caption("页面每15秒自动刷新")

# 统计卡片
counts = count_by_status()
col1, col2, col3 = st.columns(3)
col1.metric("待处理", counts["待处理"])
col2.metric("处理中", counts["处理中"])
col3.metric("已办结", counts["已办结"])

st.divider()

# 图表与最新动态
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("各类别未办结问题分布")
    df_cat = count_by_category()
    if not df_cat.empty:
        fig = px.bar(df_cat, x="category", y="count", color="category", text="count")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("所有问题均已办结")

with col2:
    st.subheader("最新诉求")
    recent = get_recent_tickets(5)
    if not recent.empty:
        for _, row in recent.iterrows():
            st.markdown(
                f"""
                **{row['ticket_no']}**  {row['status']}  
                {row['category']} · {row['reporter']}  
                <span style="color:gray;font-size:12px;">{row['created_at'][5:16]}</span>
            """,
                unsafe_allow_html=True,
            )
            st.divider()
    else:
        st.info("暂无工单")

# 全部工单一览
st.divider()
st.subheader("全部工单状态一览")
df_all = get_all_tickets()
if not df_all.empty:
    df_display = df_all[
        ["ticket_no", "category", "description", "status", "updated_at"]
    ].head(10)
    df_display.columns = ["工单编号", "类别", "描述", "状态", "更新时间"]
    st.dataframe(df_display, use_container_width=True, hide_index=True)
else:
    st.info("暂无工单记录")

# 自动刷新（每15秒）
time.sleep(15)
st.rerun()
