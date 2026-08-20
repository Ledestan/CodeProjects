import sys

sys.dont_write_bytecode = True

import streamlit as st

from db_helper import (count_by_status, delete_ticket, get_all_tickets,
                       update_status)

# 登录验证
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

st.title("管理后台")

if not st.session_state.admin_logged_in:
    pwd = st.text_input("请输入管理密码", type="password")
    if st.button("登录"):
        if pwd == "admin":  # 演示用固定密码
            st.session_state.admin_logged_in = True
            st.rerun()
        else:
            st.error("密码错误")
    st.stop()  # 未登录则不显示后续内容

# 已登录
if st.button("退出管理"):
    st.session_state.admin_logged_in = False
    st.rerun()

# 筛选栏
col1, col2 = st.columns(2)
with col1:
    status_filter = st.selectbox("按状态筛选", ["全部", "待处理", "处理中", "已办结"])
with col2:
    category_filter = st.selectbox(
        "按类别筛选", ["全部", "道路", "水利", "网络", "医疗", "其他"]
    )

df = get_all_tickets(status_filter, category_filter)

# 统计卡片
c1, c2, c3 = st.columns(3)
counts = count_by_status()
c1.metric("待处理", counts["待处理"])
c2.metric("处理中", counts["处理中"])
c3.metric("已办结", counts["已办结"])

st.divider()

if df.empty:
    st.info("暂无工单记录")
    st.stop()

# 渲染表格
for _, row in df.iterrows():
    cols = st.columns([1, 1.2, 1.5, 0.8, 2.8, 1.5, 2.2])
    with cols[0]:
        st.write(row["ticket_no"])
    with cols[1]:
        st.write(row["reporter"])
    with cols[2]:
        st.write(row["category"])
    with cols[3]:
        st.write(row["phone"] or "-")
    with cols[4]:
        st.write(row["description"])
    with cols[5]:
        status = row["status"]
        if status == "待处理":
            st.markdown("待处理")
        elif status == "处理中":
            st.markdown("处理中")
        else:
            st.markdown("已办结")
    with cols[6]:
        ticket_no = row["ticket_no"]
        if status == "待处理":
            if st.button("派单", key=f"assign_{ticket_no}"):
                update_status(ticket_no, "处理中", "管理员")
                st.rerun()
        elif status == "处理中":
            if st.button("办结", key=f"complete_{ticket_no}"):
                update_status(ticket_no, "已办结", "管理员")
                st.rerun()
        else:
            if st.button("删除", key=f"delete_{ticket_no}"):
                delete_ticket(ticket_no)
                st.rerun()
    st.divider()
