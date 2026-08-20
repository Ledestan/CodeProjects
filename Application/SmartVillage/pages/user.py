import sys
sys.dont_write_bytecode = True
import streamlit as st
from db_helper import add_ticket, get_ticket_by_no

st.title("提交诉求")
st.caption("请填写以下信息，我们会在第一时间为您处理")

# 如果提交成功，显示成功信息
if st.session_state.get("submit_success", False):
    st.success(f"提交成功！工单编号：{st.session_state.get('last_ticket_no', '')}")
    st.info("请保存此编号，以便查询办理进度")
    if st.button("继续提交"):
        st.session_state.submit_success = False
        st.rerun()
    st.divider()

# 提交表单
with st.form(key="submit_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        reporter = st.text_input("您的姓名", placeholder="请输入真实姓名")
    with col2:
        phone = st.text_input("联系方式", placeholder="手机号（选填）")
    
    category = st.selectbox("问题类别", ["道路", "水利", "网络", "医疗", "其他"])
    description = st.text_area("问题描述", placeholder="请详细描述您遇到的问题", height=120)
    
    submitted = st.form_submit_button("提交诉求", type="primary", use_container_width=True)
    
    if submitted:
        if not reporter.strip():
            st.error("请填写您的姓名")
        elif not description.strip():
            st.error("请填写问题描述")
        else:
            ticket_no = add_ticket(reporter.strip(), phone.strip(), category, description.strip())
            st.session_state.submit_success = True
            st.session_state.last_ticket_no = ticket_no
            st.rerun()

# 查询进度
st.divider()
st.subheader("查询办理进度")
col1, col2 = st.columns([3, 1])
with col1:
    query_no = st.text_input("请输入工单编号", placeholder="如 T20260801001", label_visibility="collapsed")
with col2:
    query_btn = st.button("查询", use_container_width=True)

if query_btn:
    if not query_no.strip():
        st.warning("请输入工单编号")
    else:
        ticket = get_ticket_by_no(query_no.strip())
        if ticket is None:
            st.error("未找到该工单，请检查编号是否正确")
        else:
            status_map = {"待处理": "待处理", "处理中": "处理中", "已办结": "已办结"}  # 无表情
            st.info(f"""
                工单 {ticket['ticket_no']}
                类别：{ticket['category']}
                描述：{ticket['description']}
                当前状态：{ticket['status']}
                提交时间：{ticket['created_at']}
                {f"处理人：{ticket['handler']}" if ticket['handler'] else ""}
            """)