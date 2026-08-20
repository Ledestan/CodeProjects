import sys

sys.dont_write_bytecode = True

import streamlit as st

from db_helper import init_db

st.set_page_config(
    page_title="数治乡音 · 民生诉求分流系统", page_icon="🏠", layout="wide"
)

init_db()

st.sidebar.title("数治乡音")
st.sidebar.caption("村级民生诉求分流系统")
st.sidebar.markdown("---")
st.sidebar.markdown("""
    **团队信息**\n
    数治乡音 · 智汇乡村\n
    2026年暑期社会实践
    """)
