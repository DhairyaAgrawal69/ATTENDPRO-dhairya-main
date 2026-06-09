import streamlit as st
from src.ui.style_base_layout import style_base_layout, style_background_dashboard

from src.components.header import header_dashboard
from src.components.footer import footer_dashboard

def student_screen():

    style_background_dashboard()
    style_base_layout()

    c1, c2 = st.columns(2, vertical_alignment="center",gap = "xxlarge")
    with c1:
        header_dashboard()
    with c2:
        if st.button("Go back to Home", type="secondary", key="loginbackbtn", shortcut = 'ctrl+backspace'):
            st.session_state['login_type']= None
            st.rerun()

    st.header('Login using FaceID', text_alignment='center')
    st.space()
    st.space()
    
    st.camera_input("Position your face in the center")
    footer_dashboard()
