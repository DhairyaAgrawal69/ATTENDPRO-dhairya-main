import streamlit as st

def footer_home():
    
    st.markdown(f"""
        <div style="margin-top:2rem; display:flex; gap:6px; justify-content:center; items-align:center">
                
        <p style="font-weight:bold; color:white;"> Created with ❤️ by </p>  
                
        <a href="https://www.linkedin.com/in/dhairya-agrawal24/" style="text-decoration:none; font-weight:bold; color:#ffb703;">
        DHAIRYA
        </a>
        </div>
                
                """, unsafe_allow_html=True)
    

def footer_dashboard():
    
    st.markdown(f"""
        <div style="margin-top:2rem; display:flex; gap:6px; justify-content:center; items-align:center">
                
        <p style="font-weight:bold; color:black;"> Created with ❤️ by </p>  
                
        <a href="https://www.linkedin.com/in/dhairya-agrawal24/" style="text-decoration:none; font-weight:bold; color:#ffb703;">
        DHAIRYA
        </a>
        </div>
                
                """, unsafe_allow_html=True)
