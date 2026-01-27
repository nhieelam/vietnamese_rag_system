
import streamlit as st


def apply_custom_styles():
    st.markdown("""
        <style>
        /* User Message Bubble */
        .user-message {
            background-color: #007bff;
            color: white;
            padding: 12px 16px;
            border-radius: 18px;
            margin: 8px 0;
            max-width: 70%;
            margin-left: auto;
            text-align: right;
        }
        
        /* Assistant Message Bubble */
        .assistant-message {
            background-color: #f1f3f4;
            color: #202124;
            padding: 12px 16px;
            border-radius: 18px;
            margin: 8px 0;
            max-width: 70%;
            margin-right: auto;
        }
        
        /* Message Container */
        .message-container {
            display: flex;
            flex-direction: column;
            margin-bottom: 16px;
        }
        
        /* Input Field Styling */
        .stTextInput > div > div > input {
            border-radius: 20px;
        }
        
        /* Sidebar Styling */
        .sidebar-content {
            padding: 10px;
        }
        
        /* Custom Divider */
        hr {
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
