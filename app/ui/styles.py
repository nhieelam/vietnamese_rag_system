
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

        /* Comparison layout for RAG vs Co-RAG */
        .compare-title {
            margin: 8px 0 6px 0;
            font-size: 0.9rem;
            font-weight: 600;
            color: #555;
        }

        .compare-card {
            border: 1px solid #e6e6e6;
            border-radius: 12px;
            padding: 8px;
            margin-bottom: 12px;
            background: #fff;
        }

        .compare-rag {
            border-left: 4px solid #2a6fdb;
        }

        .compare-co-rag {
            border-left: 4px solid #1f8f66;
        }

        .compare-label {
            font-size: 0.85rem;
            color: #666;
            margin-bottom: 2px;
        }

        .compare-message {
            max-width: 100%;
            margin: 4px 0;
            border-radius: 10px;
        }

        .compare-time {
            font-size: 0.75rem;
            color: #999;
            margin-top: 4px;
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
