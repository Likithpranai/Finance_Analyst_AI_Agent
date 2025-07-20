#!/usr/bin/env python3
"""
Finance Analyst AI Agent - Simple Chatbot Interface

A clean, ChatGPT-like interface for interacting with the Finance Analyst AI Agent.
"""

import os
import sys
import time
import streamlit as st
from finance_analyst_agent import FinanceAnalystReActAgent

# Page configuration
st.set_page_config(
    page_title="Finance Analyst Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ChatGPT-like appearance
st.markdown("""
<style>
    .main {
        background-color: #343541;
        color: white;
    }
    .stTextInput > div > div > input {
        background-color: #40414f;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 15px;
        font-size: 16px;
    }
    .stTextInput > div > div > input:focus {
        box-shadow: none;
        border: 1px solid #6b6c7b;
    }
    .stButton > button {
        background-color: #10a37f;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 8px 16px;
    }
    .stButton > button:hover {
        background-color: #0d8c6d;
    }
    .user-message {
        background-color: #343541;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .assistant-message {
        background-color: #444654;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    h1 {
        color: #10a37f;
        text-align: center;
        margin-bottom: 30px;
    }
    .stMarkdown {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("Finance Analyst AI Chatbot")

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize the Finance Analyst AI Agent
@st.cache_resource
def load_agent():
    try:
        return FinanceAnalystReActAgent()
    except Exception as e:
        st.error(f"Error initializing AI Agent: {str(e)}")
        return None

agent = load_agent()

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"<div class='user-message'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-message'><strong>Finance Analyst:</strong> {message['content']}</div>", unsafe_allow_html=True)

# User input
with st.container():
    user_input = st.text_input("Ask about stocks, crypto, forex, or any financial analysis...", key="user_input")
    
    # Add a submit button
    col1, col2, col3 = st.columns([1, 1, 5])
    with col1:
        submit_button = st.button("Submit")
    with col2:
        clear_button = st.button("Clear Chat")

# Process user input
if submit_button and user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    st.markdown(f"<div class='user-message'><strong>You:</strong> {user_input}</div>", unsafe_allow_html=True)
    
    # Process query with the agent
    if agent:
        with st.spinner("Analyzing your query..."):
            try:
                start_time = time.time()
                response = agent.process_query(user_input)
                end_time = time.time()
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display assistant message
                st.markdown(f"<div class='assistant-message'><strong>Finance Analyst:</strong> {response}</div>", unsafe_allow_html=True)
                st.caption(f"Query processed in {end_time - start_time:.2f} seconds")
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    else:
        st.error("AI Agent not initialized properly. Please check your API keys and try again.")
    
    # Clear the input box
    st.session_state.user_input = ""

# Clear chat history
if clear_button:
    st.session_state.messages = []
    st.experimental_rerun()

# Footer
st.markdown("---")
st.caption("Finance Analyst AI Agent - Powered by Gemini AI")
