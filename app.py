import streamlit as st
from agent import intelligent_agent

st.set_page_config(page_title="AI Agent", layout="centered")

st.title("🤖 Intelligent AI Agent")
st.write("Ask anything (math, general knowledge, or chat)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:")

if st.button("Send"):

    if user_input:
        response = intelligent_agent(user_input)

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑 {msg}**")
    else:
        st.markdown(f"**🤖 {msg}**")
