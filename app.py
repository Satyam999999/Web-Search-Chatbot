import os, streamlit as st
from langchain_groq import ChatGroq
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun

# 1. Safe wrapper
class SafeDuckDuckGo(DuckDuckGoSearchRun):
    def _run(self, query: str) -> str:
        try:
            return super()._run(query)
        except UnboundLocalError:
            return "⚠️ DuckDuckGo returned no results."

# 2. Instantiate tools
arxiv  = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200))
wiki   = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200))
search = SafeDuckDuckGo(name="Safe‑DuckDuckGo")

st.title("🔎 LangChain Search Bot")
api_key = st.sidebar.text_input("Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"Hi! What shall we look up today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 3. Validate prompt
prompt = st.chat_input("Ask me anything…")
if not prompt or not prompt.strip():
    st.warning("Please type a non‑empty query.")
else:
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm   = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    agent = initialize_agent(
        [search, arxiv, wiki],
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    with st.chat_message("assistant"):
        cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            result = agent.run(prompt, callbacks=[cb])
        except Exception as e:
            result = f"❌ Error: {e}"
        st.session_state.messages.append({"role":"assistant","content":result})
        st.write(result)
