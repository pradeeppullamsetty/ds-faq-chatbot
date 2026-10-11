import os
import time
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from dotenv import load_dotenv

import config
import rag
import analytics
import avatar
import voice

load_dotenv()

# Page Setup
st.set_page_config(
    page_title=f"{config.COLLEGE_NAME} - FAQ Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Database
analytics.initialize_analytics_db()
analytics.seed_demo_data()

# Load CSS
def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/style.css")

# Add animated background orbs
st.markdown("""
<div class="orb one"></div>
<div class="orb two"></div>
<div class="orb three"></div>
""", unsafe_allow_html=True)

# Lottie animation loader
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# Load knowledge base cache
@st.cache_resource(show_spinner=False)
def get_cached_rag():
    # Detect if we should use Qdrant (self-hosted) or Chroma (local/cloud)
    qdrant_url = os.getenv("QDRANT_URL")

    docs, chunks, vs = rag.load_and_index_documents()

    if qdrant_url:
        print("Using Qdrant vector store as configured in environment.")
        vs = rag.get_qdrant_vector_store(chunks)
    else:
        print("Using Chroma vector store (default).")
        vs = rag.get_vector_store(chunks)

    return docs, chunks, vs

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "language" not in st.session_state:
    st.session_state.language = "en"

# Navigation Sidebar
with st.sidebar:
    st.image(config.LOGO_PATH, width=70)
    st.markdown(f"### **{config.COLLEGE_NAME}**")
    st.caption(config.TAGLINE)
    st.markdown("---")

    selected_page = option_menu(
        menu_title=None,
        options=["Home", "Chat", "Analytics", "Knowledge Base", "About"],
        icons=["house", "chat-dots", "graph-up", "book", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#38bdf8", "font-size": "16px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "4px",
                "color": "#e2e8f0",
                "border-radius": "10px"
            },
            "nav-link-selected": {"background-color": "rgba(56, 189, 248, 0.2)", "color": "#38bdf8", "font-weight": "bold"},
        }
    )

    st.markdown("---")
    # Voice / Avatar toggle
    voice_enabled = st.sidebar.checkbox("🔊 Voice & Avatar", value=True)

    # Language Toggle
    lang_choice = st.selectbox("🌐 Language / மொழி", ["English", "தமிழ் (Tamil)"], index=0 if st.session_state.language == "en" else 1)
    st.session_state.language = "en" if lang_choice == "English" else "ta"

    # API Key check
    groq_api_key = os.getenv("GROQ_API_KEY", "") or st.secrets.get("GROQ_API_KEY", "")
    if not groq_api_key:
        st.warning("⚠️ GROQ_API_KEY not found in .env or Secrets.")

# --- 1. HOME PAGE ---
if selected_page == "Home":
    col1, col2 = st.columns([1.2, 0.8])
    with col1:
        st.markdown(f"""
        <h1 style='font-size: 2.8rem; font-weight: 800; background: linear-gradient(90deg, #8b5cf6, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Welcome to {config.COLLEGE_NAME} AI Assistant
        </h1>
        <p style='font-size: 1.15rem; color: #94a3b8;'>
            Instant, accurate answers powered by Retrieval-Augmented Generation (RAG) and Groq Llama 3.1 LLM.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Start Asking Questions", use_container_width=False):
            st.session_state.nav_target = "Chat"
            st.rerun()

    with col2:
        lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_m9zragkd.json")
        if lottie_ai:
            st_lottie(lottie_ai, height=240, key="home_anim")
        else:
            st.markdown("""
            <div class="glass" style="padding: 40px; text-align: center;">
                <h1 style="font-size: 60px;">🤖</h1>
                <h3>AI Ready to Assist</h3>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # Stat counters
    try:
        docs, chunks, vs = get_cached_rag()
        stats = rag.get_document_stats(docs, chunks, vs)
        analytics_summary = analytics.get_analytics_summary()
    except Exception:
        stats = {"total_documents": 5, "total_chunks": 40}
        analytics_summary = {"total_queries": 10, "satisfaction_rate": 95}

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""
        <div class="glass kpi" style="text-align:center;">
            <h4 style="color:#94a3b8; margin-bottom:0;">Indexed Documents</h4>
            <h2 style="color:#38bdf8; font-size: 2.2rem; margin: 4px 0;">{stats.get('total_documents', 5)}</h2>
            <small>Admissions, Fees, Placements</small>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="glass kpi" style="text-align:center;">
            <h4 style="color:#94a3b8; margin-bottom:0;">Knowledge Chunks</h4>
            <h2 style="color:#8b5cf6; font-size: 2.2rem; margin: 4px 0;">{stats.get('total_chunks', 42)}</h2>
            <small>Vectorized & Embedded</small>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="glass kpi" style="text-align:center;">
            <h4 style="color:#94a3b8; margin-bottom:0;">Questions Answered</h4>
            <h2 style="color:#38bdf8; font-size: 2.2rem; margin: 4px 0;">{analytics_summary.get('total_queries', 10)}</h2>
            <small>Real-time interactions</small>
        </div>
        """, unsafe_allow_html=True)
    with s4:
        st.markdown(f"""
        <div class="glass kpi" style="text-align:center;">
            <h4 style="color:#94a3b8; margin-bottom:0;">Satisfaction Rate</h4>
            <h2 style="color:#8b5cf6; font-size: 2.2rem; margin: 4px 0;">{analytics_summary.get('satisfaction_rate', 95)}%</h2>
            <small>Based on feedback</small>
        </div>
        """, unsafe_allow_html=True)

# --- 2. CHAT PAGE ---
elif selected_page == "Chat":
    st.markdown("""
    <h2 style='background: linear-gradient(90deg, #8b5cf6, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        💬 College Assistant Chat
    </h2>
    <p style='color: #94a3b8;'>Ask any questions about admissions, fees, hostel rules, courses, and placements.</p>
    """, unsafe_allow_html=True)

    # Show avatar when enabled
    if voice_enabled:
        st.markdown(avatar.render_avatar(), unsafe_allow_html=True)

    # Suggested quick questions
    suggested_cols = st.columns(3)
    suggested = [
        "What is the highest package in placements?",
        "What are the hostel entry and exit timings?",
        "How much is the tuition fee for B.Tech CSE?"
    ]

    clicked_prompt = None
    for i, q in enumerate(suggested):
        with suggested_cols[i]:
            if st.button(f"💡 {q}", key=f"sugg_{i}", use_container_width=True):
                clicked_prompt = q

    st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)

    # Render Chat History
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", "")

        avatar = "🎓" if role == "assistant" else "👤"
        bubble_class = "assistant" if role == "assistant" else "user"

        st.markdown(f"""
        <div class="msg-row fade-slide">
            <div class="avatar">{avatar}</div>
            <div class="bubble {bubble_class}">
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if sources and role == "assistant":
            with st.expander("📚 View Supporting Sources"):
                st.markdown(f"<small>{sources}</small>", unsafe_allow_html=True)

    # Handle Input
    user_input = st.chat_input("Type your question here...") or clicked_prompt

    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message instantly
        st.markdown(f"""
        <div class="msg-row fade-slide">
            <div class="avatar">👤</div>
            <div class="bubble user">{user_input}</div>
        </div>
        """, unsafe_allow_html=True)

        # Process with RAG
        with st.spinner("Searching documents & generating answer..."):
            try:
                docs, chunks, vs = get_cached_rag()
                result = rag.process_query(user_input, chunks, vs, language=st.session_state.language)

                answer = result["answer"]
                sources = result["sources"]
                resp_time = result["response_time"]

                # Check if it was answered from docs or fallback
                is_answered = (answer != "I don't have that information. Please contact the college office.")

                # Translate fallback if Tamil
                if not is_answered and st.session_state.language == "ta":
                    answer = "என்னிடம் அந்த தகவல் இல்லை. கல்லூரி அலுவலகத்தை தொடர்பு கொள்ளவும்."

                # Simulate streaming effect
                placeholder = st.empty()
                full_text = ""
                for word in answer.split(" "):
                    full_text += word + " "
                    placeholder.markdown(f"""
                    <div class="msg-row fade-slide">
                        <div class="avatar">🎓</div>
                        <div class="bubble assistant">{full_text}▌</div>
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(0.015)

                placeholder.markdown(f"""
                <div class="msg-row fade-slide">
                    <div class="avatar">🎓</div>
                    <div class="bubble assistant">{answer}</div>
                </div>
                """, unsafe_allow_html=True)

                if sources and is_answered:
                    with st.expander("📚 View Supporting Sources"):
                        st.markdown(f"<small>{sources}</small>", unsafe_allow_html=True)

                # Trigger voice (free, edges.tts); mute on fallback if preferred; here plays for all
                if voice_enabled:
                    audio_bytes = voice.get_audio_bytes(answer, locale="en-IN", rate="+20%", volume="+0%")
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")

                # Append to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

                # Log to SQLite Analytics
                analytics.log_query(
                    question=user_input,
                    answered=is_answered,
                    response_time=resp_time,
                    language=st.session_state.language,
                    sources_text=sources
                )

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

# --- 3. ANALYTICS PAGE ---
elif selected_page == "Analytics":
    st.markdown("""
    <h2 style='background: linear-gradient(90deg, #8b5cf6, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        📊 Real-Time Analytics Dashboard
    </h2>
    <p style='color: #94a3b8;'>Monitor chatbot performance, usage trends, and identify knowledge gaps.</p>
    """, unsafe_allow_html=True)

    summary = analytics.get_analytics_summary()

    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Questions", summary["total_queries"])
    with k2:
        st.metric("Answer Rate", f"{summary['answer_rate']}%")
    with k3:
        st.metric("Avg Response Time", f"{summary['avg_response_time']}s")
    with k4:
        st.metric("Satisfaction Rate", f"{summary['satisfaction_rate']}%")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        # Donut Chart: Answered vs Unanswered
        answered = summary["answered_queries"]
        unanswered = summary["total_queries"] - answered
        df_pie = pd.DataFrame({
            "Status": ["Answered", "Unanswered / Fallback"],
            "Count": [answered, unanswered]
        })
        fig_pie = px.pie(
            df_pie, values="Count", names="Status",
            hole=0.55,
            color_discrete_sequence=["#38bdf8", "#f43f5e"],
            title="Answered vs Unanswered Breakdown"
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8fafc"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        # Line/Area Chart: Activity Over Time
        q_days = summary["questions_per_day"]
        if q_days:
            df_time = pd.DataFrame(list(q_days.items()), columns=["Date", "Questions"])
            fig_time = px.bar(
                df_time, x="Date", y="Questions",
                color_discrete_sequence=["#8b5cf6"],
                title="Questions Volume Over Time"
            )
            fig_time.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc"
            )
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No timeline data available yet.")

    # Unanswered questions table
    st.markdown("### ⚠️ Questions Needing College Office Attention")
    unanswered_list = summary.get("unanswered_questions", [])
    if unanswered_list:
        df_unanswered = pd.DataFrame({"Unanswered Question": unanswered_list})
        st.dataframe(df_unanswered, use_container_width=True)
    else:
        st.success("All received questions were successfully answered from documentation!")

    # Export & Clear Data
    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns([1, 1])
    with b1:
        csv_data = analytics.get_analytics_data()
        if csv_data:
            df_export = pd.DataFrame(csv_data)
            st.download_button(
                "📥 Download Report (CSV)",
                df_export.to_csv(index=False),
                file_name="chatbot_analytics_report.csv",
                mime="text/csv"
            )
    with b2:
        if st.button("🗑️ Reset / Clear Analytics Data"):
            analytics.clear_analytics_data()
            st.success("Analytics reset with initial seed data.")
            st.rerun()

# --- 4. KNOWLEDGE BASE PAGE ---
elif selected_page == "Knowledge Base":
    st.markdown("""
    <h2 style='background: linear-gradient(90deg, #8b5cf6, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        📚 Knowledge Base Manager
    </h2>
    <p style='color: #94a3b8;'>Manage documents, preview chunks, and rebuild the vector database.</p>
    """, unsafe_allow_html=True)

    docs, chunks, vs = get_cached_rag()

    # Documents listing
    st.markdown("### 📄 Indexed Documents")
    doc_sources = set(d.metadata.get("source", "Unknown") for d in docs)

    for src in sorted(doc_sources):
        src_chunks = [c for c in chunks if c.metadata.get("source") == src]
        st.markdown(f"""
        <div class="glass" style="padding: 12px 18px; margin-bottom: 8px; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <strong>📄 {src}</strong>
            </div>
            <span class="source-chip">{len(src_chunks)} chunks</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload new files
    st.markdown("### 🌐 DS University Live Ingestion (Auto-discovery)")
    with st.expander("Fetch a limited set of dsuniversity.ac.in pages into RAG (recommended for live demo)"):
        start_url = st.text_input("Start URL", value="https://www.dsuniversity.ac.in/")
        max_pages = st.number_input("Max pages to ingest", min_value=5, max_value=50, value=20, step=1)
        respect_robots = st.checkbox("Respect robots.txt", value=True)

        if st.button("⬇️ Fetch DS University pages and save to data/web/"):
            with st.spinner("Fetching pages and building knowledge base files..."):
                # Fetch web pages into txt files
                try:
                    rag.fetch_and_ingest_site(
                        start_url=start_url,
                        output_dir="data/web",
                        max_pages=int(max_pages),
                        respect_robots=respect_robots,
                    )
                    st.success("Web pages fetched. Now rebuild the index.")
                except Exception as e:
                    st.error(f"Web ingestion failed: {str(e)}")

    st.markdown("### 📤 Upload New College Documents")
    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded_files:
        for up in uploaded_files:
            save_path = os.path.join("data", up.name)
            with open(save_path, "wb") as f:
                f.write(up.getbuffer())
        st.success(f"Saved {len(uploaded_files)} files to data/ folder!")

    # Rebuild Index Button
    if st.button("🔄 Rebuild Knowledge Base Vector Index"):
        with st.spinner("Re-chunking and rebuilding Chroma vector index..."):
            rag.clear_vector_store()
            st.cache_resource.clear()
            get_cached_rag()
            st.success("Knowledge Base successfully rebuilt!")
            st.rerun()

    # Optional: quick ingest -> rebuild shortcut
    if st.button("✨ Fetch DS University + Rebuild Index"):
        with st.spinner("Fetching dsuniversity.ac.in pages..."):
            rag.fetch_and_ingest_site(
                start_url="https://www.dsuniversity.ac.in/",
                output_dir="data/web",
                max_pages=20,
                respect_robots=True,
            )
            rag.clear_vector_store()
            st.cache_resource.clear()
            get_cached_rag()
            st.success("Live dsuniversity content ingested and index rebuilt!")
            st.rerun()

    # Search & Preview Chunks
    st.markdown("<br>### 🔍 Chunk Preview Search")
    search_q = st.text_input("Enter search keywords to preview retrieved chunks:")
    if search_q:
        results = vs.similarity_search(search_q, k=3)
        for r in results:
            st.markdown(f"""
            <div class="glass" style="padding: 14px; margin-bottom: 10px;">
                <small style="color: #38bdf8;"><strong>Source:</strong> {r.metadata.get('source')} | <strong>Chunk ID:</strong> {r.metadata.get('chunk_id')}</small>
                <p style="margin-top: 6px; font-size: 0.95rem;">{r.page_content}</p>
            </div>
            """, unsafe_allow_html=True)

# --- 5. ABOUT PAGE ---
elif selected_page == "About":
    st.markdown(f"""
    <h2 style='background: linear-gradient(90deg, #8b5cf6, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        🎓 About the Project
    </h2>
    <p style='color: #94a3b8;'>AI-Powered College FAQ Chatbot Using LLM and RAG</p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass" style="padding: 20px; margin-bottom: 20px;">
        <h3>📌 Problem Statement</h3>
        <p>
            Prospective and current college students frequently have queries regarding admissions, fee structures,
            hostel rules, courses, and placements. Manually answering these inquiries via phone or email leads to delays
            and office overhead. This project provides a 24/7 intelligent conversational chatbot that strictly relies
            on official college documentation to answer questions, preventing hallucinations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏗️ System Architecture (RAG Flow)")
    st.markdown("""
    ```mermaid
    flowchart LR
        A[College Documents\nPDF / TXT] --> B[Text Splitter\n800 Chars]
        B --> C[Sentence Transformers\nall-MiniLM-L6-v2]
        C --> D[(Chroma Vector DB)]
        E[Student Question] --> F[Vector Search\nTop 4 Chunks]
        D --> F
        F --> G[Groq LLM\nLlama 3.1 8B]
        G --> H[Accurate Answer\n+ Source Attribution]
    ```
    """)

    st.markdown("### 🛠️ Tech Stack Badges")
    st.markdown("""
    - **Language:** Python 3.10+
    - **Frontend:** Streamlit, Streamlit Option Menu, Streamlit Lottie
    - **RAG Framework:** LangChain
    - **Vector Store:** Chroma DB (Local Persistent)
    - **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Sentence-Transformers)
    - **LLM Engine:** Groq API (`llama-3.1-8b-instant`)
    - **Analytics:** SQLite3 & Plotly
    - **Testing:** Pytest
    """)
