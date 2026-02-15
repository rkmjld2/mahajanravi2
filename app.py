# app.py - Blood Report Analyzer (FIXED for TiDB Cloud SSL on Streamlit)

import streamlit as st
import pandas as pd
from io import StringIO
from datetime import datetime
import mysql.connector
import os
import tempfile

# LangChain / Groq imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

#from langchain.chains import create_retrieval_chain
#from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_groq import ChatGroq

st.set_page_config(page_title="🩸 Blood Report Analyzer", layout="wide")

# ── 1. SECURE SECRETS CHECK
required_keys = [
    "GROQ_API_KEY",
    "connections.databases.default.host",
    "connections.databases.default.port",
    "connections.databases.default.username",
    "connections.databases.default.password",
    "connections.databases.default.database",
    "TIDB_SSL_CA"
]

missing = []
for key in required_keys:
    keys = key.split(".")
    d = st.secrets
    try:
        for k in keys[:-1]:
            d = d[k]
        if keys[-1] not in d:
            missing.append(key)
    except:
        missing.append(key)

if missing:
    st.error(f"🚨 Missing secrets: {', '.join(missing)}")
    st.info("Please check Streamlit Cloud → Settings → Secrets (TOML format)")
    st.stop()

st.success("✅ Secrets verified")

# Set Groq key
st.session_state.groq_api_key = st.secrets["GROQ_API_KEY"]

# ── 2. Write CA certificate to temp file (required for mysql-connector)
@st.cache_resource(show_spinner=False)
def ensure_ca_file():
    ca_content = st.secrets["TIDB_SSL_CA"].strip()
    if "-----BEGIN CERTIFICATE-----" not in ca_content:
        st.error("❌ TIDB_SSL_CA in secrets is invalid (missing certificate header)")
        st.stop()

    ca_path = "/tmp/tidb_ca.pem"
    if not os.path.exists(ca_path):
        with open(ca_path, "w") as f:
            f.write(ca_content)
        st.sidebar.info("🔐 TiDB CA certificate written to disk")

    return ca_path

# ── 3. Database Connection
@st.cache_resource
def get_db_connection():
    ensure_ca_file()  # Make sure file exists

    db_config = st.secrets["connections"]["databases"]["default"]
    ca_path = "/tmp/tidb_ca.pem"

    try:
        conn = mysql.connector.connect(
            host=db_config["host"],
            port=int(db_config["port"]),
            user=db_config["username"],
            password=db_config["password"],
            database=db_config["database"],
            connect_timeout=20,
            ssl_ca=ca_path,
            ssl_verify_cert=True,
            ssl_verify_identity=True
        )
        return conn
    except mysql.connector.Error as e:
        st.error(f"❌ TiDB connection failed: {e.msg} (errno {e.errno})")
        if "SSL" in str(e).upper():
            st.info("Common causes: wrong CA content, expired cert, or host mismatch")
        raise

# Test connection button in sidebar
def test_tidb_connection():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1;")
        result = cursor.fetchone()
        conn.close()
        st.sidebar.success(f"✅ TiDB connected! Test query OK: {result}")
    except Exception as e:
        st.sidebar.error(f"❌ Connection test failed: {str(e)}")

# ── 4. Embeddings (cached)
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

embeddings = load_embeddings()

# ── 5. Session state init
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None

# ── UI ──────────────────────────────────────────────────────────────
st.title("🩸 Blood Report Analyzer – Groq + TiDB Cloud")
st.caption("Fixed version | Secure secrets | Saves to TiDB")

with st.sidebar:
    st.markdown("### Status")
    st.success("Systems ready")
    if st.button("🔍 Test TiDB Connection"):
        test_tidb_connection()

tab1, tab2 = st.tabs(["📊 Upload & Analyze", "ℹ️ Instructions"])

with tab1:
    raw_text = st.text_area(
        "1. Paste your blood report (CSV-like format)",
        height=250,
        value="""Test,Result,Unit,Reference Range,Flag
Hemoglobin,12.4,g/dL,13.0-17.0,L
WBC,8.2,10^3/µL,4.0-11.0,
Glucose Fasting,102,mg/dL,70-99,H
Creatinine,1.1,mg/dL,0.6-1.2,
ALT,45,U/L,7-56,
Total Cholesterol,210,mg/dL,<200,H""",
        help="Copy from PDF / lab report / WhatsApp"
    )

    if st.button("🔍 2. Parse Table", type="primary", use_container_width=True):
        if raw_text.strip():
            try:
                df = pd.read_csv(StringIO(raw_text), sep=None, engine="python")
                df = df.dropna(how="all").reset_index(drop=True)
                st.session_state.df = df
                st.success(f"Parsed {len(df)} tests")
            except Exception as e:
                st.error(f"Parse failed: {str(e)}")

    if st.session_state.df is not None:
        st.subheader("3. Edit Results")
        edited_df = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Test": st.column_config.TextColumn("Test name", required=True),
                "Result": st.column_config.NumberColumn("Result", step=0.01),
                "Unit": st.column_config.TextColumn("Unit"),
                "Reference Range": st.column_config.TextColumn("Reference range"),
                "Flag": st.column_config.SelectboxColumn(
                    "Flag", options=["", "H", "L", "H*", "L*", "Abnormal"]
                ),
            }
        )

        if st.button("🚀 4. Process & Save to TiDB", type="primary", use_container_width=True):
            with st.spinner("Processing + Saving..."):
                # Build RAG
                lines = ["Test | Result | Unit | Reference Range | Flag"]
                for _, row in edited_df.iterrows():
                    row_str = " | ".join(str(val) for val in row if pd.notna(val))
                    lines.append(row_str)

                full_text = "\n".join(lines)
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                chunks = splitter.split_text(full_text)
                docs = [Document(page_content=ch) for ch in chunks]

                vectorstore = FAISS.from_documents(docs, embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

                prompt = ChatPromptTemplate.from_template("""
You are a helpful lab report assistant.
Use ONLY the provided report data.
Never diagnose diseases. Only report values, flags, ranges.
Context: {context}
Question: {input}
Answer (include units & flags when relevant):""")

                llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0.1,
                    api_key=st.session_state.groq_api_key
                )

                qa_chain = create_stuff_documents_chain(llm, prompt)
                st.session_state.rag_chain = create_retrieval_chain(retriever, qa_chain)

                # Save to TiDB
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    inserted = 0

                    for _, row in edited_df.iterrows():
                        cursor.execute("""
                            INSERT INTO blood_reports
                            (timestamp, test_name, result, unit, ref_range, flag)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (
                            timestamp,
                            row.get("Test", ""),
                            float(row.get("Result") or 0),
                            row.get("Unit", ""),
                            row.get("Reference Range", ""),
                            row.get("Flag", "")
                        ))
                        inserted += 1

                    conn.commit()
                    conn.close()
                    st.success(f"AI ready! Saved {inserted} tests to TiDB.")
                except Exception as e:
                    st.error(f"Database save failed: {str(e)}")

    # ── Chat ────────────────────────────────────────────────────────
    if st.session_state.rag_chain:
        st.markdown("---")
        st.subheader("5. Ask about your report")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if query := st.chat_input("Ask anything (e.g. 'Is my cholesterol high?')"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_chain.invoke({"input": query})
                    answer = response["answer"].strip()
                    st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

with tab2:
    st.markdown("""
    ### How to use
    1. Paste blood report text (CSV style)
    2. Parse → edit values
    3. Process → AI ready + saved to TiDB
    4. Ask questions
    5. Download conversation

    Data is saved to table `blood_reports`
    """)


# ── Download Q&A ────────────────────────────────────────────────
if st.session_state.messages:
    st.markdown("---")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    md_content = "# Blood Report Q&A\n"
    md_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            md_content += f"**You:**\n{msg['content']}\n\n"
        else:
            md_content += f"**Assistant:**\n{msg['content']}\n\n"
            md_content += "---\n\n"

    # Download button
    st.download_button(
        label="📥 Download this Q&A conversation",
        data=md_content,
        file_name=f"blood_report_qa_{timestamp}.md",
        mime="text/markdown",
        help="Saves all questions and answers in nicely formatted markdown",
        use_container_width=False
    )

# ── Recommendation interface ───────────────────────────────────────────
if st.session_state.rag_chain is not None:
    st.divider()
    st.subheader("General Recommendations (not medical advice)")

    if st.button("Get Recommendations for Abnormal Values", type="primary", use_container_width=True):
        with st.spinner("Generating general suggestions..."):
            # Safety check: make sure API key exists
            if "groq_api_key" not in st.session_state or not st.session_state.groq_api_key:
                st.error("Groq API key is missing or invalid. Please set it again in the sidebar.")
                st.stop()

            # Use the same retriever to get context (abnormal values)
            abnormal_context = st.session_state.rag_chain.invoke({"input": "any abnormal report"})["answer"].strip()

            # New prompt for recommendations
            rec_prompt_template = """You are a general health information assistant.
Based on the abnormal lab values below, provide ONLY very general suggestions for recovery.
For each abnormal value:
- Suggest common lifestyle, diet changes (e.g. exercise, low sugar diet)
- Mention general medicine classes if relevant (e.g. "doctors may consider statins for high cholesterol")
- ALWAYS say: "This is not medical advice. Consult a qualified doctor for personalized treatment and medicines."
- NEVER prescribe specific medicines or dosages.
- NEVER diagnose diseases.

Abnormal values from report:
{abnormal_context}

Answer in bullet points, be concise and cautious."""

            rec_prompt = ChatPromptTemplate.from_template(rec_prompt_template)

            # Use same LLM — with safety
            rec_llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=800,
                api_key=st.session_state.groq_api_key
            )

            # Simple chain for recommendations
            rec_chain = rec_prompt | rec_llm

            try:
                rec_response = rec_chain.invoke({"abnormal_context": abnormal_context})
                rec_answer = rec_response.content.strip()
                st.markdown(rec_answer)
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")

    st.caption("These are general ideas only. Always see a doctor for real advice.")








