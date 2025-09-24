import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. Page Configuration and Title ---
st.title("💬 Vokasi Assistant (RAG Version)")
st.caption("Tanyakan apapun terkait dokumen pengetahuan Anda")

# --- 2. Securely get API Key using Streamlit Secrets ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Kunci API Google AI tidak ditemukan. Harap atur di file secrets.toml.", icon="🚨")
    st.info("""
        Untuk mengatur kunci API Anda:\n
        1. Buat file bernama `.streamlit/secrets.toml` di direktori proyek Anda.\n
        2. Tambahkan baris berikut ke dalam file tersebut:\n
           `GOOGLE_API_KEY = "API_KEY_ANDA"`\n
        3. Ganti `"API_KEY_ANDA"` dengan kunci API Google AI Anda yang sebenarnya.
    """)
    st.stop()


# --- 3. Initialize LLM, Embeddings, and Vector Store (Retriever) ---

# Menggunakan cache agar tidak perlu load model setiap kali ada interaksi
@st.cache_resource
def load_components():
    """
    Fungsi ini akan memuat komponen-komponen yang dibutuhkan
    dan menyimpannya di cache Streamlit.
    """
    try:
        # Inisialisasi model LLM dari Google
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            temperature=0.3, # Temperature lebih rendah untuk jawaban lebih fokus
            convert_system_message_to_human=True # Agar bisa menerima system prompt
        )
        
        # Inisialisasi model Embeddings dari Google
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        # Memuat vector database yang sudah dibuat sebelumnya
        vector_store = Chroma(
            persist_directory="chroma_db", 
            embedding_function=embeddings
        )
        
        # Membuat retriever untuk mengambil dokumen relevan
        retriever = vector_store.as_retriever(search_kwargs={'k': 3}) # Ambil 3 dokumen paling relevan
        
        return llm, retriever
    except Exception as e:
        st.error(f"Terjadi kesalahan saat inisialisasi: {e}")
        return None, None

llm, retriever = load_components()

if not llm or not retriever:
    st.stop()


# --- 4. Create the RAG Chain ---

# Template prompt untuk memberikan instruksi kepada LLM
prompt_template = """
Anda adalah asisten AI yang membantu. Jawablah pertanyaan pengguna HANYA berdasarkan konteks yang diberikan di bawah ini.
Jika informasi tidak ditemukan dalam konteks, katakan saja "Maaf, saya tidak menemukan informasi mengenai hal tersebut di dalam dokumen."
Jangan mencoba mengarang jawaban.

Konteks:
{context}

Pertanyaan:
{question}

Jawaban yang Bermanfaat:
"""

# Membuat prompt dari template
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Membuat RAG Chain menggunakan LangChain Expression Language (LCEL)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# --- 5. Chat History and UI ---

# Sidebar untuk reset
with st.sidebar:
    st.subheader("Opsi")
    if st.button("Reset Percakapan"):
        st.session_state.messages = []
        st.rerun()

# Inisialisasi riwayat chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Menampilkan pesan-pesan lama
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Menerima input dari pengguna
if user_prompt := st.chat_input("Apa yang ingin Anda tanyakan?"):
    # Menambahkan pesan user ke riwayat dan menampilkannya
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Menampilkan pesan "Thinking..."
    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban..."):
            # Mendapatkan jawaban dari RAG chain
            response = rag_chain.invoke(user_prompt)
            st.markdown(response)
    
    # Menambahkan jawaban AI ke riwayat
    st.session_state.messages.append({"role": "assistant", "content": response})