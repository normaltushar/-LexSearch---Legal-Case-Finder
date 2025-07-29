import os
import re
import sys
import streamlit as st
# Conditional import for PyMuPDF with fallback
try:
    import fitz  # PyMuPDF
except ImportError:
    try:
        import pymupdf as fitz  
    except ImportError:
        fitz = None
from io import BytesIO
try:
    from pdfminer.high_level import extract_text
except ImportError:
    extract_text = None
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from fpdf import FPDF
from datetime import datetime
import sqlite3
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Initialize session state
if 'bookmarks' not in st.session_state:
    st.session_state.bookmarks = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Database setup
def init_db():
    conn = sqlite3.connect('lexsearch.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS bookmarks
                 (id INTEGER PRIMARY KEY, case_name TEXT, citation TEXT, summary TEXT)''')
    conn.commit()
    conn.close()

# Initialize database
init_db()

# PDF text extraction with improved error handling
def extract_pdf_text(file):
    # Create an in-memory bytes buffer
    file_bytes = BytesIO(file.read())
    file_bytes.seek(0)
    
    # First try PyMuPDF if available
    if fitz is not None:
        try:
            with fitz.open(stream=file_bytes.read(), filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
        except Exception as e:
            st.error(f"PyMuPDF error: {str(e)}")
            file_bytes.seek(0)
    
    # Fallback to pdfminer if available
    if extract_text is not None:
        try:
            return extract_text(file_bytes)
        except Exception as e:
            st.error(f"PDFMiner error: {str(e)}")
    
    # If both fail, show error and return empty string
    st.error("No PDF extraction method available. Please install PyMuPDF or pdfminer.six")
    return ""

# Web scraper for IndianKanoon
def scrape_indiankanoon(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        case_title = soup.find('title').text.strip()
        case_content = soup.find('div', {'class': 'judgments'}).text.strip()
        return f"Case: {case_title}\n\n{case_content}"
    except Exception as e:
        st.error(f"Error scraping URL: {e}")
        return ""

# Vector database setup
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    if not os.path.exists("chroma_db"):
        # Create with empty docs
        return Chroma.from_documents([Document(page_content="")], embeddings, persist_directory="chroma_db")
    return Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# LLM setup with proper error handling
def get_llm():
    if os.getenv("GROQ_API_KEY"):
        return ChatGroq(temperature=0.1, model_name="llama3-70b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
    elif os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(temperature=0.1, model="gpt-4-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))
    else:
        st.error("No API key found! Please set GROQ_API_KEY or OPENAI_API_KEY in .env file")
        st.stop()

# Case summarization prompt
SUMMARY_PROMPT = """
You are a legal expert specializing in Indian law. Generate a structured summary for the following case:

Case Name: {case_name}
Court: {court}
Date: {date}
Citation: {citation}

Full Text:
{text}

Provide output in this EXACT format:

*Facts*:
- [Concise bullet points of key facts]

*Legal Issues*:
- [List of legal questions addressed]

*Arguments*:
- [Key arguments from both sides]

*Judgment*:
- [Court's final decision]
- [Verdict and reasoning]

*Key Precedents Cited*:
- [List of important cases referenced]

*Significance*:
- [Impact on legal interpretation]
"""

# RAG setup
def setup_qa_chain():
    vectorstore = get_vector_store()
    llm = get_llm()
    
    prompt = PromptTemplate(
        template="You are a legal research assistant. Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}",
        input_variables=["context", "question"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt}
    )

# Generate PDF summary
def generate_pdf(summary, case_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=f"Case Summary: {case_name}\n\n{summary}")
    return pdf.output(dest='S').encode('latin1')

# Add document to vector store
def add_to_vector_store(text, metadata=None):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = [Document(page_content=x, metadata=metadata or {}) for x in text_splitter.split_text(text)]
    
    vectorstore = get_vector_store()
    vectorstore.add_documents(docs)
    vectorstore.persist()

# UI Components
def main():
    st.set_page_config(
        page_title="LexSearch - Legal Case Finder", 
        page_icon="⚖", 
        layout="wide"
    )
    
        # Dark theme custom CSS
    st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            color: #FAFAFA;
        }
        .main, .block-container {
            background-color: #0e1117;
            color: #FAFAFA;
        }
        .css-1v3fvcr, .css-1d391kg, .css-1544g2n {
            background-color: #262730 !important;
        }
        .stTextInput > div > input,
        .stTextArea > div > textarea,
        .stSelectbox > div {
            background-color: #1E1E1E !important;
            color: #FAFAFA !important;
        }
        .stSlider > div {
            background-color: #333333;
        }
        .stDownloadButton button,
        .stButton button {
            background-color: #1f77b4;
            color: white;
            border: none;
            border-radius: 8px;
        }
        .stDownloadButton button:hover,
        .stButton button:hover {
            background-color: #145374;
            color: white;
        }
        .case-card {
            background-color: #1e1e1e;
            color: #FAFAFA;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }
        .section-title {
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 5px;
        }
        .sidebar .sidebar-content {
            background-color: #161b22;
        }
    </style>
    """, unsafe_allow_html=True)

    
    st.title("⚖ LexSearch - Legal Case Finder")
    st.caption("AI-powered legal research tool for Indian case laws")
    
    # Show dependency warnings
    if fitz is None or extract_text is None:
        warning_message = ""
        if fitz is None:
            warning_message += (
                "⚠️ <b>PyMuPDF not installed!</b> "
                "PDF processing will be limited. "
                "Install with: <code>pip install PyMuPDF</code><br>"
            )
        if extract_text is None:
            warning_message += (
                "⚠️ <b>pdfminer.six not installed!</b> "
                "PDF extraction will not work. "
                "Install with: <code>pip install pdfminer.six</code><br>"
            )
        st.sidebar.markdown(
            f"<div class='error'>{warning_message}</div>",
            unsafe_allow_html=True
        )
    
    try:
        qa_chain = setup_qa_chain()
    except Exception as e:
        st.error(f"Error setting up QA chain: {e}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🔍 Filters")
        year_range = st.slider("Year Range", 1950, datetime.now().year, (1950, datetime.now().year))
        courts = st.multiselect("Courts", ["Supreme Court", "High Courts", "Tribunals"])
        sections = st.multiselect("Sections", ["Article 21", "Article 14", "Section 302", "Section 420"])
        
        st.divider()
        
        st.header("📁 Add Cases")
        url = st.text_input("IndianKanoon URL")
        if url and st.button("Add from URL"):
            case_text = scrape_indiankanoon(url)
            if case_text:
                # Extract metadata
                case_name = re.search(r"Case: (.*?)\n", case_text)
                case_name = case_name.group(1) if case_name else "Unknown Case"
                # Add to vector store
                add_to_vector_store(case_text, metadata={"source": url, "name": case_name})
                st.success(f"Added: {case_name}")
        
        uploaded_file = st.file_uploader("Upload Judgment PDF", type="pdf")
        if uploaded_file and uploaded_file not in st.session_state.uploaded_files:
            if fitz is None and extract_text is None:
                st.error("Cannot process PDF - no extraction libraries installed")
            else:
                with st.spinner("Processing PDF..."):
                    text = extract_pdf_text(uploaded_file)
                    if text:
                        add_to_vector_store(text, metadata={"source": uploaded_file.name})
                        st.session_state.uploaded_files.append(uploaded_file)
                        st.success("PDF added to database!")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Search Cases", "📚 Bookmarks", "⚙ Settings"])
    
    with tab1:
        query = st.text_input("Ask a legal question:", placeholder="e.g. 'Cases about Article 21 and privacy'")
        
        if query:
            with st.spinner("Searching case database..."):
                try:
                    result = qa_chain.run(query)
                except Exception as e:
                    st.error(f"Error during search: {e}")
                    result = None
            
            if result:
                # Display results
                st.subheader("📄 Relevant Cases")
                with st.expander("View Summary", expanded=True):
                    st.markdown(result)
                
                # Mock case results (in real app, these would come from the vector store)
                cases = [
                    {"name": "Justice K.S. Puttaswamy vs Union of India", 
                     "citation": "AIR 2017 SC 4161", 
                     "court": "Supreme Court",
                     "date": "2017",
                     "summary": "Landmark case establishing right to privacy as fundamental right under Article 21"},
                    
                    {"name": "R. Rajagopal vs State of Tamil Nadu", 
                     "citation": "1994 SCC (6) 632", 
                     "court": "Supreme Court",
                     "date": "1994",
                     "summary": "Right to privacy balanced against freedom of press"},
                ]
                
                for case in cases:
                    with st.container():
                        st.markdown(f"<div class='case-card'>", unsafe_allow_html=True)
                        st.subheader(case['name'])
                        st.caption(f"{case['citation']} | {case['court']} | {case['date']}")
                        st.write(case['summary'])
                        
                        col1, col2 = st.columns([1, 9])
                        with col1:
                            if st.button("🔖 Bookmark", key=f"bookmark_{case['name']}"):
                                conn = sqlite3.connect('lexsearch.db')
                                c = conn.cursor()
                                c.execute("INSERT INTO bookmarks (case_name, citation, summary) VALUES (?, ?, ?)",
                                          (case['name'], case['citation'], case['summary']))
                                conn.commit()
                                conn.close()
                                st.success("Bookmarked!")
                        with col2:
                            if st.button("📄 Generate Summary PDF", key=f"pdf_{case['name']}"):
                                pdf_data = generate_pdf(case['summary'], case['name'])
                                st.download_button(
                                    label="Download PDF",
                                    data=pdf_data,
                                    file_name=f"{case['name']}_summary.pdf",
                                    mime="application/pdf"
                                )
                        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.subheader("📚 Bookmarked Cases")
        conn = sqlite3.connect('lexsearch.db')
        c = conn.cursor()
        c.execute("SELECT * FROM bookmarks")
        bookmarks = c.fetchall()
        conn.close()
        
        if not bookmarks:
            st.info("No bookmarked cases yet")
        
        for idx, bm in enumerate(bookmarks):
            with st.container():
                st.markdown(f"<div class='case-card'>", unsafe_allow_html=True)
                st.subheader(bm[1])
                st.caption(f"Citation: {bm[2]}")
                st.write(bm[3])
                
                col1, col2 = st.columns([1, 9])
                with col1:
                    if st.button("❌ Remove", key=f"remove_{idx}"):
                        conn = sqlite3.connect('lexsearch.db')
                        c = conn.cursor()
                        c.execute("DELETE FROM bookmarks WHERE id=?", (bm[0],))
                        conn.commit()
                        conn.close()
                        st.experimental_rerun()
                with col2:
                    pdf_data = generate_pdf(bm[3], bm[1])
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name=f"{bm[1]}_summary.pdf",
                        mime="application/pdf",
                        key=f"dl_{idx}"
                    )
                st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Configuration")
        st.info("Set environment variables in .env file")
        
        # Vector DB info
        db_count = "Not created"
        if os.path.exists("chroma_db"):
            try:
                db_count = f"{len(os.listdir('chroma_db'))} documents"
            except:
                db_count = "Exists"
        st.write(f"Vector Database: ChromaDB ({db_count})")
        
        st.write(f"Embedding Model: BAAI/bge-small-en")
        
        # LLM info
        try:
            llm = get_llm()
            model_name = llm.model_name if hasattr(llm, 'model_name') else "Unknown Model"
            st.write(f"LLM: {model_name}")
        except:
            st.write("LLM: Not configured")
        
        # Dependency status
        st.subheader("Dependencies")
        st.write(f"PyMuPDF: {'Installed' if fitz is not None else 'Not installed'}")
        st.write(f"PDFMiner: {'Installed' if extract_text is not None else 'Not installed'}")

if __name__ == "__main__":
    main()
