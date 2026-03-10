"""
Resume Chatbot - Streamlit Cloud Version
Using FAISS for vector storage (more compatible than ChromaDB)
Uses Groq API (free) for LLM inference
"""

import streamlit as st
import os
import requests
from typing import List, Dict, Tuple
import numpy as np

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ============== API KEY CONFIGURATION ==============

def get_api_key(provider: str) -> str:
    """Get API key from secrets or session state"""
    
    # Try to get from Streamlit secrets first
    if "Groq" in provider:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    else:
        if "HUGGINGFACE_API_KEY" in st.secrets:
            return st.secrets["HUGGINGFACE_API_KEY"]
    
    # Fall back to session state (user input)
    return st.session_state.get("api_key", "")

# Page config
st.set_page_config(
    page_title="Vijai Venkatesan - AI Resume Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== RESUME CONTENT ==============
RESUME_CONTENT = """
VIJAI VENKATESAN
Contact Information:
- Address: NO: 108, Nehru Street, V.P.Singh Nagar, Shanmugapuram, Pondicherry-605009
- Phone: +91 8825947952
- Email: vijaibt1@gmail.com
- LinkedIn: linkedin.com/in/vijai-v-2b89841a3

PROFESSIONAL SUMMARY
Results-driven AI/ML Engineer with nearly 7 years of experience in designing and deploying scalable AI solutions, including Generative AI and Large Language Models. Expertise in Python, machine learning, natural language processing, and intelligent document processing. Proficient in building AI pipelines and deploying models via Django REST APIs on Azure and GCP VMs. Certified in Data Science and AI/ML, committed to continuous learning, innovation, and delivering business-driven AI solutions.

WORK EXPERIENCE

Position: Associate Consultant - AI/ML (Full-time)
Company: Datamatics (TruAI Division)
Location: Pondicherry
Duration: April 2022 - Present

Projects Worked On:
- Named Entity Recognition for ADB
- Image Classification for UHG and Star Health
- Receipt Extraction from Images
- Photo Matching
- Resume Data Extraction
- Trepp Field Extraction from URLs
- Web Scraper for Fiercepharma
- TruAI GPT R&D
- Trepp Newsfeed
- Resume AI
- Azure OpenAI GPT-4 Integration for XPO Project

Key Production Projects:
- Ingram Micro Invoice Automation
- BelleTire TruAI Automation

Key Responsibilities and Impact:
- Leading end-to-end production ownership of Ingram Micro and BelleTire TruAI automation systems, ensuring reliability, scalability, and business continuity
- Achieved 90% extraction accuracy for Ingram Micro invoice processing automation
- Achieved 93.40% accuracy for BelleTire invoice processing automation
- Significantly improved data quality and reduced manual effort
- Optimized system performance to achieve an average processing latency of 10-11 seconds per page (end-to-end) for both Ingram and Belle Tire production workflows
- Driving full lifecycle delivery of AI/ML, NLP, Generative AI, and LLM-based solutions from data ingestion and preprocessing to deployment and monitoring
- Independently managing multiple live production pipelines, ensuring SLA adherence and consistent high-performance outcomes
- Designing and deploying AI services via Django REST APIs for seamless integration with enterprise applications
- Deploying and maintaining solutions on Azure Virtual Machines and GCP VMs, enabling scalable and stable production environments
- Conducting model evaluation, validation, prompt engineering, and optimization to enhance accuracy, efficiency, and adaptability
- Implementing model monitoring and continuous improvement workflows to maintain production reliability
- Collaborating with cross-functional teams and stakeholders to translate business requirements into AI-driven automation solutions
- Leading R&D initiatives to evaluate emerging AI/LLM capabilities for future integration

Tech Stack Used: Python, Django REST APIs, Gemini 2.5 Flash, Prompt Engineering, Azure & GCP Virtual Machines, Python Scheduler Automation, Data Validation & Post-Processing Layer, Logging & Monitoring, Retry Mechanism & Error Handling

Position: Data Science Intern (Internship)
Company: Innodatatics
Location: Hyderabad
Duration: October 2021 - April 2022

Project 1: Recommendation Engine for Career Transition
- Developed and deployed a machine learning-based recommendation engine
- Used collaborative and content-based filtering to assist career transitions
- Achieved 85% accuracy
- Technologies Used: Python, Scikit-Learn, TensorFlow, Streamlit, Pandas, NumPy, Git

Project 2: Named Entity Recognition on Medical Journals
- Optimized entity extraction processes
- Reduced time for data analysis in medical reports
- Technologies: Python, SQL, NLP (med7, Spacy), Tika, Pdfminer, PyPDF2, sklearn, Pandas, NumPy, Streamlit

Position: Associate (Medical Summarizer) - Full-time
Company: Aosta Software Technologies India Pvt Ltd
Location: Chennai
Duration: June 2021 - August 2021
- Summarized medical records for legal experts using AI/ML and NLP techniques, ensuring clarity and relevance
- Leveraged Cognitive Science principles to improve the accuracy and contextual relevance of legal summaries
- Applied Python and SQL to organize and streamline medical records, enabling efficient review and analysis processes

Position: Medical Record Analyst - Full-time
Company: Rapid Care Transcription Pvt Ltd
Location: Pondicherry
Duration: April 2018 - May 2021
- Supported US healthcare providers and legal investigations through detailed analysis and data visualization, providing clear insights
- Enhanced quality of medical data review and reporting through effective use of SQL and visualization tools
- Analyzed medical records for accuracy and completeness, ensuring compliance with regulatory standards

Position: Project Intern (Internship)
Company: CSIR-Central Leather Research Institute
Location: Chennai
Duration: December 2016 - June 2017
- Researched antimicrobial activity of secondary metabolites from soil actinomycetes, contributing to pharmaceutical advancements

TECHNICAL SKILLS

Programming Languages: Python, R

Frameworks and Libraries: Pandas, NumPy, Scikit-learn, TensorFlow, Keras, Matplotlib, Seaborn

AI/ML Tools: Generative AI, Large Language Models (LLM), Transformer Models, BERT, NLP, Named Entity Recognition (NER), Elasticsearch, Kibana

Development Environments (IDE): Cursor, VS Code, PyCharm, Jupyter Notebook, Spyder, RStudio, Google Colab

Databases: MySQL

Data Visualization Tools: Power BI, Tableau

Version Control: GitHub

Web Technologies: Django REST API, Postman

Cloud Platforms: GCP (Google Cloud Platform), AWS (Amazon Web Services), Microsoft Azure

Remote Access and Management: RDP, Google Chrome Remote Desktop, VPN management

File Transfer and Management: WinSCP, FileZilla

Project Management Tools: Redmine

EDUCATION

Degree: B.Tech in Biotechnology
Institution: Shri Andal Alagar College of Engineering, Anna University
GPA: 72.9%
Duration: August 2013 - May 2017

HSC (Higher Secondary Certificate)
Institution: Mutharaiyar HR Sec School
Location: Pondicherry
GPA: 64.08%
Duration: June 2012 - March 2013

SSLC (Secondary School Leaving Certificate)
Institution: Indira Gandhi Govt HR Sec School
Location: Pondicherry
GPA: 79%
Duration: June 2010 - March 2011

CERTIFICATIONS

1. AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents
   Instructor: Ligency, Ed Donner
   Platform: Udemy

2. AI Engineer Agentic Track: The Complete Agent & MCP Course
   Instructor: Ligency, Ed Donner
   Platform: Udemy

3. MCP Masterclass: Complete Guide to MCP in Python [2025]
   Platform: Udemy

4. Data Science Certification Programme
   Providers: Panasonic CareerEx & 360DigiTMG

5. Data Science Using Python and R Programming
   Provider: 360DigiTMG

6. Machine Learning with Python
   Providers: IBM & 360DIGITMG

7. Introduction to Data Science, Python for Data Science, Data Analysis Using Python, Machine Learning with R, SQL and Relational Databases, Deep Learning Fundamentals
   Provider: IBM

8. Google Analytics for Beginners
   Provider: Google Analytics Academy

9. Cloud Learning: AWS DevOps, Azure DevOps, Docker
   Provider: Datamatics

AWARDS AND RECOGNITION

1. L&D Trainer Felicitation - Certificate of Appreciation
   Date: May 2025
   Organization: Datamatics

2. Spot Individual Award Winner
   Date: July 2024
   Organization: Datamatics

3. L&D Trainer Felicitation - Certificate of Appreciation
   Date: May 2024
   Organization: Datamatics

4. Spot Individual Award Winner
   Date: June 2023
   Organization: Datamatics

YEARS OF EXPERIENCE: Nearly 7 years total professional experience
CURRENT ROLE: Associate Consultant - AI/ML at Datamatics
SPECIALIZATION: AI/ML, NLP, Generative AI, LLM, Document Processing, Invoice Automation
"""

# ============== RAG FUNCTIONS ==============

@st.cache_resource(show_spinner=False)
def initialize_embeddings():
    """Initialize HuggingFace embeddings"""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


@st.cache_resource(show_spinner=False)
def create_vector_store(_embeddings):
    """Create FAISS vector store from resume content"""
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )
    
    chunks = text_splitter.create_documents([RESUME_CONTENT])
    
    # Add metadata to chunks
    section_keywords = {
        "PROFESSIONAL SUMMARY": "Summary",
        "SUMMARY": "Summary",
        "WORK EXPERIENCE": "Work Experience",
        "Position:": "Work Experience",
        "TECHNICAL SKILLS": "Skills",
        "SKILLS": "Skills",
        "EDUCATION": "Education",
        "CERTIFICATIONS": "Certifications",
        "AWARDS": "Awards",
        "Contact": "Contact Information"
    }
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = "resume"
        
        # Determine section
        content_upper = chunk.page_content.upper()
        assigned_section = "General"
        for keyword, section in section_keywords.items():
            if keyword.upper() in content_upper:
                assigned_section = section
                break
        chunk.metadata["section"] = assigned_section
    
    # Create FAISS vector store
    vector_store = FAISS.from_documents(chunks, _embeddings)
    
    return vector_store


def retrieve_context(vector_store, query: str, k: int = 4) -> Tuple[str, List[Dict]]:
    """Retrieve relevant context from vector store"""
    docs = vector_store.similarity_search(query, k=k)
    
    context_parts = []
    sources = []
    
    for doc in docs:
        context_parts.append(doc.page_content)
        sources.append({
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "section": doc.metadata.get("section", "General")
        })
    
    context = "\n\n".join(context_parts)
    return context, sources


# ============== LLM FUNCTIONS ==============

def query_groq(prompt: str, api_key: str) -> str:
    """Query Groq API (free tier available)"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant representing Vijai Venkatesan, an AI/ML Engineer. Answer questions about his professional background based on the provided context. Be professional, accurate, and helpful."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            error_msg = response.json().get("error", {}).get("message", "Unknown error")
            return f"API Error: {error_msg}"
            
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"


def query_huggingface(prompt: str, api_key: str) -> str:
    """Query HuggingFace Inference API"""
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "return_full_text": False,
            "do_sample": True
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No response generated")
            return str(result)
        elif response.status_code == 503:
            return "Model is loading, please try again in a few seconds..."
        else:
            return f"API Error: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"


def generate_response(question: str, context: str, api_key: str, provider: str) -> str:
    """Generate response using selected LLM provider"""
    
    prompt = f"""Based on the following resume information about Vijai Venkatesan, answer the question.
Be specific, professional, and use details from the context when available.
If the information is not in the context, say so politely.

RESUME CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    if provider == "Groq (Recommended - Faster)":
        return query_groq(prompt, api_key)
    else:
        return query_huggingface(prompt, api_key)


# ============== UI STYLING ==============

st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1E88E5, #7C4DFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subheader */
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Info card in sidebar */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .info-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
    }
    
    .info-card p {
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    
    .info-card a {
        color: #ffffff;
        text-decoration: underline;
    }
    
    /* Source card */
    .source-card {
        background-color: #f0f7ff;
        border-left: 3px solid #1E88E5;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
        font-size: 0.85rem;
    }
    
    .source-section {
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 0.3rem;
    }
    
    /* Stats container */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .stat-box {
        background-color: #f5f5f5;
        padding: 0.75rem;
        border-radius: 0.5rem;
        text-align: center;
        flex: 1;
        margin: 0 0.25rem;
    }
    
    .stat-number {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E88E5;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #666;
    }
    
    /* Custom button styling */
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        font-size: 0.85rem;
        padding: 0.5rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============== INITIALIZE SESSION STATE ==============

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = ""


# ============== SIDEBAR ==============

with st.sidebar:
    # Profile Card
    st.markdown("""
    <div class="info-card">
        <h3>👤 Vijai Venkatesan</h3>
        <p><strong>Associate Consultant - AI/ML</strong></p>
        <p>🏢 Datamatics (TruAI Division)</p>
        <p>📍 Pondicherry, India</p>
        <p>📧 vijaibt1@gmail.com</p>
        <p>🔗 <a href="https://linkedin.com/in/vijai-v-2b89841a3" target="_blank">LinkedIn Profile</a></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Experience", "~7 Years")
    with col2:
        st.metric("Projects", "12+")
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Certifications", "9+")
    with col4:
        st.metric("Awards", "4")
    
    st.markdown("---")
    
    # API Configuration
    st.markdown("### 🔑 API Configuration")
    
    provider = st.selectbox(
        "Select LLM Provider",
        ["Groq (Recommended - Faster)", "HuggingFace (Backup)"],
        help="Groq is faster and more reliable. Both are free!"
    )
    
    # Check if API key is in secrets
    if "Groq" in provider:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    else:
        secret_key = st.secrets.get("HUGGINGFACE_API_KEY", "")
    
    if secret_key:
        # API key is pre-configured in secrets
        api_key = secret_key
        st.success("✅ API Key pre-configured")
    else:
        # Let user input API key
        if "Groq" in provider:
            api_key = st.text_input(
                "Groq API Key",
                type="password",
                value=st.session_state.get("api_key", ""),
                help="Get your free API key from console.groq.com"
            )
            st.markdown("🔗 [Get Free Groq API Key](https://console.groq.com/keys)")
        else:
            api_key = st.text_input(
                "HuggingFace API Token",
                type="password",
                value=st.session_state.get("api_key", ""),
                help="Get your free token from huggingface.co"
            )
            st.markdown("🔗 [Get Free HF Token](https://huggingface.co/settings/tokens)")
        
        if api_key:
            st.session_state.api_key = api_key
            st.success("✅ API Key configured")
    
    st.markdown("---")
    
    # Suggested Questions
    st.markdown("### 💡 Try These Questions")
    
    suggestions = [
        "What is Vijai's total experience?",
        "What are his main technical skills?",
        "Tell me about his current role",
        "What projects has he worked on?",
        "What accuracy did he achieve in his projects?",
        "What certifications does he have?",
        "What is his educational background?",
        "What awards has he received?"
    ]
    
    for i, suggestion in enumerate(suggestions):
        if st.button(f"📌 {suggestion}", key=f"sug_{i}", use_container_width=True):
            st.session_state.pending_question = suggestion
    
    st.markdown("---")
    
    # Clear Chat Button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Footer
    st.markdown("""
    ---
    <div style="text-align: center; font-size: 0.75rem; color: #888;">
        Built with RAG + FAISS + Streamlit<br>
        🚀 Powered by Groq/HuggingFace
    </div>
    """, unsafe_allow_html=True)


# ============== MAIN CONTENT ==============

# Header
st.markdown('<h1 class="main-header">🤖 AI Resume Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask me anything about Vijai Venkatesan\'s professional background, skills, and experience</p>', unsafe_allow_html=True)

# Initialize RAG components
with st.spinner("🔄 Initializing AI assistant..."):
    embeddings = initialize_embeddings()
    vector_store = create_vector_store(embeddings)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📚 View Sources Used"):
                for src in message["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-section">📁 {src.get('section', 'General')}</div>
                        <div>{src.get('content', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

# Handle pending question from sidebar
if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question
    
    if not api_key:
        st.error("⚠️ Please enter your API key in the sidebar to continue.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Retrieve context
        context, sources = retrieve_context(vector_store, question)
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            response = generate_response(question, context, api_key, provider)
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })
        
        st.rerun()

# Chat input
if prompt := st.chat_input("Ask about Vijai's experience, skills, projects, or anything from his resume..."):
    if not api_key:
        st.error("⚠️ Please enter your API key in the sidebar to start chatting.")
    else:
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Retrieve context
        context, sources = retrieve_context(vector_store, prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = generate_response(prompt, context, api_key, provider)
            
            st.markdown(response)
            
            # Show sources
            if sources:
                with st.expander("📚 View Sources Used"):
                    for src in sources:
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-section">📁 {src.get('section', 'General')}</div>
                            <div>{src.get('content', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Add to message history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })

# Welcome message if no chat history
if not st.session_state.messages:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
    ">
        <h3>👋 Welcome!</h3>
        <p>I'm an AI assistant that can answer questions about <strong>Vijai Venkatesan's</strong> professional background.</p>
        <p>Try asking about:</p>
        <ul style="text-align: left; display: inline-block;">
            <li>🎯 Work experience and current role</li>
            <li>💻 Technical skills and expertise</li>
            <li>🏆 Projects and achievements</li>
            <li>📜 Certifications and education</li>
            <li>🥇 Awards and recognition</li>
        </ul>
        <p><em>Click a suggested question in the sidebar or type your own below!</em></p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem;">
    💼 <a href="https://linkedin.com/in/vijai-v-2b89841a3" target="_blank">Connect on LinkedIn</a> | 
    📧 <a href="mailto:vijaibt1@gmail.com">vijaibt1@gmail.com</a> | 
    Built with ❤️ using RAG, FAISS & Streamlit
</div>
""", unsafe_allow_html=True)
