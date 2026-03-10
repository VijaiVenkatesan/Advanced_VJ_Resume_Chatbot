"""
Resume Chatbot - Streamlit Cloud Version (v3 - Simplified)
No FAISS, No LangChain - Pure Python + sklearn
Uses Groq API (free) or HuggingFace for LLM inference
"""

import streamlit as st
import requests
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

# sklearn for similarity
from sklearn.metrics.pairwise import cosine_similarity

# sentence-transformers for embeddings
from sentence_transformers import SentenceTransformer

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

# =====================================================
# PAGE CONFIG - MUST BE FIRST STREAMLIT COMMAND
# =====================================================
st.set_page_config(
    page_title="Vijai Venkatesan - AI Resume Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# RESUME CONTENT
# =====================================================
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


# =====================================================
# SIMPLE RAG IMPLEMENTATION (No external dependencies)
# =====================================================

@dataclass
class Chunk:
    """A chunk of text with metadata"""
    text: str
    section: str
    index: int


class SimpleRAG:
    """Simple RAG implementation using sklearn"""
    
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray = None
        self.model: SentenceTransformer = None
    
    def load_model(self):
        """Load the embedding model"""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def split_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        sentences = text.replace('\n\n', '\n').split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append('\n'.join(current_chunk))
                
                # Keep some overlap
                overlap_text = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_text.copy()
                current_length = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def identify_section(self, text: str) -> str:
        """Identify which section a chunk belongs to"""
        text_upper = text.upper()
        
        if any(kw in text_upper for kw in ['PROFESSIONAL SUMMARY', 'SUMMARY', 'RESULTS-DRIVEN']):
            return "Summary"
        elif any(kw in text_upper for kw in ['WORK EXPERIENCE', 'POSITION:', 'COMPANY:', 'DURATION:']):
            return "Work Experience"
        elif any(kw in text_upper for kw in ['TECHNICAL SKILLS', 'PROGRAMMING', 'FRAMEWORKS']):
            return "Skills"
        elif any(kw in text_upper for kw in ['EDUCATION', 'DEGREE:', 'INSTITUTION:']):
            return "Education"
        elif any(kw in text_upper for kw in ['CERTIFICATION', 'INSTRUCTOR:', 'PLATFORM:']):
            return "Certifications"
        elif any(kw in text_upper for kw in ['AWARD', 'RECOGNITION', 'WINNER']):
            return "Awards"
        elif any(kw in text_upper for kw in ['CONTACT', 'EMAIL:', 'PHONE:', 'LINKEDIN']):
            return "Contact"
        else:
            return "General"
    
    def index_document(self, text: str):
        """Index a document by splitting and embedding"""
        # Split into chunks
        raw_chunks = self.split_text(text)
        
        # Create Chunk objects with metadata
        self.chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            section = self.identify_section(chunk_text)
            self.chunks.append(Chunk(text=chunk_text, section=section, index=i))
        
        # Create embeddings
        chunk_texts = [c.text for c in self.chunks]
        self.embeddings = self.model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=False)
    
    def search(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        """Search for relevant chunks"""
        # Embed query
        query_embedding = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return chunks with scores
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))
        
        return results


@st.cache_resource(show_spinner=True)
def initialize_rag() -> SimpleRAG:
    """Initialize and cache the RAG system"""
    rag = SimpleRAG()
    rag.load_model()
    rag.index_document(RESUME_CONTENT)
    return rag


def get_context(rag: SimpleRAG, query: str, top_k: int = 4) -> Tuple[str, List[Dict]]:
    """Get relevant context for a query"""
    results = rag.search(query, top_k=top_k)
    
    context_parts = []
    sources = []
    
    for chunk, score in results:
        context_parts.append(chunk.text)
        sources.append({
            "content": chunk.text[:250] + "..." if len(chunk.text) > 250 else chunk.text,
            "section": chunk.section,
            "relevance": f"{score:.2%}"
        })
    
    context = "\n\n---\n\n".join(context_parts)
    return context, sources


# =====================================================
# LLM API FUNCTIONS
# =====================================================

def call_groq_api(prompt: str, api_key: str) -> str:
    """Call Groq API for LLM inference"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": """You are a professional AI assistant representing Vijai Venkatesan, an experienced AI/ML Engineer. 
Your job is to answer questions about his professional background, skills, experience, and achievements.
Be helpful, professional, and accurate. Use specific details from the provided context.
If information is not available in the context, politely say so."""
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
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        elif response.status_code == 401:
            return "❌ **Invalid API Key.** Please check your Groq API key and try again."
        elif response.status_code == 429:
            return "⚠️ **Rate limit exceeded.** Please wait a moment and try again."
        else:
            error_detail = response.json().get("error", {}).get("message", "Unknown error")
            return f"❌ **API Error ({response.status_code}):** {error_detail}"
    
    except requests.exceptions.Timeout:
        return "⏱️ **Request timed out.** Please try again."
    except requests.exceptions.ConnectionError:
        return "🔌 **Connection error.** Please check your internet connection."
    except Exception as e:
        return f"❌ **Error:** {str(e)}"


def call_huggingface_api(prompt: str, api_key: str) -> str:
    """Call HuggingFace Inference API"""
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    formatted_prompt = f"""<s>[INST] You are a helpful AI assistant representing Vijai Venkatesan, an AI/ML Engineer.
Answer the question based on the provided context. Be professional and accurate.

{prompt} [/INST]"""
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No response generated.")
            return str(result)
        elif response.status_code == 503:
            return "⏳ **Model is loading.** Please wait 20-30 seconds and try again."
        elif response.status_code == 401:
            return "❌ **Invalid API Token.** Please check your HuggingFace token."
        else:
            return f"❌ **API Error ({response.status_code})**"
    
    except requests.exceptions.Timeout:
        return "⏱️ **Request timed out.** Please try again."
    except Exception as e:
        return f"❌ **Error:** {str(e)}"


def generate_answer(question: str, context: str, api_key: str, provider: str) -> str:
    """Generate answer using selected LLM provider"""
    
    prompt = f"""Based on the following resume information about Vijai Venkatesan, please answer the question.
Use specific details from the context. Be professional and helpful.

CONTEXT FROM RESUME:
{context}

QUESTION: {question}

Please provide a clear, informative answer:"""
    
    if "Groq" in provider:
        return call_groq_api(prompt, api_key)
    else:
        return call_huggingface_api(prompt, api_key)


# =====================================================
# CUSTOM CSS STYLING
# =====================================================

st.markdown("""
<style>
    /* Header styles */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5 0%, #7C4DFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
        margin-bottom: 0;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Profile card */
    .profile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.25rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    
    .profile-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.2rem;
    }
    
    .profile-card p {
        margin: 0.25rem 0;
        font-size: 0.85rem;
        opacity: 0.95;
    }
    
    .profile-card a {
        color: white !important;
        text-decoration: underline;
    }
    
    /* Source card */
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #1E88E5;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
        font-size: 0.85rem;
    }
    
    .source-header {
        color: #1E88E5;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }
    
    .source-content {
        color: #444;
        line-height: 1.4;
    }
    
    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf0 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Button hover effects */
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = ""


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    # Profile Card
    st.markdown("""
    <div class="profile-card">
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
    c1, c2 = st.columns(2)
    c1.metric("Experience", "~7 Years")
    c2.metric("Projects", "12+")
    
    c3, c4 = st.columns(2)
    c3.metric("Certs", "9+")
    c4.metric("Awards", "4")
    
    st.divider()
    
    # API Configuration
    st.markdown("### 🔑 API Configuration")
    
    provider = st.selectbox(
        "LLM Provider",
        ["Groq (Recommended)", "HuggingFace"],
        help="Groq is faster. Both are FREE!"
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
    
    st.divider()
    
    # Suggested Questions
    st.markdown("### 💡 Quick Questions")
    
    questions = [
        "What is Vijai's experience?",
        "What are his main skills?",
        "Tell me about his current role",
        "What projects has he done?",
        "What accuracy did he achieve?",
        "What certifications does he have?",
        "What is his education?",
        "What awards has he received?"
    ]
    
    for i, q in enumerate(questions):
        if st.button(f"📌 {q}", key=f"q_{i}", use_container_width=True):
            st.session_state.pending_question = q
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("""
    <div style="text-align:center; font-size:0.75rem; color:#888; margin-top:1rem;">
        Built with ❤️<br>RAG + Streamlit
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# MAIN CONTENT
# =====================================================

# Header
st.markdown('<h1 class="main-header">🤖 AI Resume Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask me anything about Vijai Venkatesan\'s professional background</p>', unsafe_allow_html=True)

# Initialize RAG
with st.spinner("🔄 Loading AI..."):
    rag = initialize_rag()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 View Sources"):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">📁 {src['section']} ({src['relevance']})</div>
                        <div class="source-content">{src['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)

# Handle sidebar question click
if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question
    
    if not api_key:
        st.error("⚠️ Please enter your API key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        
        context, sources = get_context(rag, question)
        
        with st.spinner("🤔 Thinking..."):
            answer = generate_answer(question, context, api_key, provider)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
        st.rerun()

# Chat input
if prompt := st.chat_input("Ask about experience, skills, projects..."):
    if not api_key:
        st.error("⚠️ Please enter your API key in the sidebar first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get context and generate response
        context, sources = get_context(rag, prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                answer = generate_answer(prompt, context, api_key, provider)
            
            st.markdown(answer)
            
            if sources:
                with st.expander("📚 View Sources"):
                    for src in sources:
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-header">📁 {src['section']} ({src['relevance']})</div>
                            <div class="source-content">{src['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

# Welcome message when no chat
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <h3>👋 Welcome!</h3>
        <p>I'm an AI assistant that answers questions about <strong>Vijai Venkatesan's</strong> professional background.</p>
        <br>
        <p><strong>Ask about:</strong></p>
        <p>🎯 Experience &nbsp;|&nbsp; 💻 Skills &nbsp;|&nbsp; 🏆 Projects &nbsp;|&nbsp; 📜 Certifications &nbsp;|&nbsp; 🥇 Awards</p>
        <br>
        <p><em>👈 Enter your API key in the sidebar to start!</em></p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem;">
    💼 <a href="https://linkedin.com/in/vijai-v-2b89841a3" target="_blank">LinkedIn</a> &nbsp;|&nbsp;
    📧 <a href="mailto:vijaibt1@gmail.com">Email</a> &nbsp;|&nbsp;
    Built with RAG + Streamlit
</div>
""", unsafe_allow_html=True)

