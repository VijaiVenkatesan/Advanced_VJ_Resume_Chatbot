"""
Resume Chatbot - Groq Edition
Multiple Groq Models | RAG-powered | No API key input needed
"""

import streamlit as st
import requests
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Vijai Venkatesan - AI Resume Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# GROQ MODELS CONFIGURATION
# =====================================================
GROQ_MODELS = {
    "Llama 3.1 8B (Fast)": {
        "id": "llama-3.1-8b-instant",
        "description": "Fast responses, good for quick queries",
        "icon": "⚡"
    },
    "Llama 3.3 70B (Powerful)": {
        "id": "llama-3.3-70b-versatile",
        "description": "Most capable, detailed responses",
        "icon": "🚀"
    },
    "Llama 3.1 70B": {
        "id": "llama-3.1-70b-versatile",
        "description": "Balanced speed and quality",
        "icon": "⭐"
    },
    "Mixtral 8x7B": {
        "id": "mixtral-8x7b-32768",
        "description": "Great for complex questions",
        "icon": "🔥"
    },
    "Gemma 2 9B": {
        "id": "gemma2-9b-it",
        "description": "Google's efficient model",
        "icon": "💎"
    }
}

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
# RAG IMPLEMENTATION
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
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def split_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
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
                chunks.append('\n'.join(current_chunk))
                overlap_text = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_text.copy()
                current_length = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def identify_section(self, text: str) -> str:
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
        raw_chunks = self.split_text(text)
        
        self.chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            section = self.identify_section(chunk_text)
            self.chunks.append(Chunk(text=chunk_text, section=section, index=i))
        
        chunk_texts = [c.text for c in self.chunks]
        self.embeddings = self.model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=False)
    
    def search(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        query_embedding = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))
        
        return results


@st.cache_resource(show_spinner=True)
def initialize_rag() -> SimpleRAG:
    rag = SimpleRAG()
    rag.load_model()
    rag.index_document(RESUME_CONTENT)
    return rag


def get_context(rag: SimpleRAG, query: str, top_k: int = 4) -> Tuple[str, List[Dict]]:
    results = rag.search(query, top_k=top_k)
    
    context_parts = []
    sources = []
    
    for chunk, score in results:
        context_parts.append(chunk.text)
        sources.append({
            "content": chunk.text[:250] + "..." if len(chunk.text) > 250 else chunk.text,
            "section": chunk.section,
            "relevance": f"{score:.0%}"
        })
    
    context = "\n\n---\n\n".join(context_parts)
    return context, sources


# =====================================================
# GROQ API
# =====================================================

def get_api_key() -> str:
    """Get API key from Streamlit secrets"""
    try:
        return st.secrets["GROQ_API_KEY"]
    except KeyError:
        return ""


def call_groq_api(prompt: str, model_id: str) -> str:
    """Call Groq API with selected model"""
    
    api_key = get_api_key()
    
    if not api_key:
        return """❌ **API Key Not Configured**
        
Please add your Groq API key to Streamlit secrets:

1. Go to your app settings on Streamlit Cloud
2. Click "Secrets" tab
3. Add: `GROQ_API_KEY = "your_key_here"`
4. Save and reboot the app

Get your free key at [console.groq.com](https://console.groq.com/keys)"""
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": """You are a professional AI assistant representing Vijai Venkatesan, an experienced AI/ML Engineer with nearly 7 years of experience.

Your role is to:
- Answer questions about his professional background, skills, experience, and achievements
- Be helpful, professional, accurate, and friendly
- Use specific details from the provided context
- If information is not available, politely say so
- Highlight relevant achievements with specific numbers when available

Keep responses concise but informative."""
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
            return "❌ **Invalid API Key.** Please check your Groq API key in Streamlit secrets."
        elif response.status_code == 429:
            return "⚠️ **Rate limit exceeded.** Please wait a moment and try again."
        elif response.status_code == 503:
            return "⚠️ **Service temporarily unavailable.** Please try again in a few seconds."
        else:
            error_detail = response.json().get("error", {}).get("message", "Unknown error")
            return f"❌ **API Error ({response.status_code}):** {error_detail}"
    
    except requests.exceptions.Timeout:
        return "⏱️ **Request timed out.** Please try again."
    except requests.exceptions.ConnectionError:
        return "🔌 **Connection error.** Please check your internet connection."
    except Exception as e:
        return f"❌ **Error:** {str(e)}"


def generate_answer(question: str, context: str, model_id: str) -> str:
    """Generate answer using Groq"""
    
    prompt = f"""Based on the following resume information about Vijai Venkatesan, please answer the question.
Use specific details from the context. Be professional and helpful.

CONTEXT FROM RESUME:
{context}

QUESTION: {question}

Please provide a clear, informative answer:"""
    
    return call_groq_api(prompt, model_id)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
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
    
    .model-info {
        background: #e8f4fd;
        border-radius: 0.5rem;
        padding: 0.5rem 0.75rem;
        font-size: 0.8rem;
        color: #1565c0;
        margin-top: 0.5rem;
    }
    
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
    
    .welcome-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf0 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    .powered-by {
        background: linear-gradient(90deg, #f97316 0%, #ea580c 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Llama 3.1 8B (Fast)"


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
    
    # Model Selection
    st.markdown("### 🤖 Select AI Model")
    
    selected_model = st.selectbox(
        "Choose a model",
        options=list(GROQ_MODELS.keys()),
        index=list(GROQ_MODELS.keys()).index(st.session_state.selected_model),
        format_func=lambda x: f"{GROQ_MODELS[x]['icon']} {x}",
        help="Different models have different capabilities"
    )
    
    st.session_state.selected_model = selected_model
    
    # Show model description
    model_info = GROQ_MODELS[selected_model]
    st.markdown(f"""
    <div class="model-info">
        ℹ️ {model_info['description']}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="powered-by">⚡ Powered by Groq</div>', unsafe_allow_html=True)
    
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
    
    # Clear Chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Footer
    st.markdown("""
    <div style="text-align:center; font-size:0.75rem; color:#888; margin-top:1rem;">
        Built with ❤️<br>
        RAG + Groq + Streamlit
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

# Get current model
current_model_id = GROQ_MODELS[st.session_state.selected_model]["id"]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Show model used for assistant messages
        if msg["role"] == "assistant":
            if msg.get("model"):
                st.caption(f"🤖 Model: {msg['model']}")
            
            if msg.get("sources"):
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
    
    st.session_state.messages.append({"role": "user", "content": question})
    
    context, sources = get_context(rag, question)
    
    with st.spinner(f"🤔 Thinking with {st.session_state.selected_model}..."):
        answer = generate_answer(question, context, current_model_id)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "model": st.session_state.selected_model
    })
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask about experience, skills, projects..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get context and generate response
    context, sources = get_context(rag, prompt)
    
    with st.chat_message("assistant"):
        with st.spinner(f"🤔 Thinking with {st.session_state.selected_model}..."):
            answer = generate_answer(prompt, context, current_model_id)
        
        st.markdown(answer)
        st.caption(f"🤖 Model: {st.session_state.selected_model}")
        
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
        "sources": sources,
        "model": st.session_state.selected_model
    })

# Welcome message
if not st.session_state.messages:
    st.markdown(f"""
    <div class="welcome-card">
        <h3>👋 Welcome!</h3>
        <p>I'm an AI assistant powered by <strong>{st.session_state.selected_model}</strong></p>
        <p>Ask me anything about <strong>Vijai Venkatesan's</strong> professional background.</p>
        <br>
        <p><strong>Try asking about:</strong></p>
        <p>🎯 Experience &nbsp;|&nbsp; 💻 Skills &nbsp;|&nbsp; 🏆 Projects &nbsp;|&nbsp; 📜 Certifications &nbsp;|&nbsp; 🥇 Awards</p>
        <br>
        <p><em>👈 Select a different AI model in the sidebar, or click a quick question!</em></p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem;">
    💼 <a href="https://linkedin.com/in/vijai-v-2b89841a3" target="_blank">LinkedIn</a> &nbsp;|&nbsp;
    📧 <a href="mailto:vijaibt1@gmail.com">Email</a> &nbsp;|&nbsp;
    Built with RAG + Groq + Streamlit
</div>
""", unsafe_allow_html=True)
