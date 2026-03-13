"""
Resume Chatbot - Groq Edition (Latest Models)
Using TF-IDF for embeddings | Multiple Groq Models
"""

import streamlit as st
import requests
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
# GROQ MODELS (Updated - Latest Active Models)
# =====================================================
GROQ_MODELS = {
    "Llama 3.1 8B Instant": {
        "id": "llama-3.1-8b-instant",
        "description": "Very fast, 131K context",
        "icon": "⚡"
    },
    "Llama 3.3 70B Versatile": {
        "id": "llama-3.3-70b-versatile",
        "description": "Most powerful, detailed responses",
        "icon": "🚀"
    },
    "Llama 4 Scout 17B": {
        "id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "description": "Latest Llama 4, fast & capable",
        "icon": "🦙"
    },
    "Qwen 3 32B": {
        "id": "qwen/qwen3-32b",
        "description": "Alibaba's powerful model",
        "icon": "🌟"
    },
    "Kimi K2 Instruct": {
        "id": "moonshotai/kimi-k2-instruct",
        "description": "Moonshot AI model",
        "icon": "🌙"
    },
    "Compound (Groq)": {
        "id": "groq/compound",
        "description": "Groq's compound AI",
        "icon": "🔮"
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

KEY FACTS:
- Total Years of Experience: Nearly 7 years
- Current Role: Associate Consultant - AI/ML at Datamatics (TruAI Division)
- Location: Pondicherry, India
- Specialization: AI/ML, NLP, Generative AI, LLM, Document Processing, Invoice Automation
- Key Achievement: 90% accuracy for Ingram Micro, 93.40% accuracy for BelleTire
- Processing Speed: 10-11 seconds per page end-to-end
"""


# =====================================================
# LIGHTWEIGHT RAG (TF-IDF)
# =====================================================

@dataclass
class Chunk:
    text: str
    section: str
    index: int


class LightweightRAG:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.vectorizer: TfidfVectorizer = None
        self.tfidf_matrix = None
    
    def split_text(self, text: str) -> List[str]:
        sections = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        max_chunk_size = 500
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            if current_length + len(section) > max_chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(section)
            current_length += len(section)
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def identify_section(self, text: str) -> str:
        text_upper = text.upper()
        
        if any(kw in text_upper for kw in ['PROFESSIONAL SUMMARY', 'RESULTS-DRIVEN']):
            return "Summary"
        elif any(kw in text_upper for kw in ['WORK EXPERIENCE', 'POSITION:', 'COMPANY:', 'DURATION:', 'KEY RESPONSIBILITIES']):
            return "Work Experience"
        elif any(kw in text_upper for kw in ['TECHNICAL SKILLS', 'PROGRAMMING LANGUAGES', 'FRAMEWORKS']):
            return "Skills"
        elif any(kw in text_upper for kw in ['EDUCATION', 'DEGREE:', 'B.TECH', 'HSC', 'SSLC']):
            return "Education"
        elif any(kw in text_upper for kw in ['CERTIFICATION', 'UDEMY', 'IBM']):
            return "Certifications"
        elif any(kw in text_upper for kw in ['AWARD', 'RECOGNITION', 'WINNER', 'FELICITATION']):
            return "Awards"
        elif any(kw in text_upper for kw in ['CONTACT', 'EMAIL', 'PHONE', 'LINKEDIN', 'ADDRESS']):
            return "Contact"
        elif any(kw in text_upper for kw in ['KEY FACTS', 'TOTAL YEARS', 'CURRENT ROLE']):
            return "Key Facts"
        elif any(kw in text_upper for kw in ['PROJECT', 'AUTOMATION', 'INGRAM', 'BELLETIRE']):
            return "Projects"
        else:
            return "General"
    
    def index_document(self, text: str):
        raw_chunks = self.split_text(text)
        
        self.chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            if chunk_text.strip():
                section = self.identify_section(chunk_text)
                self.chunks.append(Chunk(text=chunk_text, section=section, index=i))
        
        chunk_texts = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
    
    def search(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self.chunks[idx], float(similarities[idx])))
        
        if not results:
            for idx in top_indices[:2]:
                results.append((self.chunks[idx], float(similarities[idx])))
        
        return results


@st.cache_resource
def initialize_rag() -> LightweightRAG:
    rag = LightweightRAG()
    rag.index_document(RESUME_CONTENT)
    return rag


def get_context(rag: LightweightRAG, query: str, top_k: int = 4) -> Tuple[str, List[Dict]]:
    results = rag.search(query, top_k=top_k)
    
    context_parts = []
    sources = []
    
    for chunk, score in results:
        context_parts.append(chunk.text)
        sources.append({
            "content": chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text,
            "section": chunk.section,
            "relevance": f"{score:.0%}" if score > 0 else "Related"
        })
    
    context = "\n\n---\n\n".join(context_parts)
    return context, sources


# =====================================================
# GROQ API
# =====================================================

def get_api_key() -> str:
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return ""


def call_groq_api(prompt: str, model_id: str) -> str:
    api_key = get_api_key()
    
    if not api_key:
        return """❌ **API Key Not Configured**

Please add your Groq API key to Streamlit secrets:

1. Go to app settings on [share.streamlit.io](https://share.streamlit.io)
2. Click **Settings** → **Secrets**
3. Add: `GROQ_API_KEY = "gsk_your_key_here"`
4. Save and reboot

🔗 [Get free key](https://console.groq.com/keys)"""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": """You are a professional AI assistant representing Vijai Venkatesan, an experienced AI/ML Engineer with nearly 7 years of experience at Datamatics.

Your role:
- Answer questions about his professional background, skills, and achievements
- Be helpful, professional, and friendly
- Use specific details and numbers from the context
- If information isn't available, say so politely
- Keep responses concise but informative

Key facts:
- ~7 years experience
- Current: Associate Consultant - AI/ML at Datamatics
- Achievements: 90% accuracy (Ingram Micro), 93.40% (BelleTire)
- Specializes in: AI/ML, NLP, GenAI, LLM, Document Processing"""
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
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        elif response.status_code == 401:
            return "❌ **Invalid API Key.** Please check your Groq API key."
        elif response.status_code == 429:
            return "⚠️ **Rate limit.** Please wait and try again."
        elif response.status_code == 400:
            error_msg = response.json().get("error", {}).get("message", "")
            if "decommissioned" in error_msg.lower():
                return "⚠️ **Model unavailable.** Please select a different model."
            return f"❌ **Error:** {error_msg[:200]}"
        else:
            return f"❌ **Error ({response.status_code})**"
    
    except requests.exceptions.Timeout:
        return "⏱️ **Timeout.** Please try again."
    except Exception as e:
        return f"❌ **Error:** {str(e)}"


def generate_answer(question: str, context: str, model_id: str) -> str:
    prompt = f"""Based on the following resume information about Vijai Venkatesan, answer the question.
Use specific details from the context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    return call_groq_api(prompt, model_id)


# =====================================================
# CSS
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
    
    .profile-card h3 { margin: 0 0 0.5rem 0; font-size: 1.2rem; }
    .profile-card p { margin: 0.25rem 0; font-size: 0.85rem; opacity: 0.95; }
    .profile-card a { color: white !important; text-decoration: underline; }
    
    .model-info {
        background: #e3f2fd;
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
    
    .source-header { color: #1E88E5; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.25rem; }
    .source-content { color: #444; line-height: 1.4; }
    
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
        padding: 0.4rem 0.8rem;
        border-radius: 2rem;
        font-size: 0.7rem;
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
    st.session_state.selected_model = "Llama 3.1 8B Instant"


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("""
    <div class="profile-card">
        <h3>👤 Vijai Venkatesan</h3>
        <p><strong>Associate Consultant - AI/ML</strong></p>
        <p>🏢 Datamatics (TruAI Division)</p>
        <p>📍 Pondicherry, India</p>
        <p>📧 vijaibt1@gmail.com</p>
        <p>🔗 <a href="https://linkedin.com/in/vijai-v-2b89841a3" target="_blank">LinkedIn</a></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Quick Stats")
    c1, c2 = st.columns(2)
    c1.metric("Experience", "~7 Years")
    c2.metric("Projects", "12+")
    c3, c4 = st.columns(2)
    c3.metric("Certs", "9+")
    c4.metric("Awards", "4")
    
    st.divider()
    
    st.markdown("### 🤖 Select AI Model")
    
    selected_model = st.selectbox(
        "Choose a model",
        options=list(GROQ_MODELS.keys()),
        index=list(GROQ_MODELS.keys()).index(st.session_state.selected_model),
        format_func=lambda x: f"{GROQ_MODELS[x]['icon']} {x}",
        label_visibility="collapsed"
    )
    st.session_state.selected_model = selected_model
    
    model_info = GROQ_MODELS[selected_model]
    st.markdown(f'<div class="model-info">ℹ️ {model_info["description"]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="powered-by">⚡ Powered by Groq</div>', unsafe_allow_html=True)
    
    st.divider()
    
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
    <div style="text-align:center; font-size:0.7rem; color:#888; margin-top:1rem;">
        Built with ❤️ using RAG + Groq
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# MAIN
# =====================================================

st.markdown('<h1 class="main-header">🤖 AI Resume Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask me anything about Vijai Venkatesan\'s professional background</p>', unsafe_allow_html=True)

rag = initialize_rag()
current_model_id = GROQ_MODELS[st.session_state.selected_model]["id"]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("model"):
                st.caption(f"🤖 {msg['model']}")
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-header">📁 {src['section']} ({src['relevance']})</div>
                            <div class="source-content">{src['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)

if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question
    
    st.session_state.messages.append({"role": "user", "content": question})
    context, sources = get_context(rag, question)
    
    with st.spinner(f"🤔 {st.session_state.selected_model}..."):
        answer = generate_answer(question, context, current_model_id)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "model": st.session_state.selected_model
    })
    st.rerun()

if prompt := st.chat_input("Ask about experience, skills, projects..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    context, sources = get_context(rag, prompt)
    
    with st.chat_message("assistant"):
        with st.spinner(f"🤔 {st.session_state.selected_model}..."):
            answer = generate_answer(prompt, context, current_model_id)
        
        st.markdown(answer)
        st.caption(f"🤖 {st.session_state.selected_model}")
        
        if sources:
            with st.expander("📚 Sources"):
                for src in sources:
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">📁 {src['section']} ({src['relevance']})</div>
                        <div class="source-content">{src['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "model": st.session_state.selected_model
    })

st.markdown("""
<style>
.welcome-card {
    background: linear-gradient(135deg, #0d2137 0%, #1a3a5c 100%) !important;
    border-radius: 15px !important;
    padding: 25px !important;
    margin: 20px 0 !important;
    border-left: 5px solid #4CAF50 !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
}

.welcome-card h3 {
    color: #ffffff !important;
    font-size: 1.4rem !important;
    margin-bottom: 15px !important;
    font-weight: 700 !important;
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5) !important;
}

.welcome-card p {
    color: #ffffff !important;
    font-size: 1rem !important;
    line-height: 1.7 !important;
    margin: 8px 0 !important;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.4) !important;
}

.welcome-card strong {
    color: #5dde6e !important;
    font-weight: 600 !important;
}

.welcome-card em {
    color: #ffe44d !important;
    font-style: normal !important;
}
</style>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown(f"""
    <div class="welcome-card">
        <h3>👋 Welcome!</h3>
        <p>I'm powered by <strong>{st.session_state.selected_model}</strong></p>
        <p>Ask me about <strong>Vijai Venkatesan's</strong> professional background.</p>
        <br>
        <p>🎯 Experience &nbsp;|&nbsp; 💻 Skills &nbsp;|&nbsp; 🏆 Projects &nbsp;|&nbsp; 📜 Certifications &nbsp;|&nbsp; 🥇 Awards</p>
        <br>
        <p><em>👈 Click a quick question or type below!</em></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem;">
    💼 <a href="https://linkedin.com/in/vijai-v-2b89841a3" target="_blank">LinkedIn</a> &nbsp;|&nbsp;
    📧 <a href="mailto:vijaibt1@gmail.com">Email</a>
</div>
""", unsafe_allow_html=True)
