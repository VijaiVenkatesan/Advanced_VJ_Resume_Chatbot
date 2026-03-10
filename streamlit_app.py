"""
Resume Chatbot - Streamlit Cloud Version
Uses HuggingFace Inference API (free) instead of Ollama
"""

import streamlit as st
import os
from typing import List, Dict
import requests

# For RAG
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Page config
st.set_page_config(
    page_title="Vijai Venkatesan - AI Resume Assistant",
    page_icon="🤖",
    layout="wide"
)

# Resume content (embedded)
RESUME_CONTENT = """
VIJAI VENKATESAN
Contact: NO: 108, Nehru Street, V.P.Singh Nagar, Shanmugapuram, Pondicherry-605009
Phone: +91 8825947952
Email: vijaibt1@gmail.com
LinkedIn: linkedin.com/in/vijai-v-2b89841a3

SUMMARY
Results-driven AI/ML Engineer with nearly 7 years of experience in designing and deploying scalable AI solutions, including Generative AI and Large Language Models. Expertise in Python, machine learning, natural language processing, and intelligent document processing. Proficient in building AI pipelines and deploying models via Django REST APIs on Azure and GCP VMs. Certified in Data Science and AI/ML, committed to continuous learning, innovation, and delivering business-driven AI solutions.

WORK EXPERIENCE

Associate Consultant - AI/ML | Full-time
Datamatics (TruAI Division) | Pondicherry
04/2022 - Present

Projects: Named Entity Recognition for ADB, Image Classification for UHG and Star Health, Receipt Extraction from Images, Photo Matching, Resume Data Extraction, Trepp Field Extraction from URLs, Web Scraper for Fiercepharma, TruAI GPT R&D, Trepp Newsfeed, Resume AI, Azure OpenAI GPT-4 Integration for XPO Project.

Key Production Projects: Ingram Micro Invoice and BelleTire TruAI Automation.

Key Responsibilities & Impact:
- Leading end-to-end production ownership of Ingram Micro and BelleTire TruAI automation systems
- Achieved 90% extraction accuracy for Ingram Micro and 93.40% accuracy for BelleTire invoice processing
- Optimized system performance to achieve 10-11 seconds per page processing latency
- Driving full lifecycle delivery of AI/ML, NLP, Generative AI, and LLM-based solutions
- Designing and deploying AI services via Django REST APIs
- Deploying solutions on Azure Virtual Machines and GCP VMs

Tech Stack: Python, Django REST APIs, Gemini 2.5 Flash, Prompt Engineering, Azure & GCP Virtual Machines

Data Science Intern | Innodatatics | Hyderabad | 10/2021 - 04/2022
- Developed recommendation engine for career transition (85% accuracy)
- Named Entity Recognition on Medical Journals
- Technologies: Python, Scikit-Learn, TensorFlow, Streamlit, NLP

Associate (Medical Summarizer) | Aosta Software Technologies | Chennai | 06/2021 - 08/2021
- Summarized medical records using AI/ML and NLP techniques

Medical Record Analyst | Rapid Care Transcription | Pondicherry | 04/2018 - 05/2021
- Supported US healthcare providers through detailed analysis

Project Intern | CSIR-CLRI | Chennai | 12/2016 - 06/2017
- Researched antimicrobial activity of secondary metabolites

SKILLS
Programming Languages: Python, R
Frameworks & Libraries: Pandas, NumPy, Scikit-learn, TensorFlow, Keras, Matplotlib, Seaborn
AI/ML Tools: Generative AI, LLM, Transformer Models, BERT, NLP, NER, Elasticsearch
IDE: Cursor, VS Code, PyCharm, Jupyter Notebook, Google Colab
Cloud Platforms: GCP, AWS, Microsoft Azure
Web Technologies: Django REST API, Postman

EDUCATION
B.Tech in Biotechnology | Anna University | GPA: 72.9% | 2013-2017

CERTIFICATIONS
- AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents (Udemy)
- AI Engineer Agentic Track: Complete Agent & MCP Course (Udemy)
- MCP Masterclass: Complete Guide to MCP in Python (Udemy)
- Data Science Certification (Panasonic CareerEx & 360DigiTMG)
- Machine Learning with Python (IBM)

AWARDS
- L&D Trainer Felicitation (05/2025, 05/2024) - Datamatics
- Spot Individual Award Winner (07/2024, 06/2023) - Datamatics
"""


@st.cache_resource
def initialize_rag():
    """Initialize RAG components"""
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    chunks = text_splitter.create_documents([RESUME_CONTENT])
    
    # Add metadata
    sections = ["SUMMARY", "WORK EXPERIENCE", "SKILLS", "EDUCATION", "CERTIFICATIONS", "AWARDS"]
    for chunk in chunks:
        content_upper = chunk.page_content.upper()
        for section in sections:
            if section in content_upper:
                chunk.metadata["section"] = section
                break
        else:
            chunk.metadata["section"] = "General"
    
    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="resume"
    )
    
    return vectorstore


def query_huggingface(prompt: str, api_key: str) -> str:
    """Query HuggingFace Inference API"""
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "No response generated")
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def query_groq(prompt: str, api_key: str) -> str:
    """Query Groq API (free tier)"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"


def generate_response(question: str, context: str, api_key: str, provider: str) -> str:
    """Generate response using selected provider"""
    
    prompt = f"""You are an AI assistant representing Vijai Venkatesan, an AI/ML Engineer with nearly 7 years of experience.
Answer the following question based on the resume context provided. Be professional, helpful, and specific.

Resume Context:
{context}

Question: {question}

Provide a concise and helpful response:"""
    
    if provider == "Groq (Recommended)":
        return query_groq(prompt, api_key)
    else:
        return query_huggingface(prompt, api_key)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
    }
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize RAG
vectorstore = initialize_rag()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.markdown("### 👤 About Me")
    st.markdown("""
    <div class="info-card">
        <h3>Vijai Venkatesan</h3>
        <p><strong>Associate Consultant - AI/ML</strong></p>
        <p>📧 vijaibt1@gmail.com</p>
        <p>🔗 <a href="https://linkedin.com/in/vijai-v-2b89841a3" style="color: white;">LinkedIn</a></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔑 API Configuration")
    
    provider = st.selectbox(
        "Select Provider",
        ["Groq (Recommended)", "HuggingFace"],
        help="Groq offers faster responses with free tier"
    )
    
    if provider == "Groq (Recommended)":
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Get free API key at console.groq.com"
        )
        st.markdown("[Get Free Groq API Key](https://console.groq.com)")
    else:
        api_key = st.text_input(
            "HuggingFace API Key",
            type="password",
            help="Get free API key at huggingface.co"
        )
        st.markdown("[Get Free HuggingFace Token](https://huggingface.co/settings/tokens)")
    
    st.markdown("---")
    
    st.markdown("### 💡 Suggested Questions")
    suggestions = [
        "What is Vijai's total experience?",
        "What are his key skills?",
        "Tell me about his current role",
        "What projects has he worked on?",
        "What certifications does he have?"
    ]
    
    for suggestion in suggestions:
        if st.button(suggestion, use_container_width=True):
            st.session_state.pending_question = suggestion

# Main content
st.markdown('<h1 class="main-header">🤖 AI Resume Assistant</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Ask me anything about Vijai's professional background</p>", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle pending question
if hasattr(st.session_state, 'pending_question'):
    question = st.session_state.pending_question
    del st.session_state.pending_question
    
    if not api_key:
        st.error("Please enter your API key in the sidebar")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Retrieve relevant context
        docs = vectorstore.similarity_search(question, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        with st.spinner("Thinking..."):
            response = generate_response(question, context, api_key, provider)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# Chat input
if prompt := st.chat_input("Ask about Vijai's experience, skills, or projects..."):
    if not api_key:
        st.error("Please enter your API key in the sidebar")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Retrieve relevant context
        docs = vectorstore.similarity_search(prompt, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt, context, api_key, provider)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #666; font-size: 0.8rem;'>
    Built with RAG + LangChain + Streamlit | 
    <a href="https://linkedin.com/in/vijai-v-2b89841a3">Connect on LinkedIn</a>
</p>
""", unsafe_allow_html=True)