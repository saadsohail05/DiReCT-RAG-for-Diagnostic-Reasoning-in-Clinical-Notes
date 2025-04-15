# Import all necessary libraries
import streamlit as st
import os
import shutil
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3


# Load environment variables
load_dotenv()

# Required environment variables
REQUIRED_ENV_VARS = {
    'NVIDIA_API_KEY': 'NVIDIA AI API key for the LLM',
    'GOOGLE_API_KEY': 'Google API key for embeddings',
    'HUGGINGFACE_TOKEN': 'Hugging Face token for model access'
}

# Check for missing environment variables
missing_vars = []
for var, description in REQUIRED_ENV_VARS.items():
    if not os.getenv(var):
        missing_vars.append(f"{var} ({description})")

# If running in Streamlit Cloud or other deployment environment
if missing_vars:
    st.error("Missing required environment variables. Please set the following in your deployment environment:")
    for var in missing_vars:
        st.code(var)
    st.info("These variables should be set in your deployment platform's environment configuration, not in a .env file.")
    st.stop()

# RAG prompt template
template = """You are an expert medical AI assistant specialized in clinical decision support. Use the following retrieved medical documents to answer the question.

Here are two examples of well-structured responses:

Example 1:
CLINICAL QUERY: What are the early warning signs of acute coronary syndrome?

Key Findings:
- Chest pain/discomfort (primary symptom)
- Associated symptoms: shortness of breath, nausea, sweating
- Risk factors: age >50, smoking history, hypertension

Clinical Interpretation:
- Classic presentation suggests acute coronary pathology
- High-risk features include radiation to arm/jaw, diaphoresis
- Moderate to high probability based on symptom cluster

Chain of Thought:
1. Identified cardinal symptoms matching ACS profile
2. Evaluated risk factors strengthening diagnosis
3. Considered alternative diagnoses (PE, aortic dissection)
4. Risk stratification based on presentation severity

Recommendations:
- Immediate ECG and cardiac biomarkers
- Vital signs monitoring
- Consider aspirin 325mg if no contraindications
- Emergency department evaluation

Example 2:
CLINICAL QUERY: How do you differentiate between COPD exacerbation and heart failure?

Key Findings:
- Overlapping symptoms: dyspnea, fatigue
- Distinguishing features: timing, triggers, associated symptoms
- Past medical history crucial

Clinical Interpretation:
- COPD: productive cough, wheezing, respiratory infection trigger
- Heart failure: orthopnea, peripheral edema, cardiac history
- Severity assessment based on vital signs and mental status

Chain of Thought:
1. Analyzed symptom pattern and progression
2. Considered temporal relationship with triggers
3. Evaluated response to previous treatments
4. Integrated risk factors and comorbidities

Recommendations:
- Physical examination focusing on cardiac and respiratory systems
- Chest X-ray for congestion vs hyperinflation
- BNP testing
- Oxygen saturation monitoring

Now, please address the following query using the provided context:

CONTEXT DOCUMENTS:
{context}

CLINICAL QUERY: {question}

Provide a structured analysis using the following format:

Key Findings:
- List the most relevant clinical information
- Include primary symptoms, risk factors, and key data points

Clinical Interpretation:
- Analyze the medical significance of the findings
- Discuss potential diagnoses and their likelihood
- Evaluate severity and urgency

Chain of Thought:
- Document your clinical reasoning process step by step
- Show how you arrived at your conclusions
- Highlight key decision points
- Consider alternative diagnoses and rule-outs

Recommendations:
- Provide specific, evidence-based suggestions
- Include diagnostic tests if relevant
- Outline management approaches
- Specify urgency of interventions

References:
- Cite specific sections from the provided context
- Include relevant guidelines or studies mentioned

If you cannot provide a complete answer based on the available context, explicitly state the limitations.

Response:"""

# document formatting with clinical section emphasis
def format_docs(docs):
    formatted_sections = []
    for i, doc in enumerate(docs, 1):
        # Extract key clinical sections
        sections = doc.page_content.split("===")
        formatted = f"\nSOURCE {i}:\n"
        formatted += f"Category: {doc.metadata['category']}\n"
        formatted += f"Document Type: {doc.metadata['document_type']}\n"
        formatted += "-" * 40 + "\n"
        formatted += "\n".join(section.strip() for section in sections if section.strip())
        formatted_sections.append(formatted)
    return "\n\n".join(formatted_sections)

def process_medical_query(query: str, retriever, rag_chain, min_sources: int = 5, confidence_threshold: float = 0.7) -> dict:
    try:
        # Get retrieved documents and their similarity scores (distances)
        retrieved_docs_with_scores = retriever.vectorstore.similarity_search_with_score(query, k=min_sources)
        
        # Check source adequacy
        if len(retrieved_docs_with_scores) < min_sources:
            return {
                "answer": "Insufficient clinical evidence found. Please refine the query.",
                "sources": [],
                "confidence": 0.0,
                "warnings": ["Insufficient source documents found"]
            }

        # Separate documents and distances (lower = more similar)
        retrieved_docs = [doc for doc, distance in retrieved_docs_with_scores]
        distances = [distance for doc, distance in retrieved_docs_with_scores]

        # Enhanced confidence scoring with dynamic thresholding
        similarity_scores = [1 / (1 + d) for d in distances]
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        std_similarity = (sum((s - avg_similarity) ** 2 for s in similarity_scores) / len(similarity_scores)) ** 0.5
        
        # Adjust confidence threshold based on query complexity
        adjusted_threshold = confidence_threshold - (0.1 if len(query.split()) > 10 else 0)
        
        # Calculate final confidence score
        confidence = avg_similarity * (1 - std_similarity)  # Penalize high variance
        
        # Generate response
        response = rag_chain.invoke(query)

        # Enhanced warning system
        warnings = []
        if confidence < adjusted_threshold:
            warnings.append("Low confidence response - please verify with healthcare provider")
        if std_similarity > 0.3:  # High variance in relevance
            warnings.append("Mixed relevance in sources - interpretation may be limited")

        return {
            "answer": response,
            "sources": [doc.metadata.get("source", "unknown") for doc in retrieved_docs],
            "confidence": round(confidence, 2),
            "reasoning_confidence": round(avg_similarity, 2),
            "source_consistency": round(1 - std_similarity, 2),
            "warnings": warnings
        }

    except Exception as e:
        return {
            "answer": f"Error processing clinical query: {str(e)}",
            "sources": [],
            "confidence": 0.0,
            "warnings": ["Processing error occurred"]
        }

def process_response_sections(response_text: str) -> dict:
    """Process the response text into structured sections with better parsing."""
    sections = {
        "Key Findings": [],
        "Clinical Interpretation": [],
        "Chain of Thought": [],
        "Recommendations": [],
        "References": []
    }
    
    current_section = None
    bullet_pattern = r'[-•*]\s*(.+)'
    
    for line in response_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check if line is a section header
        for section in sections.keys():
            if section.lower() in line.lower():
                current_section = section
                break
                
        # Add content to current section
        if current_section and not any(section.lower() in line.lower() for section in sections.keys()):
            # Clean up bullet points for consistency
            import re
            bullet_match = re.match(bullet_pattern, line)
            if bullet_match:
                line = f"• {bullet_match.group(1)}"
            elif line[0].isdigit() and '. ' in line:
                # Format numbered points
                number, text = line.split('. ', 1)
                line = f"{number}. {text}"
            sections[current_section].append(line)
    
    return sections

# Initialize Streamlit app
st.set_page_config(
    page_title="Clinical Query Assistant",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        height: 150px;
        background-color: #2f3337;
        border: 1px solid #404548;
        color: #e9ecef;
    }
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div.stButton > button:first-child {
        background-color: #344955;
        color: white;
        padding: 0.5rem 2rem;
        font-size: 16px;
        border: none;
        border-radius: 4px;
    }
    div.stButton > button:hover {
        background-color: #232f34;
        color: white;
    }
    .stAlert {
        background-color: #2f3337;
        border: 1px solid #404548;
        color: #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

# Header section
st.title("Clinical Query Assistant")
st.markdown("""
    <div style='background-color: #2f3337; padding: 1.5rem; border-radius: 4px; margin-bottom: 2rem; border: 1px solid #404548;'>
        <h4 style='color: #e9ecef;'>MIMIC-IV Enhanced Clinical Decision Support</h4>
        <p style='color: #adb5bd;'>This system provides evidence-based clinical insights using advanced RAG technology and the MIMIC-IV-Ext Direct dataset.</p>
        <p style='color: #adb5bd;'><strong>Note:</strong> For research and educational purposes only.</p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Initialize RAG system
if not st.session_state.initialized:
    try:
        with st.spinner("Loading RAG system..."):
            # Initialize embeddings with error handling
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    task_type="retrieval_document"
                )
            except Exception as e:
                st.error(f"Failed to initialize embeddings: {str(e)}")
                st.info("Please verify your GOOGLE_API_KEY is set correctly in the deployment environment.")
                st.stop()
            
            db_directory = "chroma_db"
            
            # Load existing ChromaDB or create new if doesn't exist
            if not os.path.exists(db_directory):
                st.error("Database not found. The ChromaDB files must be included in your deployment.")
                st.info("Please run mainfile.py first to create the database, then include the chroma_db directory in your deployment.")
                st.stop()
            
            try:
                vectorstore = Chroma(
                    persist_directory=db_directory,
                    embedding_function=embeddings
                )
            except Exception as e:
                st.error(f"Failed to initialize vector store: {str(e)}")
                st.stop()

            # Set up retriever with same parameters as mainfile.py
            st.session_state.retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10,
                    "lambda_mult": 0.7,
                }
            )

            # Initialize LLM with error handling
            try:
                llm = ChatNVIDIA(model="writer/palmyra-med-70b-32k")
            except Exception as e:
                st.error(f"Failed to initialize NVIDIA LLM: {str(e)}")
                st.info("Please verify your NVIDIA_API_KEY is set correctly in the deployment environment.")
                st.stop()

            prompt = ChatPromptTemplate.from_template(template)
            st.session_state.rag_chain = (
                {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            st.session_state.initialized = True
    except Exception as e:
        st.error(f"Failed to initialize the application: {str(e)}")
        st.stop()

# Main interface
st.markdown("### Clinical Query Input")
st.markdown("Enter your clinical question or scenario below:")
query = st.text_area(
    label="",
    placeholder="Example: What are the key diagnostic criteria for acute coronary syndrome?",
    height=120
)

col1, col2, col3 = st.columns([2,1,1])
with col1:
    submit_button = st.button("Analyze Query", use_container_width=True)

if submit_button and query:
    with st.spinner("Processing clinical data..."):
        result = process_medical_query(
            query,
            st.session_state.retriever,
            st.session_state.rag_chain
        )
        
        # Results section
        st.markdown("---")
        st.markdown("### Analysis Results")
        
        # Confidence metrics in a more visual format
        metrics_container = st.container()
        m1, m2, m3 = metrics_container.columns(3)
        with m1:
            st.metric(
                "Overall Confidence",
                f"{result['confidence']:.2%}",
                delta=None,
                delta_color="normal"
            )
        with m2:
            st.metric(
                "Reasoning Quality",
                f"{result.get('reasoning_confidence', 0):.2%}",
                delta=None,
                delta_color="normal"
            )
        with m3:
            st.metric(
                "Source Consistency",
                f"{result.get('source_consistency', 0):.2%}",
                delta=None,
                delta_color="normal"
            )
        
        # Warnings with better formatting
        if result["warnings"]:
            st.markdown("""
                <div style='background-color: #2f3337; padding: 1rem; border-radius: 4px; margin: 1rem 0; border: 1px solid #404548;'>
                    <h4 style='color: #e9ecef;'>Important Considerations</h4>
                    <ul style='color: #adb5bd;'>
            """ + "".join([f"<li>{warning}</li>" for warning in result["warnings"]]) + """
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        # Clinical Response
        st.markdown("#### Clinical Analysis")
        
        response_container = st.container()
        with response_container:
            response_text = result["answer"]
            processed_sections = process_response_sections(response_text)
            
            for section_title, content_lines in processed_sections.items():
                if content_lines:
                    content_html = "<br>".join(content_lines)
                    st.markdown(f"""
                        <div style='background-color: #2f3337; padding: 1.5rem; border-radius: 4px; margin-bottom: 1.5rem; border: 1px solid #404548;'>
                            <h5 style='color: #e9ecef; border-bottom: 1px solid #404548; padding-bottom: 0.75rem; margin-bottom: 1rem;'>{section_title}</h5>
                            <div style='color: #adb5bd; padding: 0.5rem 0; line-height: 1.6;'>
                                {content_html}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        # Sources in expandable section
        with st.expander("Reference Sources"):
            st.markdown("The following clinical sources were consulted:")
            for idx, source in enumerate(result["sources"], 1):
                st.markdown(f"**Source {idx}:** {source}")

# Footer with disclaimer
st.markdown("---")
st.markdown("""
    <div style='background-color: #2f3337; padding: 1.5rem; border-radius: 4px; margin-top: 2rem; border: 1px solid #404548;'>
        <h4 style='color: #e9ecef;'>Medical Disclaimer</h4>
        <p style='color: #adb5bd;'>This AI-powered system is designed for research and educational purposes only. It should not be used as a substitute for:</p>
        <ul style='color: #adb5bd;'>
            <li>Professional medical advice</li>
            <li>Clinical diagnosis</li>
            <li>Treatment decisions</li>
        </ul>
        <p style='color: #adb5bd;'><strong>Always consult with qualified healthcare providers for medical concerns.</strong></p>
    </div>
""", unsafe_allow_html=True)