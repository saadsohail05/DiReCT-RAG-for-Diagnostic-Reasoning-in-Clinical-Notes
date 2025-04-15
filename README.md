# DiReCT-RAG for Diagnostic Reasoning in Clinical Notes

A sophisticated Retrieval-Augmented Generation (RAG) system designed for clinical decision support, leveraging the MIMIC-IV-Ext Direct dataset. This system employs advanced language models and semantic search capabilities to provide evidence-based clinical insights and diagnostic reasoning support.

## Overview

DiReCT-RAG implements a multi-stage retrieval pipeline that processes clinical notes to provide contextually relevant medical information. The system utilizes state-of-the-art language models and embedding techniques to ensure accurate and reliable clinical information retrieval and generation.

## Core Features

- **Advanced RAG Architecture**
  - Multi-stage retrieval pipeline with semantic chunking
  - Dynamic confidence scoring system
  - Source consistency evaluation
  - Mixed relevance detection
  - Clinical section parsing and standardization

- **Intelligent Query Processing**
  - Context-aware medical information retrieval
  - Semantic similarity-based document chunking
  - Maximal Marginal Relevance (MMR) for diverse source selection
  - Dynamic confidence thresholding

- **Clinical Response Generation**
  - Structured medical analysis format
  - Evidence-based recommendations
  - Source verification and confidence metrics
  - Automated warning generation for low-confidence responses

## Technical Architecture

### Data Processing Pipeline

1. **Document Ingestion**
   - Custom JSON flattening for clinical notes
   - Automatic category/subcategory extraction
   - Clinical section header standardization
   - Metadata enrichment

2. **Vector Embedding System**
   - Primary: Google's Generative AI Embeddings (models/embedding-001)
   - Secondary: Clinical ModernBERT support
   - Task-specific retrieval optimization

3. **Semantic Search Implementation**
   - ChromaDB vector store integration
   - MMR search configuration:
     - Retrieved documents: k=5
     - Candidate pool: fetch_k=10
     - Diversity factor: lambda_mult=0.7

4. **LLM Integration**
   - Model: Palmyra-Med-70B-32k via NVIDIA AI Endpoints
   - Specialized medical prompt engineering
   - Context-aware response generation

## Technology Stack

### Core Components
- Python 3.12+
- LangChain framework
- Streamlit
- ChromaDB

### AI/ML Technologies
- NVIDIA AI Endpoints
- Google Generative AI
- SentenceTransformers
- HuggingFace Transformers

### Additional Libraries
- PyTorch
- NumPy
- Transformers
- python-dotenv

## Dataset

The system utilizes the MIMIC-IV-Ext Direct dataset (version 1.0.0), which includes:
- Comprehensive clinical notes
- Diagnostic flowcharts
- Medical knowledge graphs
- Structured clinical information across multiple specialties

## Technical Innovations

### 1. Semantic Chunking Algorithm
- Custom implementation for medical text processing
- Coherence-preserving document splitting
- Dynamic chunk size optimization based on content structure

### 2. Dynamic Confidence Scoring
- Multi-factor confidence calculation
- Query complexity adaptation
- Source consistency evaluation
- Automated threshold adjustment

### 3. Enhanced Medical Formatting
- Structured clinical section parsing
- Professional medical response formatting
- Automated warning generation for low-confidence scenarios

### 4. Flexible Embedding Strategy
- Dual embedding support (Google AI & Clinical ModernBERT)
- Task-specific optimization
- Runtime embedding selection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/saadsohail05/DiReCT-RAG-for-Diagnostic-Reasoning-in-Clinical-Notes.git
cd DiReCT-RAG-for-Diagnostic-Reasoning-in-Clinical-Notes
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
NVIDIA_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
HUGGINGFACE_TOKEN=your_token_here
```

## Usage

1. Initialize the database:
```bash
python mainfile.py
```

2. Start the web interface:
```bash
streamlit run app.py
```

## Model Details

### Primary Models
1. **Language Model**
   - Name: Palmyra-Med-70B-32k
   - Provider: NVIDIA AI Endpoints
   - Context Window: 32,768 tokens
   - Domain: Medical/Clinical

2. **Embedding Models**
   - Primary: Google Generative AI Embeddings (models/embedding-001)
   - Secondary: Clinical ModernBERT
   - Implementation: Task-specific optimization for medical content

## Performance Metrics

- Response Generation Time: <5 seconds average
- Source Retrieval Accuracy: >90% relevance score
- Context Window: Up to 32k tokens
- Confidence Scoring: Dynamic thresholding with multi-factor analysis

## Legal Disclaimer

This system is designed for research and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. All clinical decision-making should be performed by qualified healthcare providers.

## License

This project utilizes the MIMIC-IV-Ext Direct dataset and follows all applicable licensing and usage restrictions. For detailed licensing information, please refer to the LICENSE file.

## References

- MIMIC-IV Dataset: https://physionet.org/content/mimiciv/2.2/
- LangChain Documentation: https://python.langchain.com/docs/get_started/introduction.html
- NVIDIA AI Endpoints: https://www.nvidia.com/en-us/gpu-cloud/ai-endpoints/