# 💪 FitRAG - AI Fitness Coach

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3.27-orange.svg)
![GPT-4](https://img.shields.io/badge/GPT--4-OpenAI-green.svg)

**Production-ready RAG (Retrieval-Augmented Generation) system for evidence-based fitness and nutrition coaching.**

FitRAG combines the power of GPT-4 with semantic search over fitness literature to provide accurate, sourced answers to training and nutrition questions. Unlike generic chatbots, FitRAG cites its sources and limits responses to evidence-based information from your knowledge base.

---

## 🎯 What is FitRAG?

FitRAG is an AI fitness assistant that uses **Retrieval-Augmented Generation (RAG)** to answer questions based on your curated fitness documents. Instead of relying on potentially outdated training data, RAG retrieves relevant information from your knowledge base in real-time, ensuring:

- ✅ **Evidence-based answers** from trusted sources
- ✅ **Source citations** for every claim
- ✅ **Up-to-date information** (add new research anytime)
- ✅ **No hallucination** (answers grounded in your documents)
- ✅ **Domain-specific expertise** (focused on fitness/nutrition)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUESTION                            │
│            "What's optimal protein for muscle building?"        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT INTERFACE                          │
│                        (app.py)                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │  EMBEDDING (Voyage AI)            │
         │  Question → Vector [0.12, -0.45,…]│
         └───────────────┬───────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │  CHROMADB VECTOR SEARCH           │
         │  Cosine Similarity → Top 4 Chunks │
         └───────────────┬───────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 RETRIEVAL QA CHAIN                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  PROMPT TEMPLATE                                         │  │
│  │  Context: [Retrieved Chunks]                             │  │
│  │  Question: [User Question]                               │  │
│  │  Instructions: Cite sources, be evidence-based           │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │  OPENAI GPT-4                     │
         │  Temperature=0 (Consistent)       │
         │  Generate Answer + Citations      │
         └───────────────┬───────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESPONSE TO USER                             │
│  Answer: "Research shows 1.6g/kg/day…"                         │
│  Sources: [fitness_basics.txt (4 chunks)]                      │
└─────────────────────────────────────────────────────────────────┘

                    DOCUMENT INGESTION PIPELINE
                          (ingest.py)

┌─────────────┐      ┌──────────────┐      ┌────────────────┐
│ PDFs & TXT  │──────│  TEXT        │──────│  CHUNK WITH    │
│ Files       │      │  SPLITTER    │      │  OVERLAP       │
│ (./data/)   │      │  (Recursive) │      │  1000±200 chars│
└─────────────┘      └──────────────┘      └────────┬───────┘
                                                     │
                                                     ▼
                                     ┌───────────────────────────┐
                                     │  VOYAGE AI EMBEDDINGS     │
                                     │  voyage-2 model           │
                                     │  Text → Vectors           │
                                     └────────────┬──────────────┘
                                                  │
                                                  ▼
                                     ┌───────────────────────────┐
                                     │  CHROMADB STORAGE         │
                                     │  Persist to ./chroma_db/  │
                                     │  HNSW Index for Fast ANN  │
                                     └───────────────────────────┘
```

---

## 🛠️ Tech Stack & Justification

| Component | Technology | Why This Choice? |
|-----------|-----------|------------------|
| **LLM** | OpenAI GPT-4 | Industry-leading quality, excellent instruction following, reliable API, strong at citation and reasoning |
| **Embeddings** | Voyage AI (voyage-2) | State-of-the-art quality, optimized for RAG, free tier available, 1024 dimensions |
| **Vector DB** | ChromaDB | Local-first, persistent, production-ready, HNSW indexing, no server required |
| **Framework** | LangChain | Industry-standard RAG abstractions, excellent OpenAI integration, active development |
| **UI** | Streamlit | Rapid prototyping, built-in caching, clean interface, perfect for ML apps |
| **Doc Loading** | PyPDF | Reliable PDF parsing, metadata extraction, widely used |

**Alternative Embedding Options:**
- **Cohere** (`embed-english-v3.0`): Free tier, excellent quality
- **HuggingFace** (`all-mpnet-base-v2`): Fully local, no API costs, privacy-first

---

## 📦 Project Structure

```
fitrag/
├── app.py                  # Streamlit RAG interface
├── ingest.py               # Document ingestion pipeline
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── .gitignore              # Git ignore patterns
├── README.md               # This file
├── TECHNICAL_REPORT.md     # Deep-dive technical documentation
├── data/                   # Knowledge base documents
│   └── fitness_basics.txt  # Sample fitness content
└── chroma_db/              # Vector database (created after ingestion)
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.9 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Voyage AI API key ([Get one here](https://www.voyageai.com/)) *or* Cohere API key

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/fitrag.git
cd fitrag
```

**2. Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Configure API keys:**
```bash
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=sk-proj-...
# VOYAGE_API_KEY=pa-...
```

**5. Ingest documents:**
```bash
python ingest.py
```

Expected output:
```
============================================================
🏋️  FitRAG Document Ingestion Pipeline
============================================================

📋 Step 1: Loading documents...
✓ Loaded 1 text file(s)

✂️  Step 2: Chunking documents...
✓ Created 16 chunks from 1 document(s)

🔢 Step 3: Initializing embeddings...
✓ Using voyage-2 embedding model

💾 Step 4: Creating vector store...
✓ Vector store created and persisted to ./chroma_db

============================================================
✅ Ingestion Complete!
============================================================
```

**6. Launch the application:**
```bash
streamlit run app.py
```

The interface will open at `http://localhost:8501`

---

## 💡 Usage Guide

### Example Queries to Try

1. **"What's the optimal protein intake for muscle building?"**
   - Tests retrieval of research-backed nutrition guidelines
   - Should cite Morton et al. 2018 meta-analysis

2. **"Explain progressive overload and how to implement it"**
   - Tests understanding of training principles
   - Should cover multiple progression methods

3. **"How does sleep affect muscle recovery?"**
   - Tests scientific explanation capability
   - Should cite hormonal and neurological processes

4. **"What's better for fat loss: cardio or strength training?"**
   - Tests nuanced comparison
   - Should discuss both benefits and interference effects

5. **"When should I eat protein for maximum muscle synthesis?"**
   - Tests nutrition timing knowledge
   - Should reference leucine thresholds and distribution

### Interpreting Results

Every answer includes:
- **Main response**: GPT-4's evidence-based answer
- **Source documents**: Expandable section showing:
  - Exact text snippets used
  - Source filename and page (if applicable)
  - Up to 4 most relevant chunks

**Pro Tip:** If an answer seems incomplete, check the sources - you may need to add more detailed documents to your knowledge base.

---

## 📚 How It Works (RAG Pipeline Explained)

### 1. **Document Ingestion** (Run Once or When Adding Docs)
```python
# Load PDFs and text files
documents = load_documents("./data")

# Split into chunks with overlap
chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # ~200-250 words
    chunk_overlap=200     # Prevents context loss
).split_documents(documents)

# Convert to vectors
embeddings = VoyageEmbeddings(model="voyage-2")
vectors = [embeddings.embed(chunk.text) for chunk in chunks]

# Store in ChromaDB
vectorstore = Chroma.from_documents(chunks, embeddings)
```

**Why chunking?**  
LLMs have context limits. Chunks let us store large documents but retrieve only relevant portions. Overlap ensures important context isn't lost at boundaries.

### 2. **Query Processing** (Every User Question)
```python
# User asks: "How much protein should I eat?"
question_vector = embeddings.embed(question)

# Find similar chunks
similar_chunks = vectorstore.similarity_search(
    question_vector, 
    k=4  # Top 4 most relevant
)

# Build prompt with context
prompt = f"""
Context: {similar_chunks}
Question: {question}
Provide evidence-based answer with citations.
"""

# Generate answer
answer = gpt_4(prompt, temperature=0)
```

**Why semantic search?**  
Unlike keyword matching, semantic search understands meaning. "muscle building" matches "hypertrophy" and "muscle protein synthesis."

### 3. **Answer Generation**
GPT-4 receives:
- **Context**: 4 most relevant chunks from your documents
- **Question**: The user's query
- **Instructions**: Custom prompt enforcing citations and accuracy

Temperature=0 ensures consistent, deterministic answers.

---

## 🔑 API Keys Setup

### OpenAI (Required)
1. Visit [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create account and generate new API key
3. Copy to `.env` as `OPENAI_API_KEY=sk-proj-...`
4. Pricing: ~$0.03/1K input tokens, ~$0.06/1K output tokens for GPT-4

### Voyage AI (Recommended for Embeddings)
1. Visit [voyageai.com](https://www.voyageai.com/)
2. Sign up and get API key
3. Add to `.env` as `VOYAGE_API_KEY`
4. Free tier: 100M tokens/month (very generous)

### Alternative: Cohere Embeddings
1. Visit [dashboard.cohere.com](https://dashboard.cohere.com/)
2. Create API key
3. Add to `.env` as `COHERE_API_KEY`
4. Uncomment Cohere lines in `ingest.py` and `app.py`
5. Free tier available

### Alternative: HuggingFace (Local, No API)
1. Uncomment HuggingFace lines in `ingest.py` and `app.py`
2. No API key needed - runs locally
3. First run downloads model (~500MB)
4. Slower embedding speed, but zero cost

---

## 🐛 Troubleshooting

### "ChromaDB not found" Error
**Problem:** Running `app.py` before `ingest.py`  
**Solution:** Run `python ingest.py` to create the vector database first

### "API Key Not Found" Error
**Problem:** Environment variables not loaded  
**Solution:** Ensure `.env` file exists in project root with correct keys

### Slow Embedding Speed
**Problem:** Large documents taking too long  
**Solution:** 
- Use Voyage AI (fastest) instead of HuggingFace
- Reduce document size
- Process incrementally

### Poor Answer Quality
**Problem:** Answers not citing sources or hallucinating  
**Solution:**
- Ensure documents cover the topic sufficiently
- Check if question is too far from document content
- Increase `k` value in retrieval (currently 4)
- Add more specific documents

### Import Errors
**Problem:** Module not found  
**Solution:** 
```bash
pip install -r requirements.txt --force-reinstall
```

### Streamlit Port Already in Use
**Problem:** Port 8501 occupied  
**Solution:**
```bash
streamlit run app.py --server.port 8502
```

---

## 📊 Adding Your Own Documents

1. **Add files to `./data/` directory:**
   - Supported formats: `.pdf`, `.txt`
   - Organize in subdirectories if desired

2. **Re-run ingestion:**
   ```bash
   python ingest.py
   ```
   This will replace the existing vector database.

3. **Best practices:**
   - Use high-quality, evidence-based sources
   - Include research citations in your documents
   - Organize by topic (nutrition/, training/, recovery/)
   - Name files descriptively

**Recommended sources:**
- Scientific review papers
- Evidence-based fitness blogs (Stronger by Science, Renaissance Periodization)
- Textbooks (NSCA, ACSM)
- Guidelines from professional organizations

---

## 🎨 Customization Options

### Adjust Chunk Size
Edit `ingest.py`:
```python
CHUNK_SIZE = 1500      # Larger chunks, more context
CHUNK_OVERLAP = 300    # More overlap, less context loss
```

### Change LLM Model
Edit `app.py`:
```python
LLM_MODEL = "gpt-4-turbo-preview"  # Faster, cheaper
# or
LLM_MODEL = "gpt-3.5-turbo"        # Much cheaper, good quality
```

### Modify Retrieval Count
Edit `app.py`:
```python
RETRIEVAL_K = 6  #  Retrieve more chunks for complex questions
```

### Custom Prompt Template
Edit `CUSTOM_PROMPT_TEMPLATE` in `app.py` to change:
- Tone (professional vs casual)
- Response format (bulleted vs narrative)
- Citation style
- Domain focus

---

## 🔮 Future Enhancements

**Planned Features:**
- [ ] **Conversational Memory**: Remember previous questions in session
- [ ] **Multi-hop Reasoning**: Answer complex questions requiring multiple retrievals
- [ ] **Hybrid Search**: Combine semantic + keyword search for better recall
- [ ] **Re-ranking**: Use cross-encoder to reorder retrieved chunks
- [ ] **Query Expansion**: Automatically rephrase questions for better retrieval
- [ ] **Evaluation Dashboard**: Track retrieval quality and answer faithfulness
- [ ] **Multi-modal Support**: Process images (exercise form diagrams)
- [ ] **Structured Outputs**: Generate workout plans, meal plans in JSON
- [ ] **User Feedback Loop**: Thumbs up/down to improve retrieval

---

## 📄 License

MIT License - see LICENSE file for details.

**Attribution:**  
If you use FitRAG in your project, a link back to this repository is appreciated but not required.

---

## 🙏 Acknowledgments

**Built with:**
- [LangChain](https://github.com/langchain-ai/langchain) - RAG framework
- [OpenAI GPT-4](https://openai.com/) - Language model
- [Voyage AI](https://www.voyageai.com/) - Embedding model
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Streamlit](https://streamlit.io/) - UI framework

**Inspired by research:**
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "Lost in the Middle: How Language Models Use Long Contexts" (Liu et al., 2023)
- "REALM: Retrieval-Augmented Language Model Pre-Training" (Guu et al., 2020)

---

## 📞 Support & Contributing

**Questions or Issues?**
- Open an issue on GitHub
- Check `TECHNICAL_REPORT.md` for deep-dive explanations

**Contributing:**
- PRs welcome! Focus on: better chunking strategies, evaluation metrics, UI improvements
- Please follow PEP 8 and include docstrings
- Test changes with `python ingest.py` and `streamlit run app.py`

---

## 📖 Learn More

**Recommended Reading:**
- [TECHNICAL_REPORT.md](./TECHNICAL_REPORT.md) - Deep-dive into RAG architecture
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [OpenAI Cookbook](https://cookbook.openai.com/)
- [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)

---

## 🎯 Project Features for Your Portfolio

### Technical Skills Demonstrated

**ML/AI Engineering:**
- ✅ Built production RAG pipeline with LangChain + GPT-4
- ✅ Implemented semantic search with vector embeddings (Voyage AI)
- ✅ Designed chunking strategy with overlap for context preservation
- ✅ Engineered domain-specific prompts for accurate, cited responses
- ✅ Deployed ML application with Streamlit + persistent ChromaDB

**Software Engineering:**
- ✅ Clean architecture with separation of concerns (ingest vs query)
- ✅ Environment-based configuration management
- ✅ Error handling and user-friendly messages
- ✅ Caching strategies for performance (`@st.cache_resource`)
- ✅ Type hints and comprehensive docstrings

**Gen AI / LLM:**
- ✅ Prompt engineering for fitness domain
- ✅ Temperature tuning for deterministic outputs
- ✅ RAG design patterns (chunking, retrieval, generation)
- ✅ Vector similarity search implementation
- ✅ Source attribution and citation tracking

---

**⭐ Star this repo if you found it helpful! Questions? Open an issue.**
