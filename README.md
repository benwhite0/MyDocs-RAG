# MyDocs-RAG Assistant

A retrieval-augmented-generation chatbot service for any document collection.

## Features

- Ingest PDFs & text files (chunk & embed)  
- Store & query in FAISS  
- FastAPI endpoint & optional Streamlit UI  
- Automated evaluation (DeepEval/RAGAS)

## Installation

```bash
git clone https://github.com/benwhite0/MyDocs-RAG.git
cd MyDocs-RAG
pip install -r requirements.txt
```

## Usage

1. Place your documents in `data/raw/`  
2. Build the index:  
   ```bash
   python -m src.myrag.index
   ```  
3. Run the API:  
   ```bash
   uvicorn src.myrag.api:app --reload
   ```  
4. (Optional) Start the UI:  
   ```bash
   streamlit run app/ui.py
   ```

## Repository Structure

```
MyDocs-RAG/
├── data/               # Raw & processed documents
├── notebooks/          # Exploratory analysis
├── src/
│   └── myrag/          # Core package (ingest, index, rag, api, eval)
├── app/                # Optional Streamlit UI
├── tests/              # Unit & integration tests
├── Dockerfile
├── requirements.txt
└── README.md
```


## Contributing

1. Fork the repo and create a feature branch (`git checkout -b feature/your-feature`).
2. Run `pytest` to ensure all tests pass.
3. Submit a pull request describing your changes.

## Licence

MIT Licence – see [LICENSE](LICENSE) for details.
