# 🍎 Fruit Classifier AI Workflow

Custom CNN trained from scratch in PyTorch that classifies fruits and generates nutritional insights via a local RAG pipeline using Gemma and Ollama — no cloud required.

## Setup

```bash
pip install -r requirements.txt
ollama pull gemma4:e2b
ollama serve
```

## Model Weights

[Download from Hugging Face](https://huggingface.co/tarakaprabhchinta/fruit-classifier) and place in project root.

## Usage

- `test.ipynb` — single image inference
- `rag_pipeline.ipynb` — full RAG pipeline

## License

MIT
