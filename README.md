# LLM-Based Interview Question Retriever

## Introduction
This project implements an intelligent system for retrieving and recommending relevant interview questions based on job descriptions. It utilizes Large Language Models (LLMs) and vector similarity search to match job requirements with appropriate interview questions. The system implements three different methods:
- Job Analysis (JA): Analyzes job requirements first, then generates question templates
- Job Description (JD): Directly generates question templates from job description
- Hypothetical Document Embedding (HyDE): Uses hypothetical questions to find similar existing questions

The project includes both a command-line interface `main.py` for evaluation and a web interface `app.py` for interactive use.

## Project Structure
```
llm-question-retriever/
├── config/
│   └── config.py                    # API keys, model names, paths, templates
├── data/
├── qdrant_storage/
├── services/
│   ├── evaluation/
│   │   ├── init.py
│   │   ├── evaluator.py            # Evaluation metrics calculation
│   │   └── post_processor.py
│   ├── question_analysis/
│   │   ├── init.py
│   │   └── question_analyzer.py     # Initial question template generation
│   ├── job_analysis/
│   │   ├── init.py
│   │   ├── job_analyzer.py         # JD template generation
│   │   └── metadata_generator.py    # JA and JD metadata template generation (10 templates each)
│   ├── hyde/
│   │   ├── init.py
│   │   └── hyde_generator.py       # HyDE question generation
│   ├── embedding/
│   │   ├── init.py
│   │   └── embedding_service.py     # OpenAI embeddings generation
│   ├── storage/
│   │   ├── init.py
│   │   └── qdrant_service.py       # Vector storage operations
│   └── ranking/
│       ├── init.py
│       └── custom_ranker.py        # Custom ranking logic (top-two + remaining)
├── utils/
│   ├── init.py
│   ├── data_loader.py              # Load questions, jobs data
│   └── result_processor.py         # Process and save results
├── main.py                         # Main execution script
├── app.py                          # Streamlit web application
├── eval_dataset_gen.ipynb          # Notebook for generation evaluation dataset
└── requirements.txt                # Project dependencies
```

### Prerequisites
- Python 3.10 or higher
- API keys from Anthropic (Claude) and OpenAI
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-question-retriever.git
cd llm-question-retriever
```

2. Create and activate virtual environment:
```bash
python -m venv myenv
# On Windows
myenv\Scripts\activate
# On Unix or MacOS
source myenv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a .env file in the root directory with your API keys:
```
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Usage

The project includes a pre-populated vector database (qdrant_storage), so you can directly run the Streamlit app:

```bash
streamlit run app.py
```

This will open a web interface where you can:
- Input job descriptions
- Select the method (JA/JD/HyDE)
- Choose number of questions to retrieve
- View detailed process logs
- Get recommended questions and AI analysis of missing aspects and additional generated questions

To run the full pipeline including database population and evaluation:

```bash
python main.py
```

**Note:** This is only needed if you want to rebuild the vector database or run evaluations.

### Features
- Three different methods for question retrieval
- Interactive web interface
- Detailed process logging
- AI analysis of question coverage
- Customizable number of questions
- Vector similarity search
- Custom ranking

### Limitations
- Requires API keys from Anthropic and OpenAI
- Results quality depends on the underlying LLM responses
- Processing time varies based on method selected
- API rate limits may apply