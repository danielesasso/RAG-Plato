# Text Summarization Pipeline

A modular, configurable pipeline for **summarizing educational text content**, generating flashcards and MCQs, and storing results in **LanceDB**. Includes an interactive **Streamlit app** with hierarchical summarization support.

---

## Features

- **Hierarchical or flat text summarization** using Ollama LLM
- Adjustable **chunk size, batch size, and hierarchy depth**
- **Streamlit app** with:
  - File upload and processing
  - Toggle between flat and hierarchical summarization
  - Flashcard and MCQ generation from uploaded files or existing database
  - Export options (CSV, Excel)
  - Database explorer with hierarchy filters and search
- **LanceDB** vector database for storing summaries, flashcards, and MCQs
- **Auto-creation** of missing database tables
- **Detailed logging** of all summarization steps

---

## Project Scope

- **Focus:** Italian educational text content
- **Input:** Plain text files (`.txt`)
- **Output:** Summaries, flashcards, MCQs, exports
- **Note:** Audio processing is **not currently supported**

---

## Setup Instructions

1. **Clone the repository**

2. **Create a Python environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```
The version must be 3.9.21

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Ensure Ollama is installed and running**

- [https://ollama.com/](https://ollama.com/)
- Pull the required model:

```bash
ollama pull llama3.2
```

5. **Run the app**

```bash
python -m streamlit run src/app.py
```

This will check dependencies, ensure models are available, and launch the Streamlit app in your browser.

---

## Usage

### **1. Upload a text file**

- Use the sidebar to upload `.txt` files
- Choose **Flat** or **Hierarchical** summarization mode
- Adjust chunk size, summary length, batch size, and max levels as needed

### **2. Process the file**

- Click **"Process Text"**
- View chunk summaries and final summary
- Export results as CSV or Excel

### **3. Generate study materials**

- Generate **flashcards** or **MCQs** from uploaded file or existing database
- Flashcards and MCQs are stored in LanceDB

### **4. Explore database**

- Use the **Database Explorer** to search and filter summaries
- Filter by hierarchy level or search term

---

## Project Structure

```
src/
  app.py                 # Streamlit app
  pipeline.py            # Summarization pipeline
  run-script.py          # Launcher script
  generate_mcq.py        # MCQ generation
  generate_simple_flashcards.py  # Flashcard generation
  cleaned_speech_text.txt # Example input
memory-bank/             # Project documentation and context
lancedb/                 # LanceDB data (ignored by git)
test_hierarchical_pipeline.py  # Test script for hierarchical pipeline
requirements.txt
.gitignore
README.md
```

---

## Future Work

- Benchmark comparison and reporting
- Metadata enrichment and filtering improvements
- Anki export options
- Visualization of hierarchical summary tree
- Further UI polish and error handling

---

## Known Issues

- Flashcard table availability may require manual refresh
- Audio processing is **not implemented**
- Benchmark criteria and report formats to be defined

---

## License

[MIT License](LICENSE)

---

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [LanceDB](https://lancedb.com/)
- [Ollama](https://ollama.com/)
