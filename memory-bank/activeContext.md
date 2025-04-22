# Active Context

## Current Focus
- Integrated **Streamlit app** with:
  - File upload and **configurable summarization** (flat or hierarchical)
  - Flashcard and MCQ generation from uploaded text files
  - Flashcard and MCQ generation from existing LanceDB data with topic/query filters
  - Export options (CSV, Excel)
  - **Improved database explorer** with hierarchy level filters and search
- **Hierarchical summarization pipeline** with adjustable chunk size, batch size, and max levels
- **Auto-creation of LanceDB tables** (MCQ, flashcards) if missing
- Fixing LanceDB connection and table access issues
- Improving topic filtering and UI usability

## Recent Changes
- Implemented **recursive hierarchical summarization** with metadata storage
- Added **sidebar toggle** for Flat vs. Hierarchical summarization mode
- Added **batch size** and **max levels** parameters in UI
- Updated processing logic to call appropriate pipeline based on mode
- Enhanced **logging** of all chunk and batch summaries
- Improved **database explorer** to show all summaries with hierarchy filters
- Fixed **Streamlit page config error** by moving `set_page_config()` to top
- Added **auto-creation** of MCQ and flashcard tables on startup and button press
- Improved **LanceDB reconnection** logic during flashcard generation
- Updated Memory Bank with latest project state

## Next Steps
- Finalize UI polish and error handling
- Develop benchmark comparison and reporting
- Improve metadata enrichment and filtering
- Consider Anki export options
- Further stabilize LanceDB table creation and connection handling
- Add visualization of hierarchical summary tree (optional)

## Key Decisions
- Use **LanceDB** as central data store
- Modular, batch-oriented design
- Prioritize Italian educational content
- Use Ollama LLM for summarization and question generation
- Support both **flat and hierarchical** summarization modes
- **Focus exclusively on text input** (audio not currently supported)

## Important Patterns
- Persistent DB connection and table handles
- Session state management in Streamlit
- Batch processing of chunks and recursive summarization
- Semantic search + metadata filtering
- Hierarchical metadata (`level`, `batch_index`, `parent_batch`) in LanceDB
