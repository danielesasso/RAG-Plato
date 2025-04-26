import streamlit as st
import os
import pandas as pd
import hashlib
from datetime import datetime
import tempfile
from pathlib import Path
import lancedb
from streamlit_mermaid import st_mermaid

from pipeline import (
    chunk_text, 
    generate_summary, 
    process_transcriptions, 
    process_transcriptions_hierarchical,
    collect_summarized_sections,
    generate_final_summary, 
    get_file_hash,
    create_summary_dataframe,
)

from generate_simple_flashcards import generate_simple_flashcard_for_chunk, init_simple_flashcard_table
from generate_mcq import generate_mcq_for_chunk, init_mcq_table

# Set page config
st.set_page_config(
    page_title="Text Summarization Pipeline",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global LanceDB Connection and Table Initialization ---
if 'db' not in st.session_state:
    try:
        st.session_state['db'] = lancedb.connect("./lancedb")
    except Exception as e:
        st.session_state['db'] = None
        st.error(f"Failed to connect to LanceDB: {e}")
db = st.session_state.get('db')

# Ensure MCQ table exists, create if missing
if db is not None:
    try:
        existing_tables = db.table_names()
        if "mcq_questions" not in existing_tables:
            mcq_table = init_mcq_table()
            st.session_state['mcq_table'] = mcq_table
            st.success("Created missing MCQ table at startup.")
        else:
            mcq_table = db.open_table("mcq_questions")
            st.session_state['mcq_table'] = mcq_table
    except Exception as e:
        st.warning(f"Error ensuring MCQ table exists: {e}")
        st.session_state['mcq_table'] = None

# Insert placeholder MCQ if table empty
try:
    mcq_table = st.session_state.get('mcq_table')
    if mcq_table is not None:
        df = mcq_table.to_pandas()
        if df.empty:
            import pandas as pd
            placeholder_mcq = pd.DataFrame([{
                "question": "PLACEHOLDER",
                "choices": ["A", "B", "C", "D"],
                "answer": "A",
                "lesson_number": -1,
                "topic": "placeholder",
                "difficulty": "",
                "source_chunk_id": -1,
                "generated_at": str(pd.Timestamp.now())
            }])
            mcq_table.add(placeholder_mcq)
except:
    pass

if 'lesson_table' not in st.session_state:
    try:
        st.session_state['lesson_table'] = db.open_table("lesson_chunks") if db else None
    except Exception as e:
        st.session_state['lesson_table'] = None
        st.warning(f"Could not open lesson_chunks table: {e}")

# Insert placeholder MCQ if table empty
try:
    mcq_table = st.session_state.get('mcq_table')
    if mcq_table is not None:
        df = mcq_table.to_pandas()
        if df.empty:
            import pandas as pd
            placeholder_mcq = pd.DataFrame([{
                "question": "PLACEHOLDER",
                "choices": ["A", "B", "C", "D"],
                "answer": "A",
                "lesson_number": -1,
                "topic": "placeholder",
                "difficulty": "",
                "source_chunk_id": -1,
                "generated_at": str(pd.Timestamp.now())
            }])
            mcq_table.add(placeholder_mcq)
except:
    pass

if 'flashcard_table' not in st.session_state:
    try:
        st.session_state['flashcard_table'] = db.open_table("simple_flashcards") if db else None
    except Exception as e:
        st.warning(f"Could not open simple_flashcards table: {e}")
        try:
            st.session_state['flashcard_table'] = init_simple_flashcard_table()
            st.success("Created missing flashcard table.")
        except Exception as ce:
            st.session_state['flashcard_table'] = None
            st.error(f"Failed to create flashcard table: {ce}")

# Insert placeholder flashcard if table empty
try:
    flashcard_table = st.session_state.get('flashcard_table')
    if flashcard_table is not None:
        df = flashcard_table.to_pandas()
        if df.empty:
            import pandas as pd
            placeholder_flashcard = pd.DataFrame([{
                "front": "PLACEHOLDER",
                "back": "PLACEHOLDER",
                "lesson_number": -1,
                "topic": "placeholder",
                "difficulty": "",
                "source_chunk_id": -1,
                "generated_at": str(pd.Timestamp.now())
            }])
            flashcard_table.add(placeholder_flashcard)
except:
    pass

# --- Sidebar: File Upload and Config ---
st.sidebar.header("Configuration")

# Navigation
page = st.sidebar.selectbox("Navigate", ["Home", "Flashcard Generator", "MCQ Generator", "Summary Tree"])

# File upload and processing config (global, always visible)
st.sidebar.subheader("Process New File")
uploaded_file = st.sidebar.file_uploader("Upload a text file", type=['txt'])

summarization_mode = st.sidebar.radio(
    "Summarization Mode",
    ["Flat", "Hierarchical"],
    index=0
)

chunk_size = st.sidebar.slider("Chunk Size (words)", 100, 1000, 500, 50)
summary_words = st.sidebar.slider("Summary Length (words)", 20, 200, 50, 10)

if summarization_mode == "Hierarchical":
    batch_size = st.sidebar.number_input("Batch Size (summaries per group)", min_value=1, max_value=10, value=2)
    max_levels = st.sidebar.number_input("Max Hierarchy Levels", min_value=1, max_value=5, value=2)
else:
    batch_size = None
    max_levels = None

lesson_num = st.sidebar.number_input("Lesson Number", min_value=1, value=1)
lesson_topic = st.sidebar.text_input("Lesson Topic", "Text Processing")

# Initialize session state variables
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'chunk_summaries' not in st.session_state:
    st.session_state.chunk_summaries = {}
if 'final_summary' not in st.session_state:
    st.session_state.final_summary = ""
if 'file_flashcards' not in st.session_state:
    st.session_state.file_flashcards = []
if 'file_mcqs' not in st.session_state:
    st.session_state.file_mcqs = []

# Save uploaded file to temp and hash
if uploaded_file is not None and 'temp_file_path' not in st.session_state:
    temp_dir = tempfile.gettempdir()
    temp_file = Path(temp_dir) / uploaded_file.name
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.temp_file_path = str(temp_file)
    st.session_state.file_hash = get_file_hash(st.session_state.temp_file_path)

    # Check cache
    lesson_table = st.session_state.get('lesson_table')
    if lesson_table:
        try:
            existing = lesson_table.search().where(f"file_hash = '{st.session_state.file_hash}'").limit(1).to_pandas()
            if not existing.empty:
                st.sidebar.success("Found cached results for this file!")
        except Exception as e:
            st.sidebar.warning(f"Could not check cache: {e}")

# Process button
if uploaded_file is not None:
    if st.sidebar.button("Process Text"):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Processing text file...")

            with st.spinner("Generating chunk summaries..."):
                if summarization_mode == "Flat":
                    import src.pipeline as pipeline_module
                    text_data = open(st.session_state.temp_file_path, 'r', encoding='utf-8').read()
                    chunks = pipeline_module.chunk_text(text_data, chunk_size=chunk_size)
                    prior_summary = "None"
                    chunk_summaries = {}
                    total_chunks = len(chunks)
                    for i, chunk in enumerate(chunks):
                        status_text.text(f"Processing chunk {i+1} of {total_chunks}...")
                        current_summary = pipeline_module.generate_summary(prior_summary, chunk, summary_words)
                        chunk_summaries[f"chunk_{i+1}"] = current_summary
                        prior_summary = current_summary
                        progress_bar.progress(int(100 * (i + 1) / total_chunks))
                    status_text.text("Generating final summary...")
                    st.session_state.chunk_summaries = chunk_summaries
                    progress_bar.progress(100)
                    st.session_state.processing_complete = True
                    status_text.text("Processing complete!")
                else:
                    status_text.text("Generating hierarchical summaries...")
                    chunk_summaries = process_transcriptions_hierarchical(
                        st.session_state.temp_file_path,
                        summarized_words=summary_words,
                        chunk_size=chunk_size,
                        batch_size=batch_size,
                        max_levels=max_levels,
                        lesson_number=lesson_num,
                        topic=lesson_topic
                    )
                    status_text.text("Generating final summary...")
                    st.session_state.chunk_summaries = chunk_summaries
                    progress_bar.progress(100)
                    st.session_state.processing_complete = True
                    status_text.text("Processing complete!")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# --- Main Content Area ---
if page == "Home":
    st.title("📚 Text Summarization Pipeline")
    st.markdown("""
    Welcome! This app processes educational text files, generates summaries, flashcards, and MCQs.
    
    **Use the sidebar to upload and process a file.**
    
    Then navigate to:
    - **Flashcard Generator** to create flashcards
    - **MCQ Generator** to create multiple choice questions
    """)

    if st.session_state.processing_complete:
        st.header("📋 Final Summary")
        st.markdown(st.session_state.final_summary)

        st.header("🧩 Chunk Summaries")
        if isinstance(st.session_state.chunk_summaries, dict):
            sorted_chunks = sorted(
                st.session_state.chunk_summaries.keys(),
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            cols = st.columns(3)
            for i, chunk_key in enumerate(sorted_chunks):
                with cols[i % 3]:
                    with st.expander(f"Chunk {i+1}"):
                        st.markdown(st.session_state.chunk_summaries[chunk_key])
        else:
            st.warning("No chunk summaries available.")

        st.header("📊 Summary Data")
        df = create_summary_dataframe(
            st.session_state.chunk_summaries,
            st.session_state.final_summary,
            lesson_num,
            lesson_topic
        )
        st.dataframe(df)

        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "summary_data.csv", "text/csv")
        with col2:
            excel_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
            df.to_excel(excel_file.name, index=False)
            with open(excel_file.name, 'rb') as f:
                excel_data = f.read()
            st.download_button("Download Excel", excel_data, "summary_data.xlsx")

    # --- Database Explorer ---
    with st.expander("Database Explorer"):
        st.subheader("Explore Summaries in LanceDB")

        lesson_table = st.session_state.get('lesson_table')
        if not lesson_table:
            st.warning("Lesson chunks table not available.")
        else:
            level_filter = st.selectbox(
                "Hierarchy Level Filter",
                ["All Levels", "Chunks (level 0)", "Batch Summaries (level 1+)", "Final Summaries (highest level)"]
            )
            search_query_db = st.text_input("Optional Search Term", key="db_search")

            if st.button("Refresh Results"):
                try:
                    query_obj = lesson_table.search()

                    if search_query_db.strip():
                        query_obj = query_obj.search(search_query_db.strip())

                    if level_filter == "Chunks (level 0)":
                        query_obj = query_obj.where("level = 0")
                    elif level_filter == "Batch Summaries (level 1+)":
                        query_obj = query_obj.where("level > 0")
                    elif level_filter == "Final Summaries (highest level)":
                        df_all = lesson_table.to_pandas()
                        if not df_all.empty:
                            max_level = df_all['level'].max()
                            query_obj = query_obj.where(f"level = {max_level}")

                    results = query_obj.limit(100).to_pandas()
                    if not results.empty:
                        st.dataframe(results[[
                            'text', 'level', 'batch_index', 'parent_batch',
                            'lesson_number', 'section', 'topic', 'score'
                        ]])
                    else:
                        st.info("No results found.")
                except Exception as e:
                    st.error(f"Error querying database: {e}")

elif page == "Flashcard Generator":
    st.title("🃏 Flashcard Generator")

    # --- Reconnect to LanceDB if needed ---
    db = st.session_state.get('db')
    if db is None:
        try:
            db = lancedb.connect("./lancedb")
            st.session_state['db'] = db
        except:
            db = None

    lesson_table = st.session_state.get('lesson_table')
    flashcard_table = st.session_state.get('flashcard_table')

    if (lesson_table is None or flashcard_table is None) and db is not None:
        try:
            if lesson_table is None:
                lesson_table = db.open_table("lesson_chunks")
                st.session_state['lesson_table'] = lesson_table
            if flashcard_table is None:
                flashcard_table = db.open_table("simple_flashcards")
                st.session_state['flashcard_table'] = flashcard_table
        except:
            pass

    # --- Check if existing chunk summaries are available in DB ---
    existing_data_available = False
    if lesson_table:
        try:
            df_existing = lesson_table.search().where("section LIKE 'chunk_%'").limit(1).to_pandas()
            if not df_existing.empty:
                existing_data_available = True
        except:
            pass

    st.subheader("Generate Flashcards from Processed File")
    if st.session_state.get('processing_complete', False) or existing_data_available:
        if st.button("Generate Flashcards"):
            # Lazy init flashcard table
            db = st.session_state.get('db')
            flashcard_table = st.session_state.get('flashcard_table')
            if (flashcard_table is None or db is None):
                try:
                    if db is None:
                        db = lancedb.connect("./lancedb")
                        st.session_state['db'] = db
                    # Always create empty table if missing
                    if "simple_flashcards" not in db.table_names():
                        flashcard_table = init_simple_flashcard_table()
                        st.success("Initialized empty Flashcard table.")
                    else:
                        flashcard_table = db.open_table("simple_flashcards")
                    st.session_state['flashcard_table'] = flashcard_table
                except Exception as e:
                    st.error(f"Failed to initialize flashcard table: {e}")
                    flashcard_table = None

            flashcards = []
            if flashcard_table:
                # Use summaries from session if available, else from DB
                chunk_summaries = st.session_state.get('chunk_summaries', {})
                if not chunk_summaries and lesson_table:
                    try:
                        df_chunks = lesson_table.search().where("section LIKE 'chunk_%'").limit(1000).to_pandas()
                        for idx, row in df_chunks.iterrows():
                            chunk_summaries[f"chunk_{row['index']}"] = row['text']
                    except:
                        chunk_summaries = {}

                for idx, (chunk_key, chunk_text) in enumerate(chunk_summaries.items()):
                    card = generate_simple_flashcard_for_chunk(
                        chunk_text,
                        lesson_num,
                        lesson_topic,
                        idx,
                        difficulty=""
                    )
                    if card['front'] and card['back']:
                        flashcards.append(card)
                if flashcards:
                    # Remove placeholder flashcards before adding real ones
                    try:
                        df_existing = flashcard_table.to_pandas()
                        placeholders = df_existing[df_existing['lesson_number'] == -1]
                        if not placeholders.empty:
                            flashcard_table.delete(f"lesson_number = -1")
                    except:
                        pass

                    flashcard_table.add(pd.DataFrame(flashcards))
                    st.session_state.file_flashcards = flashcards
                    st.success(f"Generated and saved {len(flashcards)} flashcards.")
                else:
                    st.warning("No flashcards generated.")
            else:
                st.warning("Flashcard table not available.")
    else:
        st.info("Please process a file on the Home page or ensure existing data is available in the database.")

    # --- Flashcards in Database ----------------------------------------------------
    flashcard_table = st.session_state.get("flashcard_table")
    if flashcard_table:
        try:
            df_flashcards = flashcard_table.to_pandas()
            if not df_flashcards.empty:
                st.subheader("Flashcards in Database")
                # Hide the single dummy row, if it still exists
                df_flashcards = df_flashcards[df_flashcards["lesson_number"] != -1]

                st.dataframe(
                    df_flashcards[
                        [
                            "front",
                            "back",
                            "lesson_number",
                            "topic",
                            "difficulty",
                            "source_chunk_id",
                            "generated_at",
                        ]
                    ]
                )
            else:
                st.info("No flashcards found in database.")
        except Exception as e:
            st.warning(f"Could not load flashcards from database: {e}")
    else:
        st.warning("Flashcard table not available.")

    # --- Flashcard Search ----------------------------------------------------------
    st.subheader("🔍 Search Flashcards")

    search_query = st.text_input("Enter a keyword or phrase")
    top_k = st.slider("Top-K (semantic)", 1, 20, 10, help="How many nearest-neighbor hits to fetch")

    if st.button("Search"):
        flash_tbl = st.session_state.get("flashcard_table")
        if flash_tbl is None:
            st.warning("Flashcard table not available.")
        elif not search_query.strip():
            st.info("Type something first 🙂")
        else:
            import pandas as pd
            from lancedb.embeddings import EmbeddingFunctionRegistry

            # ---------- 1. keyword hits (cheap, pandas) ----------
            df_cards = flash_tbl.to_pandas()
            kw_hits = df_cards[
                df_cards["front"].str.contains(search_query, case=False, na=False)
                | df_cards["back"].str.contains(search_query, case=False, na=False)
            ]

            # ---------- 2. semantic hits (vector search) ----------
            sem_hits = (
                flash_tbl                     # table knows its embedding function
                .search(search_query)         # just pass the text!
                .limit(top_k)
                .to_pandas()
                .drop(columns=["vector"], errors="ignore")
            )

            # ---------- unify + dedupe ----------
            hits = (
                pd.concat([kw_hits, sem_hits])
                .drop_duplicates(subset=["front", "back"])
                .reset_index(drop=True)
            )

            if hits.empty:
                st.info("No flashcards matched your query.")
            else:
                st.success(f"Found {len(hits)} matching flashcards.")

                # cache in session for downstream review mode
                st.session_state.search_flashcards = hits[
                    ["front", "back", "lesson_number", "topic", "difficulty"]
                ].to_dict(orient="records")
                st.session_state.search_idx = 0
                st.session_state.search_show_answer = False

     # --- Search Review Mode --------------------------------------------------------
    if "search_flashcards" in st.session_state and st.session_state.search_flashcards:
        cards = st.session_state.search_flashcards
        idx = st.session_state.search_idx
        card = cards[idx]

        # callbacks -------------------------------------------------
        def _s_prev():
            st.session_state.search_idx = max(0, st.session_state.search_idx - 1)
            st.session_state.search_show_answer = False

        def _s_next():
            st.session_state.search_idx = min(
                len(cards) - 1, st.session_state.search_idx + 1
            )
            st.session_state.search_show_answer = False

        def _s_reveal():
            st.session_state.search_show_answer = True

        # UI --------------------------------------------------------
        st.markdown(f"### 🔎 Result {idx + 1} of {len(cards)}")
        st.markdown(f"**Question:** {card['front']}")
        st.caption(
            f"Lesson: {card.get('lesson_number','N/A')}  |  "
            f"Topic: {card.get('topic','N/A')}  |  "
            f"Difficulty: {card.get('difficulty','N/A')}"
        )

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.button("Previous", on_click=_s_prev, disabled=idx == 0)
        with c2:
            st.button("Show Answer", on_click=_s_reveal, disabled=st.session_state.search_show_answer)
        with c3:
            st.button("Next", on_click=_s_next, disabled=idx == len(cards) - 1)

        if st.session_state.search_show_answer:
            st.markdown(f"**Answer:** {card['back']}")


    # --- Flashcard Visualization --------------------------------------------------
    flashcard_table = st.session_state.get("flashcard_table")
    if flashcard_table:
        try:
            df_flashcards = flashcard_table.to_pandas()
            if df_flashcards.empty:
                st.info("No flashcards found in database.")
            else:
                st.subheader("Flashcards Sequential Review Mode")

                # ---------- helper callbacks ----------
                def _start_review():
                    """Load all flashcards into session state and reset counters."""
                    st.session_state.flashcard_list = df_flashcards[
                        ["front", "back", "lesson_number", "topic", "difficulty"]
                    ].to_dict(orient="records")
                    st.session_state.current_flashcard_index = 0
                    st.session_state.show_answer = False

                def _prev_card():
                    st.session_state.current_flashcard_index = max(
                        0, st.session_state.current_flashcard_index - 1
                    )
                    st.session_state.show_answer = False

                def _next_card():
                    st.session_state.current_flashcard_index = min(
                        len(st.session_state.flashcard_list) - 1,
                        st.session_state.current_flashcard_index + 1,
                    )
                    st.session_state.show_answer = False

                def _reveal():
                    st.session_state.show_answer = True

                # ---------- session‑state defaults ----------
                for k, v in {
                    "flashcard_list": [],
                    "current_flashcard_index": 0,
                    "show_answer": False,
                }.items():
                    st.session_state.setdefault(k, v)

                # ---------- UI ----------
                st.button("Start Review Mode", on_click=_start_review)

                if st.session_state.flashcard_list:
                    cards = st.session_state.flashcard_list
                    idx = st.session_state.current_flashcard_index
                    card = cards[idx]

                    st.markdown(f"### Flashcard {idx + 1} of {len(cards)}")
                    st.markdown(f"**Question:** {card['front']}")
                    st.caption(
                        f"Lesson: {card.get('lesson_number','N/A')}  |  "
                        f"Topic: {card.get('topic','N/A')}  |  "
                        f"Difficulty: {card.get('difficulty','N/A')}"
                    )

                    col_prev, col_show, col_next = st.columns([1, 1, 1])
                    with col_prev:
                        st.button("Previous", on_click=_prev_card, disabled=idx == 0)
                    with col_show:
                        st.button(
                            "Show Answer",
                            on_click=_reveal,
                            disabled=st.session_state.show_answer,
                        )
                    with col_next:
                        st.button(
                            "Next",
                            on_click=_next_card,
                            disabled=idx == len(cards) - 1,
                        )

                    if st.session_state.show_answer:
                        st.markdown(f"**Answer:** {card['back']}")

        except Exception as e:
            st.warning(f"Could not load flashcards from database: {e}")
    else:
        st.warning("Flashcard table not available.")


elif page == "MCQ Generator":
    st.title("❓ MCQ Generator")

    # --- Reconnect to LanceDB if needed ---
    db = st.session_state.get('db')
    if db is None:
        try:
            db = lancedb.connect("./lancedb")
            st.session_state['db'] = db
        except:
            db = None

    lesson_table = st.session_state.get('lesson_table')
    mcq_table = st.session_state.get('mcq_table')

    if (lesson_table is None or mcq_table is None) and db is not None:
        try:
            if lesson_table is None:
                lesson_table = db.open_table("lesson_chunks")
                st.session_state['lesson_table'] = lesson_table
            if mcq_table is None:
                mcq_table = db.open_table("mcq_questions")
                st.session_state['mcq_table'] = mcq_table
        except:
            pass

    # --- Check if existing chunk summaries are available in DB ---
    existing_data_available = False
    if lesson_table:
        try:
            df_existing = lesson_table.search().where("section LIKE 'chunk_%'").limit(1).to_pandas()
            if not df_existing.empty:
                existing_data_available = True
        except:
            pass

    st.subheader("Generate MCQs from Processed File")
    if st.session_state.get('processing_complete', False) or existing_data_available:
        if st.button("Generate MCQs"):
            # Lazy init MCQ table
            db = st.session_state.get('db')
            mcq_table = st.session_state.get('mcq_table')
            if (mcq_table is None or db is None):
                try:
                    if db is None:
                        db = lancedb.connect("./lancedb")
                        st.session_state['db'] = db
                    # Always create empty table if missing
                    if "mcq_questions" not in db.table_names():
                        mcq_table = init_mcq_table()
                        st.success("Initialized empty MCQ table.")
                    else:
                        mcq_table = db.open_table("mcq_questions")
                    st.session_state['mcq_table'] = mcq_table
                except Exception as e:
                    st.error(f"Failed to initialize MCQ table: {e}")
                    mcq_table = None

            mcqs = []
            if mcq_table:
                # Use summaries from session if available, else from DB
                chunk_summaries = st.session_state.get('chunk_summaries', {})
                if not chunk_summaries and lesson_table:
                    try:
                        df_chunks = lesson_table.search().where("section LIKE 'chunk_%'").limit(1000).to_pandas()
                        for idx, row in df_chunks.iterrows():
                            chunk_summaries[f"chunk_{row['index']}"] = row['text']
                    except:
                        chunk_summaries = {}

                for idx, (chunk_key, chunk_text) in enumerate(chunk_summaries.items()):
                    mcq = generate_mcq_for_chunk(
                        chunk_text,
                        lesson_num,
                        lesson_topic,
                        idx,
                        difficulty=""
                    )
                    if mcq['question'] and mcq['choices'] and mcq['answer']:
                        mcqs.append(mcq)
                if mcqs:
                    # Remove placeholder MCQs before adding real ones
                    try:
                        df_existing = mcq_table.to_pandas()
                        placeholders = df_existing[df_existing['lesson_number'] == -1]
                        if not placeholders.empty:
                            mcq_table.delete(f"lesson_number = -1")
                    except:
                        pass

                    mcq_table.add(pd.DataFrame(mcqs))
                    st.session_state.file_mcqs = mcqs
                    st.success(f"Generated and saved {len(mcqs)} MCQs.")
                else:
                    st.warning("No MCQs generated.")
            else:
                st.warning("MCQ table not available.")
    else:
        st.info("Please process a file on the Home page or ensure existing data is available in the database.")

    # --- MCQ Visualization -------------------------------------------------------
    mcq_table = st.session_state.get("mcq_table")
    if mcq_table:
        try:
            df_mcqs = mcq_table.to_pandas()
            if not df_mcqs.empty:
                st.subheader("MCQs in Database")
                st.dataframe(
                    df_mcqs[
                        [
                            "question",
                            "choices",
                            "answer",
                            "lesson_number",
                            "topic",
                            "difficulty",
                            "source_chunk_id",
                            "generated_at",
                        ]
                    ]
                )
            else:
                st.info("No MCQs found in database.")
        except Exception as e:
            st.warning(f"Could not load MCQs from database: {e}")
    else:
        st.warning("MCQ table not available.")
        
    # --- MCQ Search ---------------------------------------------------------------
    st.subheader("🔍 Search MCQs")

    mcq_query  = st.text_input("Keyword or phrase")
    mcq_top_k  = st.slider("Top-K (semantic)", 1, 20, 10)

    if st.button("Search MCQs"):
        tbl = st.session_state.get("mcq_table")
        if tbl is None:
            st.warning("MCQ table not available.")
        elif not mcq_query.strip():
            st.info("Type something first 🙂")
        else:
            import pandas as pd
            from lancedb.embeddings import EmbeddingFunctionRegistry  # only needed if you later embed manually

            # ---------- 1. keyword hits ----------
            df_all = tbl.to_pandas()
            kw_hits = df_all[
                df_all["question"].str.contains(mcq_query, case=False, na=False)
                | df_all["choices"].astype(str).str.contains(mcq_query, case=False, na=False)
            ]

            # ---------- 2. semantic hits ----------
            # The table already knows its embedder, so just pass the text
            sem_hits = (
                tbl.search(mcq_query)
                .limit(mcq_top_k)
                .to_pandas()
                .drop(columns=["vector"], errors="ignore")
            )

            # ---------- unify results ----------
            hits = (
                pd.concat([kw_hits, sem_hits])
                .drop_duplicates(subset=["question", "answer"])
                .reset_index(drop=True)
            )

            if hits.empty:
                st.info("No MCQs matched your query.")
            else:
                st.success(f"Found {len(hits)} matching MCQs.")

                st.session_state.search_mcqs = [
                    {
                        "question": row.question,
                        "choices": list(row.choices),
                        "answer": row.answer,
                    }
                    for _, row in hits.iterrows()
                ]
                st.session_state.search_mcq_idx       = 0
                st.session_state.search_mcq_choice    = None
                st.session_state.search_mcq_feedback  = False

    # --- Search Practice Mode -----------------------------------------------------
    if st.session_state.get("search_mcqs"):
        mcqs = st.session_state.search_mcqs
        idx   = st.session_state.search_mcq_idx
        item  = mcqs[idx]

        # ---------- callbacks ----------
        def _s_prev():
            st.session_state.search_mcq_idx      = max(0, idx - 1)
            st.session_state.search_mcq_choice   = None
            st.session_state.search_mcq_feedback = False

        def _s_next():
            st.session_state.search_mcq_idx      = min(len(mcqs) - 1, idx + 1)
            st.session_state.search_mcq_choice   = None
            st.session_state.search_mcq_feedback = False

        def _s_check():
            st.session_state.search_mcq_feedback = True

        # ---------- UI ----------
        st.markdown(f"### 🔎 Result {idx + 1} of {len(mcqs)}")
        st.markdown(f"**{item['question']}**")

        st.session_state.search_mcq_choice = st.radio(
            "Select your answer:",
            item["choices"],
            index=0
            if st.session_state.search_mcq_choice is None
            else item["choices"].index(st.session_state.search_mcq_choice),
            key="search_mcq_radio",
        )

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.button("Previous", on_click=_s_prev, disabled=(idx == 0))
        with c2:
            st.button("Check Answer", on_click=_s_check)
        with c3:
            st.button("Next", on_click=_s_next, disabled=(idx == len(mcqs) - 1))

        # feedback -----------------------
        if st.session_state.search_mcq_feedback:
            sel = st.session_state.search_mcq_choice or ""
            correct = str(item["answer"]).strip()
            if sel.split(")")[0].strip() == correct:
                st.success("Correct! 🎉")
            else:
                st.error(f"Incorrect. The correct answer is **{correct}**.")



    # --- MCQ Practice Mode -------------------------------------------------------
    st.subheader("📝 MCQ Practice Mode")

    # -------- helper callbacks --------
    def _load_mcqs():
        """Pull all MCQs from the DB into session state."""
        tbl = st.session_state.get("mcq_table")
        if tbl is None:
            st.warning("MCQ table not available.")
            return
        df = tbl.to_pandas()
        if df.empty:
            st.warning("No MCQs found in database.")
            return

        st.session_state.mcq_list = [
            {
                "question": row.question,
                "choices": list(row.choices),
                "answer": row.answer,
            }
            for _, row in df.iterrows()
        ]
        st.session_state.current_mcq_index = 0
        st.session_state.selected_choice = None
        st.session_state.show_feedback = False
        st.success(f"Loaded {len(st.session_state.mcq_list)} MCQs for practice.")


    def _prev_question():
        st.session_state.current_mcq_index = max(
            0, st.session_state.current_mcq_index - 1
        )
        st.session_state.selected_choice = None
        st.session_state.show_feedback = False


    def _next_question():
        st.session_state.current_mcq_index = min(
            len(st.session_state.mcq_list) - 1, st.session_state.current_mcq_index + 1
        )
        st.session_state.selected_choice = None
        st.session_state.show_feedback = False


    def _check_answer():
        st.session_state.show_feedback = True


    # -------- session‑state defaults --------
    for k, v in {
        "mcq_list": [],
        "current_mcq_index": 0,
        "selected_choice": None,
        "show_feedback": False,
    }.items():
        st.session_state.setdefault(k, v)

    # -------- UI  --------
    st.button("Start Practice Mode", on_click=_load_mcqs)

    if st.session_state.mcq_list:
        idx = st.session_state.current_mcq_index
        current_mcq = st.session_state.mcq_list[idx]

        st.markdown(f"**Question {idx + 1} of {len(st.session_state.mcq_list)}**")
        st.markdown(f"**{current_mcq['question']}**")

        # radio key is constant; we reset its value in the prev/next callbacks
        st.session_state.selected_choice = st.radio(
            "Select your answer:",
            current_mcq["choices"],
            index=0 if st.session_state.selected_choice is None else
            current_mcq["choices"].index(st.session_state.selected_choice),
            key="mcq_radio",
        )

        # navigation + answer‑check buttons
        col_prev, col_check, col_next = st.columns([1, 1, 1])
        with col_prev:
            st.button("Previous", on_click=_prev_question, disabled=idx == 0)
        with col_check:
            st.button("Check Answer", on_click=_check_answer)
        with col_next:
            st.button(
                "Next",
                on_click=_next_question,
                disabled=idx == len(st.session_state.mcq_list) - 1,
            )

        # feedback ---------------------------------------------------------------
        if st.session_state.show_feedback:
            import numpy as np

            selected_raw = st.session_state.selected_choice or ""
            # extract letter before ")"   e.g. "A) Foo" -> "A"
            selected_letter = selected_raw.split(")")[0].strip()

            correct_raw = current_mcq["answer"]
            if isinstance(correct_raw, (list, tuple, np.ndarray)):
                correct_raw = correct_raw[0] if len(correct_raw) else ""
            correct_letter = str(correct_raw).strip()

            if selected_letter == correct_letter:
                st.success("Correct! 🎉")
            else:
                st.error(f"Incorrect. The correct answer is **{correct_letter}**.")


elif page == "Summary Tree":
    st.title("🌳 Hierarchical Summary Tree")

    db = st.session_state.get('db')
    if db is None:
        try:
            db = lancedb.connect("./lancedb")
            st.session_state['db'] = db
        except:
            db = None

    lesson_table = st.session_state.get('lesson_table')
    if lesson_table is None and db is not None:
        try:
            lesson_table = db.open_table("lesson_chunks")
            st.session_state['lesson_table'] = lesson_table
        except:
            lesson_table = None

    if not lesson_table:
        st.warning("Lesson chunks table not available.")
    else:
        try:
            df = lesson_table.to_pandas()
            if df.empty:
                st.info("No summaries found in database.")
            else:
                # Topic filter
                unique_topics = sorted(set(
                    t for t in df['topic'].dropna().unique() if str(t).strip() != ""
                ))
                selected_topic = st.selectbox("Select Topic", ["All Topics"] + unique_topics)
                if selected_topic != "All Topics":
                    df = df[df['topic'] == selected_topic]

                if df.empty:
                    st.info("No summaries found for the selected topic.")
                else:
                    max_level = df['level'].max()
                    roots = df[df['level'] == max_level]

                    st.subheader("Hierarchical Summaries")

                    # Build a flat list of all nodes with indentation
                    nodes = []

                    def collect_nodes(node, df, indent=0):
                        prefix = " " * indent  # Unicode em-space for indentation
                        header = f"{prefix}Level {node['level']} - {node['section']}"
                        if node.get('topic'):
                            header += f" (Topic: {node['topic']})"
                        if 'score' in node and node['score'] != 0.0:
                            header += f" (Score: {node['score']:.2f})"
                        nodes.append((header, node))

                        children = df[
                            (df['level'] == node['level'] - 1) &
                            (df['parent_batch'] == node['batch_index'])
                        ]
                        for _, child in children.iterrows():
                            collect_nodes(child, df, indent + 1)

                    for _, root in roots.iterrows():
                        collect_nodes(root, df)

                    for header, node in nodes:
                        with st.expander(header):
                            st.write(node['text'])
                            st.caption(f"Lesson: {node['lesson_number']} | Batch: {node['batch_index']} | Parent: {node['parent_batch']} | Time: {node['processed_at']}")

                    # --- Mermaid Diagram ---
                    mermaid_lines = ["graph TD"]
                    node_ids = {}

                    # Assign unique IDs and labels
                    for idx, row in df.iterrows():
                        node_id = f"L{row['level']}_B{row['batch_index']}_{idx}"
                        label = f"{row['section']}"
                        node_ids[(row['level'], row['batch_index'], idx)] = node_id
                        mermaid_lines.append(f'  {node_id}["{label}"]')

                    # Add edges
                    for idx, row in df.iterrows():
                        child_id = node_ids.get((row['level'], row['batch_index'], idx))

                        # Heuristic: connect chunks to nearest level 1 batch with same topic and lesson
                        if row['level'] == 0:
                            parent_candidates = df[
                                (df['level'] == 1) &
                                (df['topic'] == row['topic']) &
                                (df['lesson_number'] == row['lesson_number'])
                            ]
                            if not parent_candidates.empty:
                                # Connect to the first matching batch
                                for p_idx in parent_candidates.index:
                                    parent_row = parent_candidates.loc[p_idx]
                                    parent_id = node_ids.get((parent_row['level'], parent_row['batch_index'], p_idx))
                                    if parent_id and child_id:
                                        mermaid_lines.append(f"  {parent_id} --> {child_id}")
                                        break
                            continue

                        # For level >=1, use parent_batch and level+1 as before
                        parent_level = row['level'] + 1
                        parent_batch = row['parent_batch']
                        parent_candidates = df[
                            (df['level'] == parent_level) &
                            (df['batch_index'] == parent_batch)
                        ]
                        if not parent_candidates.empty:
                            for p_idx in parent_candidates.index:
                                parent_id = node_ids.get((parent_level, parent_batch, p_idx))
                                if parent_id and child_id:
                                    mermaid_lines.append(f"  {parent_id} --> {child_id}")
                                    break

                    mermaid_code = "\n".join(mermaid_lines)
                    st.subheader("Mermaid Diagram of Summary Hierarchy")
                    st_mermaid(mermaid_code)
        except Exception as e:
            st.error(f"Error loading summaries: {e}")

# Footer
st.markdown("---")
st.markdown("Plato by Daniele Pedranghelu & Mattia Tronci")