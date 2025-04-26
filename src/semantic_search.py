"""
semantic_search.py
~~~~~~~~~~~~~~~~~~
Utility helpers (and an optional CLI) that perform **combined semantic +
keyword search** against any LanceDB table that contains a free-text
column called `text` (or `question`/`front`/`back` for MCQs/flashcards).

The code assumes the table was created with the same embedding function
(`ollama – mxbai-embed-large`) used for `lesson_chunks`, so LanceDB can
do vector search transparently.
"""
from __future__ import annotations

import re
from typing import Optional, List

import lancedb
import pandas as pd
from lancedb.table import Table
from lancedb.embeddings import EmbeddingFunctionRegistry

# --------------------------------------------------------------------- #
#  Low-level helper                                                     #
# --------------------------------------------------------------------- #
def _keyword_filter(df: pd.DataFrame, query: str, cols: List[str]) -> pd.DataFrame:
    """
    Return only the rows where ANY of `cols` contains `query` (case-insensitive).
    """
    if not query:
        return df

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    mask = False
    for c in cols:
        if c in df.columns:
            mask = mask | df[c].astype(str).str.contains(pattern, na=False)
    return df[mask]


# --------------------------------------------------------------------- #
#  Public API                                                           #
# --------------------------------------------------------------------- #
def search_lancedb(
    table: Table,
    query: str,
    topic: Optional[str] = None,
    difficulty: Optional[str] = None,
    limit: int = 10,
    use_semantic: bool = True,
    use_keyword: bool = True,
) -> pd.DataFrame:
    """
    Combined semantic + keyword search.

    Parameters
    ----------
    table : lancedb.table.Table
        Table to query (lesson_chunks, simple_flashcards, mcq_questions,…).
    query : str
        The user’s search text.
    topic, difficulty : str | None
        Optional filters that assume the table has `topic` / `difficulty`
        columns; ignored if missing.
    limit : int
        Maximum rows returned **after** merging semantic & keyword hits.
    use_semantic / use_keyword : bool
        Toggle either search mode.
    """
    if not query.strip():
        raise ValueError("query string is blank")

    frames: list[pd.DataFrame] = []

    # ---------- semantic -------------------------------------------------------
    if use_semantic:
        sem_q = table.search(query)
        if topic:
            sem_q = sem_q.where(f"topic = '{topic}'")
        if difficulty:
            sem_q = sem_q.where(f"difficulty = '{difficulty}'")
        frames.append(sem_q.limit(limit).to_pandas())

    # ---------- keyword --------------------------------------------------------
    if use_keyword:
        # Cheap WHERE for meta-columns; text filtering in pandas for flexibility
        kw_q = table.search()  # ← no vector search
        where_clauses = []
        if topic:
            where_clauses.append(f"topic = '{topic}'")
        if difficulty:
            where_clauses.append(f"difficulty = '{difficulty}'")
        if where_clauses:
            kw_q = kw_q.where(" AND ".join(where_clauses))

        df = kw_q.to_pandas()
        text_cols = [c for c in ["text", "front", "back", "question"] if c in df.columns]
        df = _keyword_filter(df, query, text_cols)
        frames.append(df.head(limit))

    # ---------- merge & trim ---------------------------------------------------
    if not frames:
        return pd.DataFrame()

    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["_rowid"] if "_rowid" in frames[0].columns else None)
        .head(limit)
    )
    return merged


# --------------------------------------------------------------------- #
#  Simple interactive CLI (optional)                                    #
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    db = lancedb.connect("./lancedb")

    # Let the user pick which table to explore
    tables = {
        "Lesson chunks": "lesson_chunks",
        "Flashcards"   : "simple_flashcards",
        "MCQs"         : "mcq_questions",
    }
    print("Choose a table:")
    for i, (label, _) in enumerate(tables.items(), 1):
        print(f"{i}) {label}")
    choice = input("Table # [1–3]: ").strip() or "1"
    table_name = list(tables.values())[int(choice) - 1]

    if table_name not in db.table_names():
        raise SystemExit(f"Table {table_name!r} not found")

    table = db.open_table(table_name)

    print("\nInteractive Semantic Search Tool –", table_name)
    print("Type 'exit' to quit.")
    while True:
        q = input("\nSearch: ").strip()
        if q.lower() == "exit":
            break
        topic = input("Topic filter (enter to skip): ").strip() or None
        diff  = input("Difficulty filter (enter to skip): ").strip() or None
        k     = input("Results (default 5): ").strip()
        k     = int(k) if k.isdigit() else 5

        try:
            df = search_lancedb(table, q, topic=topic, difficulty=diff, limit=k)
            if df.empty:
                print("No matches.")
            else:
                for i, row in df.reset_index().iterrows():
                    print(f"\n— Result {i+1} —")
                    for col in ["topic", "difficulty", "section", "front", "question"]:
                        if col in row and pd.notna(row[col]) and str(row[col]).strip():
                            print(f"{col.capitalize()}: {row[col]}")
                    snippet = row.get("text") or row.get("back")  # pick one
                    print("Text:", (snippet[:300] + "…") if len(snippet) > 300 else snippet)
        except Exception as e:
            print("ERROR:", e)
