from src.pipeline import process_transcriptions_hierarchical
import lancedb
import pandas as pd

def test_hierarchical_pipeline(
    file_path,
    chunk_size=250,
    batch_size=2,
    max_levels=3,
    summarized_words=40,
    lesson_number=99,
    topic="Test Sample"
):
    """
    Run hierarchical summarization pipeline on a text file and print results.

    :param file_path: Path to input .txt file
    :param chunk_size: Words per chunk
    :param batch_size: Summaries per batch
    :param max_levels: Max hierarchy levels
    :param summarized_words: Words per summary
    :param lesson_number: Lesson number metadata
    :param topic: Topic metadata
    """
    print(f"Running hierarchical summarization on {file_path}")
    final_summary = process_transcriptions_hierarchical(
        file_path,
        summarized_words=summarized_words,
        chunk_size=chunk_size,
        batch_size=batch_size,
        max_levels=max_levels,
        lesson_number=lesson_number,
        topic=topic
    )
    print("\n=== FINAL SUMMARY ===\n")
    print(final_summary)

    # Optional: list all summaries stored in LanceDB
    try:
        db = lancedb.connect("./lancedb")
        table = db.open_table("lesson_chunks")
        df = table.to_pandas()
        # Filter by latest file hash (assuming last added corresponds to this run)
        if not df.empty:
            latest_hash = df['file_hash'].iloc[-1]
            df = df[df['file_hash'] == latest_hash]
            print("\n=== Stored Summaries Metadata ===\n")
            print(df[['section', 'level', 'batch_index', 'parent_batch', 'text']])
        else:
            print("No summaries found in LanceDB.")
    except Exception as e:
        print(f"Error accessing LanceDB: {e}")

if __name__ == "__main__":
    # Modify these parameters as needed
    test_hierarchical_pipeline(
        file_path="test_text.txt",
        chunk_size=250,
        batch_size=2,
        max_levels=3,
        summarized_words=40,
        lesson_number=99,
        topic="Test Sample"
    )
