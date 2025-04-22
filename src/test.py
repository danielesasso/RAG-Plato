# query_db.py
import lancedb
import pandas as pd
from pipeline import LessonChunkSchema  # Import the schema

# Connect to the LanceDB database
db = lancedb.connect("./lancedb")

# Open the table
table_name = "lesson_chunks"
try:
    table = db.open_table(table_name)
except lancedb.exceptions.TableNotFoundError:
    print(f"Error: Table '{table_name}' not found in the database.")
    exit()

# Example: Get the total summary
total_summary_df = table.search().where("section = 'total_summary'").limit(1).to_pandas()
if not total_summary_df.empty:
    print("Total Summary:")
    print(total_summary_df['text'].iloc[0])
else:
    print("No total summary found.")

# Example: Get all chunk summaries
chunk_summaries_df = table.search().where("section LIKE 'chunk_%'").to_pandas()
if not chunk_summaries_df.empty:
    print("\nChunk Summaries:")
    for index, row in chunk_summaries_df.iterrows():
        print(f"Chunk {row['index']}: {row['text']}")
else:
    print("No chunk summaries found.")

# Example: Get all entries for a specific file hash (if you know it)
# Assuming you have a way to know the file hash
# file_hash_to_query = "your_file_hash_here"
# file_data_df = table.search().where(f"file_hash = '{file_hash_to_query}'").to_pandas()
# if not file_data_df.empty:
#     print(f"\nData for file hash '{file_hash_to_query}':")
#     print(file_data_df)
# else:
#     print(f"No data found for file hash '{file_hash_to_query}'.")