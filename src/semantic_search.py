import lancedb
from lancedb.embeddings import EmbeddingFunctionRegistry

# Connect to DB and open lesson_chunks table
db = lancedb.connect("./lancedb")
table_name = "lesson_chunks"

if table_name not in db.table_names():
    print(f"Error: Table '{table_name}' not found in LanceDB.")
    exit()

table = db.open_table(table_name)

# Initialize embedder (same as used for lesson chunks)
registry = EmbeddingFunctionRegistry.get_instance()
embedder = registry.get("ollama").create(name="mxbai-embed-large")

print("Interactive Semantic Search Tool")
print("--------------------------------")

while True:
    query_text = input("\nEnter your search query (or 'exit'): ")
    if query_text.lower() == 'exit':
        break

    # Optional: prompt for filters
    topic_filter = input("Filter by topic (or leave blank): ").strip()
    difficulty_filter = input("Filter by difficulty (or leave blank): ").strip()
    limit = input("Number of results (default 5): ").strip()
    limit = int(limit) if limit.isdigit() else 5

    try:
        # Pass raw query text directly to search; LanceDB handles embedding
        search = table.search(query_text)

        # Apply filters
        where_clauses = []
        if topic_filter:
            where_clauses.append(f"topic = '{topic_filter}'")
        if difficulty_filter:
            where_clauses.append(f"difficulty = '{difficulty_filter}'")
        if where_clauses:
            search = search.where(" AND ".join(where_clauses))

        results = search.limit(limit).to_pandas()

        if results.empty:
            print("No results found.")
        else:
            print(f"\n--- Found {len(results)} results ---")
            for idx, row in results.iterrows():
                print(f"\n--- Result {idx+1} ---")
                print(f"Topic: {row.get('topic','')}")
                print(f"Difficulty: {row.get('difficulty','')}")
                print(f"Speaker: {row.get('speaker','')}")
                print(f"Section: {row.get('section','')}")
                print(f"Text:\n{row['text']}\n")

    except Exception as e:
        print(f"An error occurred: {e}")

print("\nExiting search tool.")
