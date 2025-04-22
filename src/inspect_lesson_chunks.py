import lancedb

db = lancedb.connect("./lancedb")
table = db.open_table("lesson_chunks")

df = table.search().to_pandas()

print(f"Total rows: {len(df)}")
print(df[['section', 'text']].head(10))
