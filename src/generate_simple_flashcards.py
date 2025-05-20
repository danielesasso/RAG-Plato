import os
import pandas as pd
import ollama
import lancedb
from datetime import datetime
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry

# Initialize LanceDB connection
db = lancedb.connect("./lancedb")

#registry
registry = EmbeddingFunctionRegistry.get_instance()
embedder = registry.get("ollama").create(name="mxbai-embed-large") 

class SimpleFlashcardSchema(LanceModel):
    front: str = embedder.SourceField() # The question 
    vector: Vector(embedder.ndims()) = embedder.VectorField() # Optional vector
    back: str # The answer 
    lesson_number: int
    topic: str
    difficulty: str
    source_chunk_id: int
    generated_at: str = ""

def init_simple_flashcard_table():
    table_name = "simple_flashcards"
    if table_name not in db.table_names():
        table = db.create_table(table_name, schema=SimpleFlashcardSchema)
        print(f"Created simple flashcard table '{table_name}'")
    else:
        table = db.open_table(table_name)
        print(f"Simple flashcard table '{table_name}' already exists")
    return table

def generate_simple_flashcard_prompt(chunk_text):
    prompt = f"""
Dato il seguente estratto di una lezione, genera UNA flashcard semplice e concisa. La flashcard deve avere un lato "Fronte" (una domanda breve o un termine chiave) e un lato "Retro" (la risposta diretta o la definizione). Formatta la risposta ESATTAMENTE come segue:

**Fronte:** [Testo del fronte qui]
**Retro:** [Testo del retro qui]

Estratto:
\"\"\"
{chunk_text}
\"\"\"

**Fronte:**
**Retro:**
"""
    return prompt

def parse_simple_flashcard_response(response):
    front = ""
    back = ""
    try:
        if "**Fronte:**" in response and "**Retro:**" in response:
            front_part = response.split("**Fronte:**", 1)[1].split("**Retro:**", 1)[0].strip()
            back_part = response.split("**Retro:**", 1)[1].strip()
            front = front_part
            back = back_part
        else:
            print("Warning: Could not parse simple flashcard response format.")
    except Exception as e:
        print(f"Error parsing simple flashcard response: {e}")
    return front, back

def generate_simple_flashcard_for_chunk(chunk_text, lesson_number, topic, chunk_id, difficulty=""):
    prompt = generate_simple_flashcard_prompt(chunk_text)
    front, back = "", ""
    try:
        result = ollama.generate(
            model='llama3.2:latest', 
            prompt=prompt,
        )
        response = result['response']
        print(f"\n--- Ollama raw response for chunk {chunk_id} ---\n{response}\n--- End response ---\n")
        front, back = parse_simple_flashcard_response(response)

    except Exception as e:
        print(f"Error generating simple flashcard for chunk {chunk_id}: {e}")

    return {
        "front": front,
        "back": back,
        "lesson_number": lesson_number,
        "topic": topic,
        "difficulty": difficulty,
        "source_chunk_id": chunk_id,
        "generated_at": datetime.now().isoformat()
    }

def main():
    # Open existing lesson_chunks table
    lesson_table = db.open_table("lesson_chunks")
    # Query all summaries 
    df = lesson_table.search().where("section LIKE 'chunk_%'").to_pandas()

    flashcards = []
    for _, row in df.iterrows():
        chunk_text = row['text']
        lesson_number = row.get('lesson_number', 1)
        topic = row.get('topic', '')
        chunk_id = row.get('index', 0)
        difficulty = ""  # Placeholder
        card_data = generate_simple_flashcard_for_chunk(chunk_text, lesson_number, topic, chunk_id, difficulty)
        # Ensure essential parts were parsed
        if card_data['front'] and card_data['back']:
            flashcards.append(card_data)
            print(card_data)

    if flashcards:
        flashcard_table = init_simple_flashcard_table()
        flashcard_table.add(pd.DataFrame(flashcards))
        print(f"Saved {len(flashcards)} simple flashcards to LanceDB.")
    else:
        print("No valid simple flashcards generated.")

if __name__ == "__main__":
    main()
