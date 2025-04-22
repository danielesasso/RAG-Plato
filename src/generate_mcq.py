import os
import pandas as pd
import ollama
import lancedb
from datetime import datetime
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry

# Initialize LanceDB connection
db = lancedb.connect("./lancedb")

# Embedding function for MCQs (optional, for semantic search)
registry = EmbeddingFunctionRegistry.get_instance()
embedder = registry.get("ollama").create(name="mxbai-embed-large")

class MCQSchema(LanceModel):
    question: str = embedder.SourceField() # The question stem
    vector: Vector(embedder.ndims()) = embedder.VectorField()
    choices: list[str] # List of choices (e.g., ["A) ...", "B) ..."])
    answer: str # The correct choice (e.g., "B")
    lesson_number: int
    topic: str
    difficulty: str
    source_chunk_id: int
    question_type: str = "multiple_choice" # Type identifier
    generated_at: str = ""

def init_mcq_table():
    table_name = "mcq_questions"
    if table_name not in db.table_names():
        table = db.create_table(table_name, schema=MCQSchema)
        print(f"Created MCQ table '{table_name}'")
    else:
        table = db.open_table(table_name)
        print(f"MCQ table '{table_name}' already exists")

    # Insert placeholder if table is empty
    if table.search().limit(1).to_pandas().empty:
        placeholder_mcq = {
            "question": "Placeholder question",
            "choices": ["A) Placeholder A", "B) Placeholder B", "C) Placeholder C", "D) Placeholder D"],
            "answer": "A",
            "lesson_number": -1,
            "topic": "Placeholder",
            "difficulty": "",
            "source_chunk_id": -1,
            "question_type": "multiple_choice",
            "generated_at": datetime.now().isoformat()
        }
        df_placeholder = pd.DataFrame([placeholder_mcq])
        table.add(df_placeholder)
        print("Inserted placeholder MCQ to avoid cold start.")

    return table

def generate_mcq_prompt(chunk_text):
    prompt = f"""
Dato il seguente estratto di una lezione, genera UNA domanda a scelta multipla (con opzioni A, B, C, D) per verificare la comprensione dei concetti chiave. Includi la domanda, le opzioni di risposta e indica chiaramente la risposta corretta. Formatta la risposta ESATTAMENTE come segue:

Estratto:
\"\"\"
{chunk_text}
\"\"\"

**Domanda:** [Testo della domanda qui]

**Opzioni:**
A) [Testo opzione A]
B) [Testo opzione B]
C) [Testo opzione C]
D) [Testo opzione D]

**Risposta corretta:** [Lettera della risposta corretta, es. B]
"""
    return prompt

def parse_mcq_response(response):
    question = ""
    choices = []
    answer = ""
    try:
        if "**Domanda:**" in response and "**Opzioni:**" in response and "**Risposta corretta:**" in response:
            question_part = response.split("**Domanda:**", 1)[1].split("**Opzioni:**", 1)[0].strip()
            options_part = response.split("**Opzioni:**", 1)[1].split("**Risposta corretta:**", 1)[0].strip()
            answer_part = response.split("**Risposta corretta:**", 1)[1].strip()

            question = question_part
            # Extract choices, assuming format like "A) Text"
            choices = [opt.strip() for opt in options_part.split('\n') if opt.strip()]
            # Keep only the letter for the answer
            answer = answer_part.split(')')[0].strip() # Get just 'A', 'B', etc.
        else:
            print("Warning: Could not parse MCQ response format.")
    except Exception as e:
        print(f"Error parsing MCQ response: {e}")

    return question, choices, answer


def generate_mcq_for_chunk(chunk_text, lesson_number, topic, chunk_id, difficulty=""):
    prompt = generate_mcq_prompt(chunk_text)
    question, choices, answer = "", [], ""
    try:
        result = ollama.generate(
            model='llama3.2:latest',
            prompt=prompt,
        )
        response = result['response']
        print(f"\n--- Ollama raw response for chunk {chunk_id} ---\n{response}\n--- End response ---\n")
        question, choices, answer = parse_mcq_response(response)

    except Exception as e:
        print(f"Error generating MCQ for chunk {chunk_id}: {e}")

    return {
        "question": question,
        "choices": choices,
        "answer": answer,
        "lesson_number": lesson_number,
        "topic": topic,
        "difficulty": difficulty,
        "source_chunk_id": chunk_id,
        "question_type": "multiple_choice",
        "generated_at": datetime.now().isoformat()
    }

def main():
    # Open existing lesson_chunks table
    lesson_table = db.open_table("lesson_chunks")
    # Query all summaries (customize filter as needed)
    df = lesson_table.search().where("section LIKE 'chunk_%'").to_pandas()

    mcqs = []
    for _, row in df.iterrows():
        chunk_text = row['text']
        lesson_number = row.get('lesson_number', 1)
        topic = row.get('topic', '')
        chunk_id = row.get('index', 0)
        difficulty = ""  # Placeholder, can be set via metadata or LLM
        mcq_data = generate_mcq_for_chunk(chunk_text, lesson_number, topic, chunk_id, difficulty)
        # Ensure essential parts were parsed
        if mcq_data['question'] and mcq_data['choices'] and mcq_data['answer']:
            mcqs.append(mcq_data)

    if mcqs:
        mcq_table = init_mcq_table()
        # Ensure data types match schema, especially list for choices
        df_mcqs = pd.DataFrame(mcqs)
        # Convert choices column to list if it's not already (might be needed depending on pandas version)
        # df_mcqs['choices'] = df_mcqs['choices'].apply(lambda x: list(x) if not isinstance(x, list) else x)
        mcq_table.add(df_mcqs)
        print(f"Saved {len(mcqs)} MCQs to LanceDB.")
    else:
        print("No valid MCQs generated.")

if __name__ == "__main__":
    main()
    