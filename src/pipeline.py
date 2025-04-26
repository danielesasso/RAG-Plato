# Importazione librerie
import os
import pandas as pd
import hashlib
from datetime import datetime
import numpy as np
from io import BytesIO
import textwrap
import ollama
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry
from rich.console import Console

# Creazione Summary

# Step 1: Legge il contenuto di un file .txt, lo divide in chunk e genera un riassunto per ogni chunk

# 1.1 chunk_text: Divide il testo in chunk e restituisce le stringhe
def get_file_hash(file_path):
    """Calculate SHA-256 hash of file content"""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(4096):
            hasher.update(chunk)
    return hasher.hexdigest()

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# 1.2 generate_summary:
# * Genera un riassunto del chunk, considerando il riassunto precedente
# * Crea un prompt per Ollama per generare il riassunto
def generate_summary(prior_summary, chunk_text, summarized_words=50):
    if not chunk_text.strip():
        print("Warning: Il chunk di testo è vuoto.")
        return prior_summary

    system_prompt = textwrap.dedent(f"""\
    Il tuo obiettivo è riassumere il seguente contenuto della lezione in un testo conciso,
    evidenziando i punti chiave. 
    Fornisci SOLO il testo finale del riassunto, senza alcuna introduzione, commento o frase aggiuntiva.
    Limite di circa {summarized_words} parole.
    

    Prior Summary: {prior_summary}

    Original Section: {chunk_text}


    Summary:
    """)


    try:
        print("Invio prompt al modello...")
        result = ollama.generate(
            model='llama3.2:latest',
            prompt=system_prompt,
        )
        print("Ricevuta risposta dal modello.")
        return str(result['response'])
    except Exception as e:
        print(f"Errore durante la generazione del riassunto: {e}")
        return prior_summary


# 1.3 process_transcriptions:
# Utilizza le funzioni precedenti per leggere il contenuto di un file .txt,
# dividerlo in chunk e generare un riassunto per ogni chunk
def process_transcriptions(text_file_path, summarized_words=50, chunk_size=500):
    """
    :param text_file_path: Percorso del file di testo.
    :param summarized_words: Numero massimo di parole suggerite per il riassunto.
    :param chunk_size: Numero di parole per chunk.
    :return: Un dizionario con chiavi tipo 'chunk_1', 'chunk_2', ... e relativi riassunti come valori.
    """
    # Initialize database and table
    current_hash = get_file_hash(text_file_path)
    db = lancedb.connect("./lancedb")
    table_name = "lesson_chunks"
    
    # Create table if it doesn't exist
    if table_name not in db.table_names():
        table = db.create_table(
            table_name,
            schema=LessonChunkSchema,
            mode="overwrite"
        )
        print(f"Created new table '{table_name}' with schema")
    else:
        table = db.open_table(table_name)
    
    # Search for existing results with same file hash
    existing = table.search().where(f"file_hash = '{current_hash}'").limit(1).to_pandas()
    if not existing.empty:
        print("Trovati risultati in cache per questo file.")
        # Reconstruct summaries dictionary from cached results
        cached_chunks = table.search().where(f"file_hash = '{current_hash}' AND section LIKE 'chunk_%'").to_pandas()
        return {row['section']: row['text'] for _, row in cached_chunks.iterrows()}
    with open(text_file_path, 'r', encoding='utf-8') as f:
        text_data = f.read()

    chunks = chunk_text(text_data, chunk_size=chunk_size)
    prior_summary = "None"
    summaries = {}

    for i, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {i+1}/{len(chunks)}")
        try:
            current_summary = generate_summary(prior_summary, chunk, summarized_words)
            print(f"\n--- Summary for Chunk {i+1}/{len(chunks)} ---\n{current_summary}\n{'-'*50}\n")
            summaries[f"chunk_{i+1}"] = current_summary
            prior_summary = current_summary
        except Exception as e:
            print(f"Errore nel processare chunk_{i+1}: {e}")
            summaries[f"chunk_{i+1}"] = prior_summary

    # Save results to cache
    current_time = datetime.now().isoformat()
    data = []
    for section, text in summaries.items():
        data.append({
            "text": text,
            "index": int(section.split('_')[1]),
            "lesson_number": 1,  # Default, can be updated later
            "section": section,
            "topic": "",  # Default, can be updated later
            "score": 0.0,
            "file_hash": current_hash,
            "processed_at": current_time
        })
    
    # Add final summary to cache
    final_summary = generate_final_summary(collect_summarized_sections(summaries))
    if final_summary:
        data.append({
            "text": final_summary,
            "index": len(summaries),
            "lesson_number": 1,
            "section": "total_summary",
            "topic": "",
            "score": 0.0,
            "file_hash": current_hash,
            "processed_at": current_time
        })
    
    # Update cache
    table.add(pd.DataFrame(data))
    print(f"Risultati salvati in cache per il file {text_file_path}")
    
    return summaries


def process_transcriptions_hierarchical(
    text_file_path: str,
    summarized_words: int = 50,
    chunk_size: int = 500,
    batch_size: int = 3,
    max_levels: int = 3,
    lesson_number: int = 1,
    topic: str = ""
):
    """
    Process text file with hierarchical summarisation
    **and** produce one ultimate summary that merges the
    highest-level summaries (at double word-budget).

    Returns
    -------
    str
        The final merged summary of the entire file.
    """
    # ----------  Set-up & read file  ----------
    current_hash = get_file_hash(text_file_path)
    db = lancedb.connect("./lancedb")
    table_name = "lesson_chunks"

    if table_name not in db.table_names():
        table = db.create_table(table_name, schema=LessonChunkSchema)
        print(f"Created new table '{table_name}' with schema")
    else:
        table = db.open_table(table_name)

    with open(text_file_path, encoding="utf-8") as f:
        text_data = f.read()

    # ----------  Level-0 chunk summaries  ----------
    chunks = chunk_text(text_data, chunk_size)
    chunk_summaries = []
    now_iso = datetime.now().isoformat()

    for idx, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {idx+1}/{len(chunks)}")
        summary = generate_summary("", chunk, summarized_words)
        print(f"\n--- Summary for chunk {idx+1} ---\n{summary}\n" + "-"*60)
        chunk_summaries.append(summary)

        table.add(pd.DataFrame([{
            "text": summary,
            "index": idx,
            "lesson_number": lesson_number,
            "section": f"chunk_{idx+1}",
            "topic": topic,
            "score": 0.0,
            "file_hash": current_hash,
            "processed_at": now_iso,
            "level": 0,
            "batch_index": idx,
            "parent_batch": -1
        }]))

    # ----------  Hierarchical batching  ----------
    final_level_summaries = hierarchical_summarize(
        summaries=chunk_summaries,
        batch_size=batch_size,
        max_levels=max_levels,
        summarized_words=summarized_words,
        db_table=table,
        file_hash=current_hash,
        lesson_number=lesson_number,
        topic=topic
    )

    # ----------  FINAL merge summary  ----------
    if not final_level_summaries:
        print("No summaries generated – returning empty string.")
        return ""

    combined_text = "\n\n".join(final_level_summaries)
    print("\nGenerating FINAL merged summary…")
    final_summary = generate_summary(
        prior_summary="",
        chunk_text=combined_text,
        summarized_words=summarized_words * 2        # ← double budget here
    )
    print("\n*** FINAL SUMMARY ***\n" + final_summary)

    # save to LanceDB
    table.add(pd.DataFrame([{
        "text": final_summary,
        "index": len(chunks),          # put after last chunk
        "lesson_number": lesson_number,
        "section": "total_summary",
        "topic": topic,
        "score": 0.0,
        "file_hash": current_hash,
        "processed_at": now_iso,
        "level": max_levels + 1,       # one level above highest batches
        "batch_index": 0,
        "parent_batch": -1
    }]))

    return final_summary

# Step 2: collect_summarized_sections
# Prende tutti i chunk riassunti dal dizionario e li prepara per il riassunto finale
def collect_summarized_sections(chunk_summaries):
    """
    :param chunk_summaries: Dizionario con identificatori chunk e testi riassunti.
    :return: Lista di sezioni riassunte formattate pronte per il riassunto finale.
    """
    sorted_chunks = sorted(chunk_summaries.keys(), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return [f"Chunk ({i+1} of {len(sorted_chunks)}):\n{chunk_summaries[chunk_key]}" 
            for i, chunk_key in enumerate(sorted_chunks)]

# Step 3: Utilizzo di Ollama per generare il riassunto dell'intero file di testo e delle slide

# 3.1 generate_final_summary: Genera il riassunto finale dalle sezioni
def generate_final_summary(summarized_sections):
    """
    :param summarized_sections: Lista di sezioni riassunte formattate.
    :return: Stringa con il riassunto finale.
    """
    if not summarized_sections:
        print("Nessuna sezione riassunta da processare.")
        return ""

    summarized_sections_text = "\n\n".join(summarized_sections)

    system_prompt = textwrap.dedent(f"""
    Il tuo obiettivo principale è condensare il contenuto in un riassunto conciso,
    catturando i punti principali e i temi dalle seguenti sezioni:
    """)

    user_prompt = textwrap.dedent(f"""\
    Per creare un Riassunto Finale della lezione:

    1. **Rivedi le sezioni riassunte:** Analizza attentamente tutte le sezioni riassunte della lezione.
    2. **Identifica i temi principali:** Individua i temi educativi principali presenti nella lezione.
    3. **Consolida le informazioni:** Unisci le informazioni dalle diverse sezioni, focalizzandoti sui temi principali.
    4. **Mantieni i dettagli essenziali:** Preserva i dettagli cruciali per la comprensione della lezione.
    5. **Verifica la completezza:** Assicurati che il riassunto rappresenti accuratamente i concetti principali.

    **Sezioni riassunte:**
    {summarized_sections_text}

    Riassunto Finale:
    """)

    try:
        print("Invio prompt al modello per il riassunto finale...")
        result = ollama.generate(
            model='llama3.2:latest',
            prompt=f"{system_prompt}\n\n{user_prompt}",
        )
        print("Ricevuto riassunto finale dal modello.")
        return str(result['response'])
    except Exception as e:
        print(f"Errore durante la generazione del riassunto finale: {e}")
        return ""


def hierarchical_summarize(
    summaries,
    batch_size=3,
    max_levels=3,
    current_level=1,
    summarized_words=50,
    prior_summaries=None,
    db_table=None,
    file_hash="",
    lesson_number=1,
    topic="",
    parent_batch=-1
):
    """
    Recursively summarize a list of summaries into hierarchical summaries.

    :param summaries: List of strings (summaries)
    :param batch_size: Number of summaries per batch
    :param max_levels: Max hierarchy depth
    :param current_level: Current hierarchy level (starts from 1)
    :param summarized_words: Target words per summary
    :param prior_summaries: List of prior summaries for coherence (optional)
    :param db_table: LanceDB table to store summaries (optional)
    :param file_hash: File hash for metadata
    :param lesson_number: Lesson number metadata
    :param topic: Topic metadata
    :param parent_batch: Parent batch index
    :return: List of higher-level summaries
    """
    if prior_summaries is None:
        prior_summaries = [""] * len(summaries)

    if len(summaries) <= 1 or current_level > max_levels:
        return summaries

    new_summaries = []
    for batch_idx, i in enumerate(range(0, len(summaries), batch_size)):
        batch = summaries[i:i + batch_size]
        prior_batch = prior_summaries[i:i + batch_size]
        batch_text = "\n\n".join(batch)
        prior_context = "\n\n".join(prior_batch)

        batch_summary = generate_summary(
            prior_context,
            batch_text,
            summarized_words
        )
        print(f"\n=== Level {current_level} - Batch {batch_idx+1}/{len(range(0, len(summaries), batch_size))} Summary ===\n{batch_summary}\n{'='*60}\n")
        new_summaries.append(batch_summary)

        # Save batch summary to LanceDB if table provided
        if db_table is not None:
            data = {
                "text": batch_summary,
                "index": batch_idx,
                "lesson_number": lesson_number,
                "section": f"level_{current_level}_batch_{batch_idx}",
                "topic": topic,
                "score": 0.0,
                "file_hash": file_hash,
                "processed_at": datetime.now().isoformat(),
                "level": current_level,
                "batch_index": batch_idx,
                "parent_batch": parent_batch
            }
            db_table.add(pd.DataFrame([data]))

    # Recursive call
    return hierarchical_summarize(
        new_summaries,
        batch_size=batch_size,
        max_levels=max_levels,
        current_level=current_level + 1,
        summarized_words=summarized_words,
        prior_summaries=new_summaries,
        db_table=db_table,
        file_hash=file_hash,
        lesson_number=lesson_number,
        topic=topic,
        parent_batch=0  # root for next level
    )

# Pipeline completa per elaborazione testuale
def run_pipeline(text_file_path):
    """
    Esegue l'intera pipeline di riassunto.
    
    :param text_file_path: Percorso del file di testo da riassumere
    """
    # Step 1: Processa le trascrizioni
    chunk_summaries = process_transcriptions(text_file_path)
    
    # Step 2: Raccogli le sezioni riassunte
    summarized_sections = collect_summarized_sections(chunk_summaries)
    
    # Step 3: Genera il riassunto finale
    final_summary = generate_final_summary(summarized_sections)
    
    if final_summary:
        print("\nRiassunto Finale:\n")
        print(final_summary)
    
    return final_summary

# Integrazione con DataFrame
def create_summary_dataframe(chunk_summaries_dict, final_summary, lesson_number, topic):
    """
    Crea un DataFrame pandas con i riassunti.
    
    :param chunk_summaries_dict: Dizionario con percorsi chunk e testi riassunti
    :param final_summary: Testo riassuntivo finale
    :param lesson_number: Numero della lezione
    :param topic: Argomento della lezione
    :return: DataFrame pandas con i chunk riassunti
    """
    data = [
        {
            "text": summary_text,
            "index": index,
            "lesson_number": lesson_number,
            "section": f"chunk_{index + 1}",
            "topic": topic,
            "score": 0.0
        }
        for index, (_, summary_text) in enumerate(chunk_summaries_dict.items())
    ]
    
    data.append({
        "text": final_summary,
        "index": len(chunk_summaries_dict),
        "lesson_number": lesson_number,
        "section": "total_summary",
        "topic": topic,
        "score": 0.0
    })
    
    return pd.DataFrame(data)

# Configurazione iniziale LanceDB e schema
registry = EmbeddingFunctionRegistry.get_instance()
embedder = registry.get("ollama").create(name="mxbai-embed-large")

class LessonChunkSchema(LanceModel):
    text: str = embedder.SourceField()
    vector: Vector(embedder.ndims()) = embedder.VectorField()
    index: int
    lesson_number: int
    section: str
    topic: str
    score: float = 0.0
    file_hash: str = ""  # SHA-256 hash of source file
    processed_at: str = ""  # ISO format timestamp
    level: int = 0  # Hierarchy level: 0=chunk, 1=batch, etc.
    batch_index: int = 0  # Index within its level
    parent_batch: int = -1  # Parent batch index (-1 if root)

# Funzione di inizializzazione del database
def init_db():
    db = lancedb.connect("./lancedb")
    table_name = "lesson_chunks"
    
    if table_name not in db.table_names():
        table = db.create_table(table_name, schema=LessonChunkSchema)
        print(f"Tabella '{table_name}' creata con schema iniziale")
    else:
        table = db.open_table(table_name)
        print(f"Tabella '{table_name}' già esistente")
    return table

# Esempio di utilizzo
if __name__ == "__main__":
    # Configurazione
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    text_file = os.path.join(base_dir, "cleaned_speech_text.txt")
    
    if not os.path.exists(text_file):
        raise FileNotFoundError(f"File di testo non trovato: {text_file}\n"
                              f"Assicurati che il file esista nella directory src/")
    lesson_num = 1
    lesson_topic = "Voice Biometrics Processing"
    
    # Esegui pipeline
    final_summary = run_pipeline(text_file)
    
    # Crea DataFrame
    chunk_summaries = process_transcriptions(text_file)
    df = create_summary_dataframe(chunk_summaries, final_summary, lesson_num, lesson_topic)
    print("\nDataFrame creato:")
    print(df)
    
    # Inizializza database
    table = init_db()
    
    # Aggiunta dati di esempio (commentato)
    # table.add(df)
    print("\nPipeline completata con successo.")