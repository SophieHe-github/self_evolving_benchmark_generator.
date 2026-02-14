import os
import json
import time
import re
import csv
import numpy as np
import ast  
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI  
from fastembed import TextEmbedding

load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("OPENAI_API_KEY") 
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_ITERATIONS = 100  # Set to None for infinite run
SIMILARITY_THRESHOLD = 0.85

if not API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env file.")

class BenchmarkState:
    def __init__(self, alpha: float = 0.3):
        self.ema_score: float = 0.5
        self.alpha: float = alpha
        self.difficulty: int = 1
        self.embedding_history: List[List[float]] = [] 
        self.history_topics: List[str] = []
        self.iteration: int = 0

    def update(self, score: float):
        self.ema_score = (self.alpha * score) + ((1 - self.alpha) * self.ema_score)
        if self.ema_score > 0.8 and self.difficulty < 10:
            self.difficulty += 1
            print(f"📈 Strong Performance. Difficulty -> {self.difficulty}")
        elif self.ema_score < 0.4 and self.difficulty > 1:
            self.difficulty -= 1
            print(f"📉 Weak Performance. Difficulty -> {self.difficulty}")

class EvolvingBenchmark:
    def __init__(self):
        # Initialize OpenAI Client
        self.client = OpenAI(api_key=API_KEY)
        
        # Initialize Local Embedding Model
        print("⏳ Loading local embedding model (BAAI/bge-small-en-v1.5)...")
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        print("✅ Embedding model loaded.")

        self.state = BenchmarkState()
        self.log_filename = "benchmark_results_vectors.csv"
        
        # Load Existing History (Persistence)
        self._load_history()

    def _load_history(self):
        """Reloads past questions/vectors so we don't repeat even after restart."""
        if not os.path.exists(self.log_filename):
            # Create new file with headers
            with open(self.log_filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Iteration", "Status", "Topic", "Difficulty", "Question", "Answer", "Score", "EMA_Score", "Embedding_Vector"])
            print(f"💾 Created new log file: {self.log_filename}")
            return

        print(f"📂 Loading history from: {self.log_filename}")
        try:
            with open(self.log_filename, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    # Restore Topics
                    if row.get("Topic") and row["Topic"] != "N/A":
                        self.state.history_topics.append(row["Topic"])
                    
                    # Restore Vectors
                    vec_str = row.get("Embedding_Vector", "[]")
                    if vec_str and vec_str != "[]":
                        try:
                            # Parse string "[0.1, 0.2]" back into a list of floats
                            vector = json.loads(vec_str)
                            if isinstance(vector, list) and len(vector) > 0:
                                self.state.embedding_history.append(vector)
                                count += 1
                        except:
                            pass # Skip malformed rows
                
                # Restore last Difficulty/EMA if available
                if count > 0:
                    self.state.iteration = int(row.get("Iteration", 0))
                    self.state.difficulty = int(row.get("Difficulty", 1))
                    self.state.ema_score = float(row.get("EMA_Score", 0.5))

            print(f"✅ Restored {count} historical questions. Resuming at Iteration {self.state.iteration + 1}.")
        except Exception as e:
            print(f"⚠️ Failed to load history: {e}")

    def _call_llm(self, system_prompt: str, user_prompt: str, retries: int = 3) -> str:
        for attempt in range(retries):
            try:
                # OpenAI call
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=1000 
                )
                content = response.choices[0].message.content
                if content: return content
            except Exception as e:
                print(f"  [API Warn] Chat Attempt {attempt+1} failed: {e}")
                time.sleep(1)
        return ""

    def _get_embedding(self, text: str) -> List[float]:
        try:
            embeddings = list(self.embed_model.embed([text]))
            return embeddings[0].tolist() 
        except Exception as e:
            print(f"  [Embedding Error] {e}")
            return []

    def _check_similarity(self, new_vector: List[float]) -> bool:
        if not self.state.embedding_history: return False 

        new_vec = np.array(new_vector)
        history_matrix = np.array(self.state.embedding_history)
        
        # Calculate Dot Product
        similarities = np.dot(history_matrix, new_vec)
        max_sim = np.max(similarities)
        
        if max_sim > SIMILARITY_THRESHOLD:
            print(f"  [Novelty Guard] Rejected! Similarity Score: {max_sim:.4f}")
            return True 
        
        return False 

    def _extract_json(self, text: str) -> Dict[str, Any]:
        if not text: return {}
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except: pass
        return {}

    def generate_novel_question(self) -> Optional[Dict[str, Any]]:
        # Use random sample of history to keep context small
        import random
        unique_topics = list(set(self.state.history_topics))
        if len(unique_topics) > 10:
            forbidden = ", ".join(random.sample(unique_topics, 10))
        else:
            forbidden = ", ".join(unique_topics)
        
        system_prompt = "You are an adversarial benchmark generator. Output MUST be valid JSON."
        prompt = (
            f"Generate a question. Difficulty: {self.state.difficulty}/10. "
            f"Avoid topics: {forbidden}. "
            f"JSON format: {{'topic': '...', 'question': '...', 'expected_criteria': '...'}}"
        )

        for attempt in range(3):
            # Generate Text
            raw = self._call_llm(system_prompt, prompt)
            data = self._extract_json(raw)
            
            if not data or 'question' not in data:
                continue
            
            # Compute Embedding
            vector = self._get_embedding(data['question'])
            
            if not vector:
                return {**data, "embedding": []}

            # Check against History
            if self._check_similarity(vector):
                prompt += " That was too similar to a previous question. Be more creative!"
                continue
            
            return {**data, "embedding": vector}
            
        return None

    def evaluate_answer(self, question: str, answer: str, criteria: str) -> float:
        system_prompt = "You are an impartial AI Judge. Return ONLY raw JSON."
        prompt = (
            f"Question: {question}\nAnswer: {answer}\nCriteria: {criteria}\n"
            f"Return JSON: {{'score': 0.0 to 1.0}}"
        )
        return float(self._extract_json(self._call_llm(system_prompt, prompt)).get("score", 0.0))

    def run_step(self):
        self.state.iteration += 1
        print(f"\n--- Iteration {self.state.iteration} ---")

        q_data = self.generate_novel_question()
        if not q_data:
            print("❌ Generator Failed.")
            self._log_to_csv("GEN_FAIL", "N/A", "N/A", "N/A", 0.0, [])
            return

        topic = q_data.get('topic', 'General')
        question = q_data.get('question')
        vector = q_data.get('embedding', [])

        # Update History (In-Memory)
        self.state.history_topics.append(topic)
        if vector: self.state.embedding_history.append(vector)

        print(f"📝 Topic: {topic} (Diff: {self.state.difficulty})")

        # Solve & Evaluate
        answer = self._call_llm("You are a helpful assistant.", question)
        print(f"🤖 Answered ({len(answer)} chars).")
        
        score = self.evaluate_answer(question, answer, q_data.get('expected_criteria'))
        self.state.update(score)
        print(f"⚖️  Score: {score} (EMA: {self.state.ema_score:.2f})")

        self._log_to_csv("SUCCESS", topic, question, answer, score, vector)

    def _log_to_csv(self, status, topic, question, answer, score, vector):
        clean_q = question.replace('\n', ' ') if question else ""
        clean_a = answer.replace('\n', ' ') if answer else ""
        
        # Dump the FULL vector to JSON string.
        vec_str = json.dumps(vector) if vector else "[]"
        
        with open(self.log_filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.state.iteration, 
                status, 
                topic, 
                self.state.difficulty, 
                clean_q[:2000], # Limit text length for Excel safety
                clean_a[:2000], 
                score, 
                self.state.ema_score, 
                vec_str # Full vector saved here
            ])

if __name__ == "__main__":
    benchmark = EvolvingBenchmark()
    try:
        # Loop continues until MAX_ITERATIONS is reached
        while MAX_ITERATIONS is None or benchmark.state.iteration < MAX_ITERATIONS:
            benchmark.run_step()
            time.sleep(1)
        print(f"\n✅ Completed {MAX_ITERATIONS} iterations.")
    except KeyboardInterrupt:
        print("\nStopped by user.")