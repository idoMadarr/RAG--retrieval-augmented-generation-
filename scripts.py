from dotenv import load_dotenv
import subprocess
import time
import os

load_dotenv()

LOCAL_RAG = os.getenv("LOCAL_RAG", "False").lower() == "true"


def start():
    try:
        print("Starting Qdrant vector database (Docker)")
        subprocess.run(["docker", "rm", "-f", "qdrant"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.Popen(["docker", "run", "-d", "--name", "qdrant", "-p", "6333:6333", "qdrant/qdrant"])
        time.sleep(5)

        print(f"🧠 Starting {os.getenv("LLM")} model...")
        if LOCAL_RAG:
            subprocess.Popen(["ollama", "run", os.getenv("LLM")])
            time.sleep(5)

        print("Starting FastAPI (Uvicorn)...")
        subprocess.Popen(["uvicorn", "main:app", "--reload"])

        time.sleep(1)
        print("Starting Streamlit UI...")
        subprocess.Popen(["streamlit", "run", "interface.py"])
    except Exception as e:
        print(f"❌ Error starting services: {e}")


if __name__ == "__main__":
    start()