import subprocess

def start():
    subprocess.Popen(["uvicorn", "main:app", "--reload"])
    subprocess.Popen(["docker", "run", "-d", "--name", "qdrant", "-p", "6333:6333", "qdrant/qdrant"])
    subprocess.Popen(["streamlit", "run", "interface.py"])

if __name__ == "__main__":
    start()