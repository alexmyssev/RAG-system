from knowledge_base import document_chunking
from vector_storage import VectorStore
from llm import Model
from authorization import Authorization
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from instructions_detector import Detector
from fastapi import FastAPI
from pydantic import BaseModel


class Registration(BaseModel):
    username: str
    password: str

class UserRequest(BaseModel):
    query: str

app = FastAPI()
security = HTTPBasic()

@app.get("/health")
def health():
    return {"status": "Server is running"}

def initialize_RAG():
    doc_folder = "docs"
    vectore_store = VectorStore()
    if vectore_store.collection.count() == 0:
        chunks = document_chunking(doc_folder)
        if not chunks:
            return None
        documents = [c["document"] for c in chunks]
        metadata = [c["metadatas"] for c in chunks]
        vectore_store.add_document(documents, metadata)
        print(f"Added {len(documents)} texts and {len(metadata)} metadata")
    return Model(vectore_store)



@app.on_event("startup")
def startup_event():
    global rag, auth, instructions_detector
    rag = initialize_RAG()
    auth = Authorization()
    instructions_detector = Detector()
    if rag is None:
        print("RAG is not initialized")
def get_user_role(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    role = auth.FA_verify(credentials.username, credentials.password)
    if role is None:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return role

@app.post("/register")
def register(request: Registration):
    try:
        role = auth.FA_register(request.username, request.password)
        return {"username": request.username, "role": role}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Registration failed. {e}")


@app.post("/query")
def query(request: UserRequest, role: str = Depends(get_user_role)):
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG is not initialized")
    if instructions_detector.is_malicious(request.query):
        return {"answer": "Malicious query!"}
    full_access = (role == "admin")
    answer = rag.generate(request.query, full_access)
    return {"answer": answer}

def main():

    rag = initialize_RAG()
    instructions_detector = Detector()
    if rag is None:
        print(f"No documents in the folder")
        return

    role = Authorization().sign_in()

    if role == "admin":
       full_access = True
    else:
        full_access = False


    while True:
        try:

            print(f"Enter 'quit' to stop the program")
            query = input("Enter query: ").strip()
            if query.lower() == "quit":
                break

            if not query:
                continue

            if instructions_detector.has_malicious_fragment(query) or instructions_detector.is_malicious(query):
                print(f"Malicious prompt!")
                continue

            result = rag.generate(query, full_access)
            print(f"\nAnswer: {result}")

        except KeyboardInterrupt:
            break

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>RAG Demo</title>
        <style>
            body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; }
            .block { border: 1px solid #ccc; border-radius: 8px; padding: 16px; margin-bottom: 20px; }
            input, textarea { width: 100%; padding: 7px; margin: 6px 0; box-sizing: border-box; }
            button { padding: 8px 16px; cursor: pointer; }
            pre { background: #f4f4f4; padding: 10px; border-radius: 6px; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <h1>RAG API — тестовая панель</h1>
        <h1>RAG API — тестовая панель</h1>

        <div class="block">
            <h3>1. Проверка работы сервера</h3>
            <button onclick="checkHealth()">Проверить статус</button>
            <pre id="healthResult"></pre>
        </div>

        <div class="block">
            <h3>2. Регистрация</h3>
            <input id="regUsername" placeholder="Username">
            <input id="regPassword" type="password" placeholder="Password">
            <button onclick="register()">Зарегистрироваться</button>
            <pre id="regResult"></pre>
        </div>

        <div class="block">
            <h3>3. Запрос к RAG</h3>
            <input id="authUsername" placeholder="Username">
            <input id="authPassword" type="password" placeholder="Password">
            <textarea id="queryText" placeholder="Введите ваш вопрос..." rows="3"></textarea>
            <button onclick="sendQuery()">Отправить запрос</button>
            <pre id="queryResult"></pre>
        </div>

        <script>
            async function checkHealth() {
                const res = await fetch('/health');
                const data = await res.json();
                if (data.status){
                    document.getElementById('healthResult').textContent = data.status;
                }
                else if (data.detail){
                    document.getElementById('healthResult').textContent = "Error: " + data.detail;
                }
            }

            async function register() {
                const username = document.getElementById('regUsername').value;
                const password = document.getElementById('regPassword').value;
                const res = await fetch('/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                const data = await res.json();
                if (data.role){
                    document.getElementById('regResult').textContent = "User:" + data.username + "   role:" + data.role;
                }
                else if (data.detail){
                    document.getElementById('regResult').textContent = "Error: " + data.detail;
                }
            }

            async function sendQuery() {
                const username = document.getElementById('authUsername').value;
                const password = document.getElementById('authPassword').value;
                const query = document.getElementById('queryText').value;

                const res = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Basic ' + btoa(username + ':' + password)
                    },
                    body: JSON.stringify({ query })
                });
                const data = await res.json();
                if (data.answer){
                    document.getElementById('queryResult').textContent = data.answer;
                }
                else if (data.detail){
                    document.getElementById('queryResult').textContent = "Error: " + data.detail;
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    #main()




