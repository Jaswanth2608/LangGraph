
import os, sys, io, traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


# LLM
app = FastAPI()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)


# State
class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


class TaskRequest(BaseModel):
    task: str


# Tools
@tool
def run_python_code(code: str) -> str:
    """Execute Python code."""
    code = code.replace("```python", "").replace("```", "").strip()

    old = sys.stdout
    sys.stdout = io.StringIO()

    try:
        exec(code, {}, {})
        result = sys.stdout.getvalue()
    except Exception:
        result = traceback.format_exc()
    finally:
        sys.stdout = old

    return result.strip() or "Success (no terminal output)"


@tool
def generate_test_cases(task: str) -> str:
    """Generate test cases."""
    prompt = f"""
    You are a QA Engineer.
    Generate 3 to 5 test cases for:
    {task}
    Include normal and edge cases.
    """
    return str(llm.invoke(prompt).content)


# Nodes
def task_input(state):
    print("[TASK INPUT]")
    return {"next_step": "developer"}


def developer(state):
    print("[DEVELOPER]")

    task = state["messages"][-1].content

    prompt = (
        f"Write a clean Python program for: {task}. "
        "Return only code."
    )

    response = llm.invoke(prompt)
    code = str(response.content)

    print(code)

    return {
        "code": code,
        "next_step": "tester"
    }


def tester(state):
    print("[TESTER]")

    task = state["messages"][-1].content

    tests = generate_test_cases.invoke(task)

    output = run_python_code.invoke({
        "code": state["code"]
    })

    report = (
        f"EXECUTION OUTPUT:\n{output}\n\n"
        f"TEST SCENARIOS:\n{tests}"
    )

    return {
        "report": report,
        "next_step": "manager"
    }


def manager(state):
    print("[MANAGER]")
    return {"next_step": "archiver"}


def archiver(state):
    print("[ARCHIVER]")
    return {"next_step": "exit"}


# Graph
graph = StateGraph(CrewState)

graph.add_node("task_input", task_input)
graph.add_node("developer", developer)
graph.add_node("tester", tester)
graph.add_node("manager", manager)
graph.add_node("archiver", archiver)

graph.add_edge(START, "task_input")
graph.add_edge("task_input", "developer")
graph.add_edge("developer", "tester")
graph.add_edge("tester", "manager")
graph.add_edge("manager", "archiver")
graph.add_edge("archiver", END)

workflow = graph.compile()


# Website
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>AI Developer & Tester</title>
        <style>
            body {
                font-family: Arial;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: #f4f4f4;
            }
            textarea {
                width: 100%;
                height: 120px;
                padding: 10px;
            }
            button {
                width: 100%;
                padding: 12px;
                margin-top: 10px;
                background: #2563eb;
                color: white;
                border: 0;
                cursor: pointer;
            }
            pre {
                background: white;
                padding: 15px;
                white-space: pre-wrap;
            }
        </style>
    </head>

    <body>

        <h1>🤖 AI Developer & Tester</h1>

        <textarea id="task"
        placeholder="Enter your coding task..."></textarea>

        <button onclick="run()">Run Workflow</button>

        <h2>Generated Code</h2>
        <pre id="code"></pre>

        <h2>Test Report</h2>
        <pre id="report"></pre>

        <script>
        async function run() {

            const task = document.getElementById("task").value;

            if (!task) {
                alert("Enter a task first.");
                return;
            }

            document.getElementById("code").textContent = "⏳ Running...";
            document.getElementById("report").textContent = "";

            const response = await fetch("/run", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({task: task})
            });

            const data = await response.json();

            document.getElementById("code").textContent =
                data.generated_code || data.message;

            document.getElementById("report").textContent =
                data.report || "";
        }
        </script>

    </body>
    </html>
    """


# Run workflow
@app.post("/run")
def run_workflow(request: TaskRequest):

    task = request.task.strip()

    if not task:
        return {"status": "error", "message": "Task is empty"}

    state = {
        "messages": [HumanMessage(content=task)],
        "next_step": "task_input",
        "code": None,
        "report": None
    }
    try:
        result = workflow.invoke(state)

        return {
            "status": "success",
            "generated_code": result["code"],
            "report": result["report"]
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# Start server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
