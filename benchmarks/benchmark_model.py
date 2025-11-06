import time
import sys


import os
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from statistics import mean
from langchain.evaluation.qa import QAEvalChain
from langchain_google_genai import ChatGoogleGenerativeAI # use any evaluator (Gemini not yet supported for eval)
from src.LangChainWithChromaAndWeb import chain_with_history  # your existing chain file
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

# --- Test Questions and Expected Answers ---
test_cases = [
    {
        "question": "How do I renew my driver's license in New York?",
        "expected": "You can renew online via the NY DMV website or at a local office."
    },
    {
        "question": "How can I apply for food assistance benefits?",
        "expected": "You can apply for SNAP benefits through your state’s social services website."
    },
    {
        "question": "What are the latest tax filing deadlines for 2025?",
        "expected": "The federal tax filing deadline for 2025 is around April 15."
    },
]

# --- Evaluate Latency and Responses ---
results = []
latencies = []

for case in test_cases:
    start = time.time()
    answer = chain_with_history.invoke({"question": case["question"]}, config={"configurable": {"session_id": "test"}})
    latency = time.time() - start
    latencies.append(latency)
    results.append({"question": case["question"], "answer": answer})
    print(f"\nQ: {case['question']}\nA: {answer}\nTime: {latency:.2f}s")

# --- Latency Metrics ---
avg_latency = mean(latencies)
print(f"\n Average response time: {avg_latency:.2f} seconds")

# --- Accuracy Evaluation (Semantic Comparison) ---
examples = [{"query": t["question"], "answer": t["expected"]} for t in test_cases]
predictions = [{"query": t["question"], "result": r["answer"]} for t, r in zip(test_cases, results)]

eval_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=API_KEY,
)

qa_eval = QAEvalChain.from_llm(eval_llm)
graded_outputs = qa_eval.evaluate(examples, predictions)

# --- Display Evaluation ---
correct = sum(1 for g in graded_outputs if "correct" in g["results"].lower())
accuracy = (correct / len(test_cases)) * 100

print(f"✅ Retrieval Accuracy: {accuracy:.1f}%")
