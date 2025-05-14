import os
import time
import json
import logging
from typing import List, Dict

import requests
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pydantic import BaseModel, Field, ValidationError

#
# ─── LOGGING ─────────────────────────────────────────────────────────────────────
#
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("eval")

#
# ─── TEST QUERY MODEL ────────────────────────────────────────────────────────────
#
class TestQuery(BaseModel):
    question: str = Field(..., description="What to ask the RAG model")
    expected: str = Field(..., description="Reference answer for BLEU scoring")

# Customize these to match your resume
TEST_QUERIES: List[TestQuery] = [
    TestQuery(
        question="What is the candidate's experience with Python?",
        expected="The candidate has experience with Python in projects at Temenos and Thomson Reuters.",
    ),
    TestQuery(
        question="Where is the candidate studying?",
        expected="The candidate is pursuing an MS in Computer Science at San Jose State University.",
    ),
]

#
# ─── EVALUATION FUNCTION ─────────────────────────────────────────────────────────
#
def evaluate(
    question: str,
    expected: str,
    chunk_size: int,
    chunk_overlap: int = 50,
    endpoint: str = "http://localhost:8000/query",
) -> Dict:
    payload = {
        "question": question,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    t0 = time.time()
    try:
        resp = requests.post(endpoint, json=payload)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("Request failed: %s", e)
        return {"chunk_size": chunk_size, "latency": None, "bleu": 0.0, "answer": ""}

    latency = data.get("latency", None)
    answer = data.get("answer", "")
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([expected.split()], answer.split(), smoothing_function=smoothie)

    return {
        "chunk_size": chunk_size,
        "latency": latency,
        "bleu": bleu,
        "answer": answer,
    }

#
# ─── MAIN SCRIPT ────────────────────────────────────────────────────────────────
#
def main():
    results = []
    for cs in [256, 512, 1024]:
        logger.info("Evaluating chunk_size=%d", cs)
        for tq in TEST_QUERIES:
            res = evaluate(tq.question, tq.expected, cs)
            logger.info(
                "Q: %s\nA: %s\nLatency: %.3fs | BLEU: %.2f\n",
                tq.question, res["answer"], res["latency"] or 0, res["bleu"],
            )
            results.append({**res, "question": tq.question})

    # Write out results for further analysis/plotting
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_file = f"eval_results_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved evaluation results to %s", out_file)

if __name__ == "__main__":
    main()