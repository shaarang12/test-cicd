from flask import Flask, request, jsonify
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import os
app = Flask(__name__)

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')


logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('app_log'),
        logging.StreamHandler()
    ]
)

TEST_CASES = [
    {
        "input": "What is the capital of France?",
        "expected": "Paris"
    },
    {
        "input": "What is 2 + 2?",
        "expected": "4"
    },
    {
        "input": "Who wrote Romeo and Juliet?",
        "expected": "Shakespeare"
    },
    {
        "input": "What is the chemical symbol for gold?",
        "expected": "Au"
    },
    {
        "input": "What planet is known as the Red Planet?",
        "expected": "Mars"
    }
]

SUMMARY_TEST_CASES = [
    {
        "input": """A paragraph is a group of sentences that develop a single idea or point of a subject. 
        Paragraphs are a common feature of writing and are used to organize information and help readers understand the
         main points of a piece."""
    },
    {
        "input": """My name is shaarang. I am 25 years old. I have a masters degree. I am applying as a mlops data
        engineer at syncron. I am a boy. i am from goa."""
    }
]


@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400

        query = data['query']
        logging.info(f"\n=== Query Request ===")
        logging.info(f"Q: {query}")

        response = model.generate_content(query)

        logging.info(f"A: {response.text.strip()}")

        return jsonify({
            "response": response.text.strip()
        })

    except Exception as e:
        logging.error(f"Query Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text']
        logging.info(f"\n=== Summarization Request ===")
        logging.info(f"Original ({len(text.split())} words):")
        logging.info(text)

        prompt = f"Please summarize this text concisely:\n\n{text}"
        response = model.generate_content(prompt)
        summary = response.text.strip()

        logging.info(f"\nSummary ({len(summary.split())} words):")
        logging.info(summary)

        return jsonify({
            "summary": summary,
            "original_length": len(text.split()),
            "summary_length": len(summary.split()),
        })

    except Exception as e:
        logging.error(f"Summarization Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/evaluate-query', methods=['GET'])
def evaluate_gemini():

    try:
        logging.info("\n=== Query Evaluation ===")
        successes = 0

        for case in TEST_CASES:
            response = model.generate_content(case["input"])
            response_text = response.text.lower()
            expected_word = case["expected"].lower()

            contains_expected = expected_word in response_text
            if contains_expected:
                successes += 1

            logging.info(f"\nQ: {case['input']}")
            logging.info(f"A: {response.text.strip()}")

        logging.info(f"\nOverall Results:")
        logging.info(f"Matches: {successes}/{len(TEST_CASES)}")

        return jsonify({
            "matches": successes,
            "total_cases": len(TEST_CASES)
        })

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/evaluate-summary', methods=['GET'])
def evaluate_summary_performance():
    try:
        logging.info("\n=== Summary Evaluation ===")
        successes = 0

        for i, case in enumerate(SUMMARY_TEST_CASES, 1):
            response = model.generate_content(
                f"Please summarize this text concisely:\n\n{case['input']}"
            )

            input_words = len(case['input'].split())
            summary_words = len(response.text.split())
            is_shorter = summary_words < input_words

            if is_shorter:
                successes += 1

            logging.info(f"\nTest Case {i}:")
            logging.info(f"Original ({input_words} words):")
            logging.info(case['input'].strip())
            logging.info(f"\nSummary ({summary_words} words):")
            logging.info(response.text.strip())

        logging.info(f"\nOverall Results:")
        logging.info(f"Successful Summaries: {successes}/{len(SUMMARY_TEST_CASES)}")

        return jsonify({
            "successful_summaries": successes,
            "total_cases": len(SUMMARY_TEST_CASES)
        })

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=8080, debug=False)
