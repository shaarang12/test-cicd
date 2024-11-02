from flask import Flask, request, jsonify
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import os
import test_cases
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

        for case in test_cases.TEST_CASES:
            response = model.generate_content(case["input"])
            response_text = response.text.lower()
            expected_word = case["expected"].lower()

            contains_expected = expected_word in response_text
            if contains_expected:
                successes += 1

            logging.info(f"\nQ: {case['input']}")
            logging.info(f"A: {response.text.strip()}")

        logging.info(f"\nOverall Results:")
        logging.info(f"Matches: {successes}/{len(test_cases.TEST_CASES)}")

        return jsonify({
            "matches": successes,
            "total_cases": len(test_cases.TEST_CASES)
        })

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/evaluate-summary', methods=['GET'])
def evaluate_summary_performance():
    try:
        logging.info("\n=== Summary Evaluation ===")
        successes = 0

        for i, case in enumerate(test_cases.SUMMARY_TEST_CASES, 1):
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
        logging.info(f"Successful Summaries: {successes}/{len(test_cases.SUMMARY_TEST_CASES)}")

        return jsonify({
            "successful_summaries": successes,
            "total_cases": len(test_cases.SUMMARY_TEST_CASES)
        })

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=8080, debug=False)
