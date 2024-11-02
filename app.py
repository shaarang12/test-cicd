from flask import Flask, request, jsonify
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import os
import test_cases
app = Flask(__name__)

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-pro')


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


@app.route('/evaluate', methods=['GET'])
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
            "total_cases": len(test_cases.TEST_CASES),
            "GK Performance": successes*100/len(test_cases.TEST_CASES)
        })

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=8080, debug=False)
