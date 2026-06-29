import requests
import logging

# Constants
MAX_ITERATIONS = 2
CODER_MODEL = "qwen2.5-coder:14b"
CHECKER_MODEL = "lfm2.5:8b"

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def call_ollama(prompt: str, model_name: str) -> str:
    res = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=600,
    )
    res.raise_for_status()
    data = res.json()
    return data["message"]["content"]


def run_self_improvement_loop(prompt: str) -> str:
    code = prompt
    improved_code = code

    for iteration in range(MAX_ITERATIONS):
        logging.info(f"Iteration {iteration + 1}: Generating improved code...")
        improved_code = call_ollama(code, CODER_MODEL)
        logging.info("Code generated. Checking for errors...")

        check_result = call_ollama(improved_code, CHECKER_MODEL)
        logging.info(f"Check result received: {check_result[:100]}")

        if "No errors found" in check_result:
            logging.info("The code is error-free. Improvement loop completed.")
            break

        logging.info("Errors found. Applying fixes...")
        fix_prompt = f"Here is the code with identified errors:\n{improved_code}\nPlease provide a corrected version."
        code = call_ollama(fix_prompt, CODER_MODEL)
        improved_code = code
        logging.info("Fixes applied.")

    return improved_code


if __name__ == "__main__":
    initial_code = "def example_function():\n    return 'Hello, World!'"
    result = run_self_improvement_loop(initial_code)
    print("Final Improved Code:\n", result)