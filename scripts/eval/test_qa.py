import requests
import json
from datetime import datetime
import time
import sys
import os
import urllib

def read_questions(filename):
    """Read questions from a text file, one per line."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Error: Could not find file '{filename}'")
        sys.exit(1)

def send_question(user_message, endpoint):
    """Send a single question to the API endpoint."""

    try:
        question = urllib.parse.quote(f"{user_message}")
        response = requests.get(f"{endpoint}/answer/{question}")
        if response.status_code == 200:
            return response.json().get("answer", "Could not fetch response.")
        else:
            return "API Error: Unable to fetch response."
    except requests.RequestException:
        return "API Error: Failed to connect to the backend service."

    '''try:
        response = requests.post(
            endpoint,
            json={"question": question},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json().get('answer', 'No answer provided')
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    '''

def save_results(results, output_filename):
    """Save results to a file in a readable format."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{output_filename}_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("Q&A Results\n")
        file.write("=" * 80 + "\n\n")
        
        for i, (question, answer) in enumerate(results, 1):
            file.write(f"Question {i}:\n")
            file.write("-" * 40 + "\n")
            file.write(f"{question}\n\n")
            file.write("Answer:\n")
            file.write("-" * 40 + "\n")
            file.write(f"{answer}\n\n")
            file.write("=" * 80 + "\n\n")
    
    return filename

def main():
    # Configuration
    RAG_LLM_QUERY_URL = os.getenv("RAG_LLM_QUERY_URL")
    INPUT_FILE = "mini_rag_questions.txt"  # Replace with your questions file name
    OUTPUT_FILE_PREFIX = "qa_results"
    
    # Read questions
    print("Reading questions from file...")
    questions = read_questions(INPUT_FILE)
    print(f"Found {len(questions)} questions")
    
    # Process questions
    results = []
    for i, question in enumerate(questions, 1):
        print(f"Processing question {i} of {len(questions)}...")
        
        # Send request and get response
        answer = send_question(question, RAG_LLM_QUERY_URL)
        results.append((question, answer))
        
        time.sleep(0.5)
    
    # Save results
    output_file = save_results(results, OUTPUT_FILE_PREFIX)
    print(f"\nResults have been saved to: {output_file}")

if __name__ == "__main__":
    main()
