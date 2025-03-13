import requests
import json
from datetime import datetime
import time
import sys
import os
import urllib
import statistics

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
        start_time = time.time()
        response = requests.get(f"{endpoint}/answer/{question}")
        end_time = time.time()
        time_taken = end_time - start_time
        if response.status_code == 200:
            return {
                "answer": response.json().get("answer", "Could not fetch response."),
                "time_taken": time_taken
            }
        else:
            return {"error": "API Error: Unable to fetch response.", "time_taken": time_taken}
    except requests.RequestException:
        return {"error": "API Error: Failed to connect to the backend service.", "time_taken": None}

def save_results(results, output_filename):
    """Save results to a file in a readable format."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{output_filename}_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("Q&A Results\n")
        file.write("=" * 80 + "\n\n")
        duration_list = []
        for i, (question, answer, duration) in enumerate(results, 1):
            file.write(f"Question {i}:\n")
            file.write("-" * 40 + "\n")
            file.write(f"{question}\n\n")
            file.write("Answer:\n")
            file.write("-" * 40 + "\n")
            file.write(f"{answer}\n\n")
            file.write("=" * 80 + "\n\n")
            file.write(f"Duration: {duration}\n\n")
            file.write("-" * 40 + "\n")
            duration_list.append(duration)

        # calculate stats about the duration
        minVal = min(duration_list)
        maxVal = max(duration_list)
        median = statistics.median(duration_list)
        file.write(f"Duration Stats: Min: {minVal}, Max: {maxVal}, Median: {median}")
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
        #answer, duration = send_question(question, RAG_LLM_QUERY_URL)
        response = send_question(question, RAG_LLM_QUERY_URL)
        answer = response['answer']
        duration = response['time_taken']

        results.append((question, answer, duration))
        print("Question:", question, "\n")
        print("Answer:", answer, "\n")
        print("Duration:", duration, "\n-------------\n")
        time.sleep(0.5)
    
    # Save results
    output_file = save_results(results, OUTPUT_FILE_PREFIX)
    print(f"\nResults have been saved to: {output_file}")

if __name__ == "__main__":
    main()
