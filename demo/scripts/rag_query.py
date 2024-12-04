import requests
import urllib.parse
import json

def main():
    hostname = "localhost:8000"
    query = input("Type your query here: ")

    # Encode the question using urllib.parse
    encoded_question = urllib.parse.quote(query)

    url = f"http://{hostname}/answer/{encoded_question}"

    response = requests.get(url)

    if response.status_code == 200:
        response = response.text.strip()
        data = json.loads(response)
        answer = data['answer']
        print(f"Answer: {answer}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    main()
