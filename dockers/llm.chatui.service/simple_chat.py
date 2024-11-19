import os
import sys
import urllib

import gradio as gr
import requests

# Environment variable setup
RAG_LLM_QUERY_URL = os.getenv("RAG_LLM_QUERY_URL")
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")

if RAG_LLM_QUERY_URL is None:
    print(
        "Please set the environment variable, RAG_LLM_QUERY_URL (to the IP of the RAG + LLM endpoint)"
    )
    sys.exit(1)

if JIRA_BASE_URL is None:
    print(
        "Please set the environment variable, JIRA_BASE_URL (to the base URL of the JIRA instance)"
    )
    sys.exit(1)

print("RAG query endpoint, RAG_LLM_QUERY_URL: ", RAG_LLM_QUERY_URL)
print("JIRA base URL, JIRA_BASE_URL: ", JIRA_BASE_URL)

USE_CHATBOT_HISTORY = os.getenv("USE_CHATBOT_HISTORY", "False") == "True"

print(f"Use history {USE_CHATBOT_HISTORY}")


# Function to generate clickable links for JIRA tickets
def generate_jira_links(answer_key, relevant_tickets):
    links = []
    for ticket_id in relevant_tickets:
        ticket_url = f"{JIRA_BASE_URL}/browse/{ticket_id}"
        links.append(f'<a href="{ticket_url}" target="_blank">{ticket_id}</a>')
    return links


# Function to fetch the response from the RAG+LLM API
def get_api_response(user_message):
    try:
        question = urllib.parse.quote(f"{user_message}")
        response = requests.get(f"{RAG_LLM_QUERY_URL}/answer/{question}")
        if response.status_code == 200:
            result = response.json()

            if "answer" not in result.keys():
                return "Could not fetch response."

            result = result["answer"]
            answer = result.get("answer", "Could not fetch response.")
            relevant_tickets = result.get("relevant_tickets", [])
            links = generate_jira_links(answer, relevant_tickets)
            clickable_links = "<br>".join(links)
            return f"{answer}<br><br>Relevant Tickets:<br>{clickable_links}"
        else:
            return "API Error: Unable to fetch response."
    except requests.RequestException:
        return "API Error: Failed to connect to the backend service."


# Chatbot response functions
def chatbot_response_no_hist(_chatbot, user_message):
    response_text = get_api_response(user_message)
    return [[user_message, response_text]], ""


def chatbot_response(history, user_message):
    response_text = get_api_response(user_message)
    history.append((user_message, response_text))
    return history, ""


# Gradio UI setup
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(placeholder="Type here...", label="Message")
            send_button = gr.Button("Send")

    if USE_CHATBOT_HISTORY:
        msg.submit(chatbot_response, inputs=[chatbot, msg], outputs=[chatbot, msg])
        send_button.click(
            chatbot_response, inputs=[chatbot, msg], outputs=[chatbot, msg]
        )
    else:
        msg.submit(
            chatbot_response_no_hist, inputs=[chatbot, msg], outputs=[chatbot, msg]
        )
        send_button.click(
            chatbot_response_no_hist, inputs=[chatbot, msg], outputs=[chatbot, msg]
        )

app.launch()
