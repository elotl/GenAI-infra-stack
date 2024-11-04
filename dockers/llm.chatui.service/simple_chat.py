import os
import requests
import sys
import urllib

import gradio as gr

RAG_LLM_QUERY_URL = os.getenv("RAG_LLM_QUERY_URL")
if  RAG_LLM_QUERY_URL is None:
    print("Please set environment variable, RAG_LLM_QUERY_URL (to the IP of the RAG + LLM endpoint)")
    sys.exit(1)
print("RAG query endpoint, RAG_LLM_QUERY_URL: ", RAG_LLM_QUERY_URL)

USE_CHATBOT_HISTORY = os.getenv("USE_CHATBOT_HISTORY", False)

def get_api_response(user_message):
    try:
        question = urllib.parse.quote(f"{user_message}")
        response = requests.get(f"{RAG_LLM_QUERY_URL}/answer/{question}")
        if response.status_code == 200:
            return response.json().get("answer", "Could not fetch response.")
        else:
            return "API Error: Unable to fetch response."
    except requests.RequestException:
        return "API Error: Failed to connect to the backend service."

def chatbot_response_no_hist(_chatbot, user_message):
    random_text = get_api_response(user_message)
    return [[user_message, random_text]], ""

def chatbot_response(history, user_message):
    random_text = get_api_response(user_message)
    history.append((user_message, random_text))
    return history, ""

with gr.Blocks() as app:
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(placeholder="Type here...", label="Message")
            send_button = gr.Button("Send")

    if USE_CHATBOT_HISTORY:
        msg.submit(chatbot_response, inputs=[chatbot, msg], outputs=[chatbot, msg])
        send_button.click(chatbot_response, inputs=[chatbot, msg], outputs=[chatbot, msg])
    else:
        msg.submit(chatbot_response_no_hist, inputs=[chatbot, msg], outputs=[chatbot, msg])
        send_button.click(chatbot_response_no_hist, inputs=[chatbot, msg], outputs=[chatbot, msg])

app.launch()