import logging
import os
import sys
import urllib
from logging.handlers import TimedRotatingFileHandler

import gradio as gr
import requests

# When running locally: export CHATUI_LOGS_PATH=logs/chatui.log
log_file_path = os.getenv("CHATUI_LOGS_PATH") or "/app/logs/chatui.log"
os.makedirs(
    os.path.dirname(log_file_path), exist_ok=True
)  # Ensure log directory exists
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # Log to file, rotate every 1H and store files from last 24 hrs * 7 days files == 168H data
        TimedRotatingFileHandler(log_file_path, when="h", interval=1, backupCount=168),
        logging.StreamHandler(),  # Also log to console
    ],
)

# Environment variable setup
INFERENCE_QUERY_URL = os.getenv("INFERENCE_QUERY_URL")

if INFERENCE_QUERY_URL is None:
    logging.error(
        "Please set the environment variable, INFERENCE_QUERY_URL (to the IP of the LLM INFERENCE endpoint)"
    )
    sys.exit(1)

logging.info(f"Inference query endpoint, INFERENCE_QUERY_URL: {INFERENCE_QUERY_URL}")

USE_CHATBOT_HISTORY = os.getenv("USE_CHATBOT_HISTORY", "True") == "True"

logging.info(f"Use history {USE_CHATBOT_HISTORY}")


def clean_answer(text: str, chatml_end_token: str) -> str:
    """
    Remove all content after and including the specified token from the text.
    Args:
        text (str): The input text to clean
        chatml_end_token (str): The token after which all content should be removed
    Returns:
        str: Cleaned text with all content after and including the token removed
    """
    if not text:  # Handle empty text
        return ""

    if not chatml_end_token:  # Handle empty end tokens
        return text

    # Split text at the token and take only the content before it
    if chatml_end_token in text:
        text = text.split(chatml_end_token, 1)[0]
        logging.info(f"Cleaned text before chatml_end_token: {text}")

    return text.strip()


# Function to generate clickable links for JIRA tickets
def generate_source_links(sources):
    links = []
    for source in sources:
        links.append(f'<a href="{source}" target="_blank">{source}</a>')
    return links


# Function to fetch the response from the RAG+LLM API
def get_api_response(user_message):
    try:
        text = urllib.parse.quote(f"{user_message}")

        logging.info(f"Received text for summarization: {text}")
        
        response = requests.get(f"{INFERENCE_QUERY_URL}/summarize/text={text}")
        if response.status_code == 200:
            logging.info(f"resonse is: {response}\n")
            result = response.json()

	    #json_key_in_response="answer"
            #if json_key_in_response not in result.keys():
            #    return "Could not fetch response."

            #result = result[json_key_in_response]

 	    # we can skip cleanup of answer for text summarizer.
            #answer = clean_answer(
            #    result.get(json_key_in_response, "Could not fetch response."), "<|im_end|>"
            #)

            # sources = result.get("sources", [])
            # links = generate_source_links(sources)
            # clickable_links = "<br>".join(links)
            #context = result.get("context", "")
            logging.info(f"Input Text: {text}\nSummary: {result}\n")

            #return f"{answer}<br><br>Relevant Tickets:<br>{clickable_links}"
            return f"{result}"
        else:
            return "API Error: Unable to fetch response."
    except requests.RequestException:
        return "API Error: Failed to connect to the backend service."


# Chatbot response functions
def chatbot_response_no_hist(_chatbot, user_message):
    response_text = get_api_response(user_message)
    return (
        [[user_message, response_text]],
        "",
        gr.update(value=1, visible=True),
        gr.update(visible=True),
        user_message,
        response_text,
    )


def chatbot_response(history, user_message):
    response_text = get_api_response(user_message)
    history.append((user_message, response_text))
    return (
        history,
        "",
        gr.update(value=1, visible=True),
        gr.update(visible=True),
        user_message,
        response_text,
    )


def submit_rating(rating, user_message, bot_response):
    logging.info(
        f"User rating: {rating}\nQuestion: {user_message}\nAnswer: {bot_response}"
    )
    # Hide the rating slider and submit button after submission
    return gr.update(visible=False), gr.update(visible=False)


# In the Gradio UI setup section, change:
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column(scale=4):
            # Change from chatbot = gr.Chatbot() to:
            chatbot = gr.Chatbot(label="Text Summarizer", height=600)

            # Rating slider and submit button initially hidden
            rating_slider = gr.Slider(
                label="Rate the response", minimum=1, maximum=5, step=1, visible=False
            )
            submit_rating_btn = gr.Button("Submit Rating", visible=False)

            msg = gr.Textbox(placeholder="Enter text to be summarized here...", label="Input Text")
            send_button = gr.Button("Send")
    # Hidden variables to hold user_message and bot_response for rating submission
    user_message = gr.State()
    bot_response = gr.State()

    if USE_CHATBOT_HISTORY:
        msg.submit(
            chatbot_response,
            inputs=[chatbot, msg],
            outputs=[
                chatbot,
                msg,
                rating_slider,
                submit_rating_btn,
                user_message,
                bot_response,
            ],
        )
        send_button.click(
            chatbot_response,
            inputs=[chatbot, msg],
            outputs=[
                chatbot,
                msg,
                rating_slider,
                submit_rating_btn,
                user_message,
                bot_response,
            ],
        )
    else:
        msg.submit(
            chatbot_response_no_hist,
            inputs=[chatbot, msg],
            outputs=[
                chatbot,
                msg,
                rating_slider,
                submit_rating_btn,
                user_message,
                bot_response,
            ],
        )
        send_button.click(
            chatbot_response_no_hist,
            inputs=[chatbot, msg],
            outputs=[
                chatbot,
                msg,
                rating_slider,
                submit_rating_btn,
                user_message,
                bot_response,
            ],
        )

    # Handle rating submission with the button
    submit_rating_btn.click(
        submit_rating,
        inputs=[rating_slider, user_message, bot_response],
        outputs=[rating_slider, submit_rating_btn],
    )

app.launch(server_name="0.0.0.0")
