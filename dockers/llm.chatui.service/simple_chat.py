import os
import sys
import urllib
import gradio as gr
import requests

# Environment variable setup
RAG_LLM_QUERY_URL = os.getenv("RAG_LLM_QUERY_URL")
if RAG_LLM_QUERY_URL is None:
    print("Please set the environment variable, RAG_LLM_QUERY_URL (to the IP of the RAG + LLM endpoint)")
    sys.exit(1)

USE_CHATBOT_HISTORY = os.getenv("USE_CHATBOT_HISTORY", "False") == "True"
print(f"RAG query endpoint, RAG_LLM_QUERY_URL: {RAG_LLM_QUERY_URL}")
print(f"Use history {USE_CHATBOT_HISTORY}")

def clean_answer(text: str, str_to_remove: str) -> str:
    """
    Remove specified string from the text and clean up the answer.
    
    Args:
        text (str): The input text to clean
        str_to_remove (str): The string to remove from the text
        
    Returns:
        str: Cleaned text with specified string removed and whitespace stripped
    """
    if not text or not str_to_remove:  # Handle empty inputs
        return ""
    # Remove specified string
    text = text.replace(str_to_remove, "")
    # Remove context if present
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1]
    return text.strip()

def get_api_response(user_message):
    try:
        response = requests.get(f"{RAG_LLM_QUERY_URL}/answer/{urllib.parse.quote(user_message)}")
        if response.status_code == 200:
            result = response.json().get("answer", {})
            answer = clean_answer(result.get("answer", "Could not fetch response."), "<|im_end|>")
            sources = result.get("sources", [])
            if sources:  # Only add sources section if there are sources
                links = "<br>".join(f"<a href='{src}' target='_blank'>{src}</a>" for src in sources)
                return f"{answer}<br><br>Relevant Tickets:<br>{links}"
            return answer
        return "API Error: Unable to fetch response."
    except requests.RequestException:
        return "API Error: Failed to connect to the backend service."

def chatbot_response_no_hist(_chatbot, user_message):
    if not user_message.strip():  # Check for empty messages
        return _chatbot, "", gr.update(visible=False), user_message, ""
    response_text = get_api_response(user_message)
    return [[user_message, response_text]], "", gr.update(visible=True), user_message, response_text

def chatbot_response(history, user_message):
    if not user_message.strip():  # Check for empty messages
        return history, "", gr.update(visible=False), user_message, ""
    response_text = get_api_response(user_message)
    history.append((user_message, response_text))
    return history, "", gr.update(visible=True), user_message, response_text

def submit_rating(rating, user_message, bot_response):
    if rating:  # Only log if rating is provided
        print(f"User rating: {rating}\nQuestion: {user_message}\nAnswer: {bot_response}")
    return gr.update(visible=False)

# Gradio UI setup
with gr.Blocks(title="Question-Answer Chatbot") as app:
    gr.Markdown("## Question-Answer Chatbot")
    
    chatbot = gr.Chatbot(show_label=False, height=400)
    
    # Rating component using Radio
    with gr.Group(visible=False) as rating_container:
        gr.Markdown("Rate this response:")
        rating = gr.Radio(
            choices=["1 ★", "2 ★★", "3 ★★★", "4 ★★★★", "5 ★★★★★"],
            label="Rating",
            show_label=False,
            value=None  # No default selection
        )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask your question here...",
            show_label=False,
            scale=9,
            container=False
        )
        send_button = gr.Button("Send", scale=1)

    # Hidden states for rating
    user_message = gr.State()
    bot_response = gr.State()

    # Event handlers
    if USE_CHATBOT_HISTORY:
        msg.submit(chatbot_response, [chatbot, msg], [chatbot, msg, rating_container, user_message, bot_response])
        send_button.click(chatbot_response, [chatbot, msg], [chatbot, msg, rating_container, user_message, bot_response])
    else:
        msg.submit(chatbot_response_no_hist, [chatbot, msg], 
                  [chatbot, msg, rating_container, user_message, bot_response])
        send_button.click(chatbot_response_no_hist, [chatbot, msg], 
                         [chatbot, msg, rating_container, user_message, bot_response])

    rating.change(submit_rating, [rating, user_message, bot_response], [rating_container])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", show_error=True)
