from typing import Any, Dict, List

import logging

def format_context(results: List[Dict[str, Any]]) -> str:
    """Format search results into context for the LLM"""
    context_parts = []

    for result in results:
        # TODO: make metadata keys configurable
        ticket_metadata = result.metadata
        ticket_content = result.page_content

        context_parts.append(
            f"Key: {ticket_metadata['ticket']} | Status: {ticket_metadata['status']} - "
            f"Type: {ticket_metadata['type']}\n"
            f"Content: {ticket_content}...\n"
        )

    return "\n\n".join(context_parts)

def trim_answer(generated_answer: str, label_separator: str) -> str:
    """
    From the generated_answer, remove all content after and including
    the provided label separator
    Args:
        generated_answer (str): the generated answer to remove from
        label_separator (str): string after which content needs to be trimmed
                               Note: this string will also be trimmed
    Returns:
        str: Cleaned answer with all content before the label separator
    """
    if not generated_answer:  # Handle empty text
        return ""
    answer = generated_answer
    # Split text at the token and take only the content before it
    if label_separator in generated_answer:
        answer = generated_answer.split(label_separator, 1)[0]
        logging.info(f"Label separator: {label_separator} seems to have been included in the generated answer and it has been removed: {answer}")

    return answer.strip()


def get_answer_with_settings(question, retriever, client, model_id, max_tokens, model_temperature, system_prompt):
    docs = retriever.invoke(input=question)
    num_of_docs = len(docs)
    logging.info(f"Number of relevant documents retrieved and that will be used as context for query: {num_of_docs}")

    logging.info(f"Relevant docs retrieved from Vector store: {docs}")
    context = format_context(docs)
    logging.info(f"Context after formatting: {context}")

    logging.info("Calling chat completions for JSON model...")
    try:
        completions = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            max_tokens=max_tokens,
            temperature=model_temperature,
            stream=False,
        )
    except Exception as e:
        # Handle any error
        logging.error(f"An unexpected error occurred: {e}")
        errorToUI = {
            "answer": f"Please try another question. Received error from LLM invocation: {e}",
            "relevant_tickets": [],
            "sources": [],
            "context": context,
        }
        return errorToUI

    generated_answer = completions.choices[0].message.content

    # Handle common hallucinations observed:
    #    1. Added-Context hallucination
    #    2. Added-Question hallucination
    #    3. Added-Context hallucination just labelled as "Content" (instead of Context like 1)
    logging.info(f"Removing any observed hallucinations in the generated answer: {generated_answer}")
    labels_to_trim = ["Context:", "Question:", "Content:"]
    answer = generated_answer

    for label in labels_to_trim:
        if label in answer:
            answer = trim_answer(answer, label)

    logging.info(f"Answer (after cleanup): {answer}")

    answerToUI = {
        "answer": answer,
        "relevant_tickets": [r.metadata["ticket"] for r in docs],
        "sources": [r.metadata["source"] for r in docs],
        "context": context,  # TODO: if this is big consider logging context here and sending some reference id to UI
    }
    return answerToUI
