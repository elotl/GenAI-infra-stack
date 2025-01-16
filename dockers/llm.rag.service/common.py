from typing import Any, Dict, List

import logging

def format_context(results: List[Dict[str, Any]]) -> str:
    """Format search results into context for the LLM"""
    context_parts = []

    for result in results:
        # TODO: make metadata keyes configurable
        ticket_metadata = result.metadata
        ticket_content = result.page_content

        context_parts.append(
            f"Key: {ticket_metadata['ticket']} | Status: {ticket_metadata['status']} - "
            f"Type: {ticket_metadata['type']}\n"
            f"Content: {ticket_content}...\n"
        )

    return "\n\n".join(context_parts)

def remove_context_from_answer(generated_answer: str):
    """
    Remove any Context included in the generated answer after and including the specified token from the text.
    Args:
        text (str): the generated answer to remove context from
    Returns:
        str: Cleaned answer with all content before the context title
    """
    if not generated_answer:  # Handle empty text
        return ""

    context_title = "Context:"
    answer = generated_answer
    # Split text at the token and take only the content before it
    if context_title in generated_answer:
        answer = generated_answer.split(context_title, 1)[0]
        logging.info(f"Context removed from generated answer: {answer}")

    return answer.strip()


def get_answer_with_settings(question, retriever, client, model_id, max_tokens, model_temperature, system_prompt):
    docs = retriever.invoke(input=question)
    logging.info(
        "Number of relevant documents retrieved and that will be used as context for query: ",
        len(docs),
    )

    context = format_context(docs)

    print("Calling chat completions for JSON model...")
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

    # force remove context for the cases when the LLM appends context to the generated answer
    generated_answer = completions.choices[0].message.content
    answer = remove_context_from_answer(generated_answer)

    answer = {
        "answer": answer,
        "relevant_tickets": [r.metadata["ticket"] for r in docs],
        "sources": [r.metadata["source"] for r in docs],
        "context": context,  # TODO: if this is big consider logging context here and sending some reference id to UI
    }
    logging.info("Received answer: ", answer)
    return answer
