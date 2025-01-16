import time

from typing import Any, Dict, List


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


def get_answer_with_settings(question, retriever, client, model_id, max_tokens, model_temperature, system_prompt=""):
    SYSTEM_PROMPT = """You are a specialized support ticket assistant. Format your responses following these rules:
                1. Answer the provided question only using the provided context.
                2. Provide a clear, direct and factual answer
                3. Include relevant technical details when present
                4. If the information is outdated, mention when it was last updated
                """
    
    if system_prompt == "":
        system_prompt = SYSTEM_PROMPT

    start_time = time.time()
    docs = retriever.invoke(input=question)
    print("--- Retrieve docs: %s seconds ---" % (time.time() - start_time))
    print(
        "Number of relevant documents retrieved and that will be used as context for query: ",
        len(docs),
    )

    context = format_context(docs)

    print("Calling chat completions for JSON model...")
    start_time = time.time()
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
    print("--- Completions: %s seconds ---" % (time.time() - start_time))

    answer = {
        "answer": completions.choices[0].message.content,
        "relevant_tickets": [r.metadata["ticket"] for r in docs],
        "sources": [r.metadata["source"] for r in docs],
        "context": context,  # TODO: if this is big consider logging context here and sending some reference id to UI
    }
    print("Received answer: ", answer)
    return answer
