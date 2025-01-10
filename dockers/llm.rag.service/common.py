from typing import Any, Dict, List


def format_context(results: List[Dict[str, Any]]) -> str:
    """Format search results into context for the LLM"""
    context_parts = []

    for result in results:
        # TODO: make metadata keyes configurable
        ticket_metadata = result.metadata
        ticket_content = result.page_content

        context_parts.append(
            f"Key: {ticket_metadata['key']} | Status: {ticket_metadata['status']} - "
            f"Type: {ticket_metadata['type']}\n"
            f"Content: {ticket_content}...\n"
        )

    return "\n\n".join(context_parts)


def get_answer_with_settings(question, retriever, client, model_id, max_tokens, model_temperature):
    SYSTEM_PROMPT = """You are a specialized Jira ticket assistant. Format your responses following these rules:
                1. Start with the most relevant ticket references
                2. Provide a clear, direct answer
                3. Include relevant technical details when present
                4. Mention related tickets if they provide additional context
                5. If the information is outdated, mention when it was last updated
                """

    docs = retriever.invoke(input=question)
    print(
        "Number of relevant documents retrieved and that will be used as context for query: ",
        len(docs),
    )

    context = format_context(docs)

    print("Sending query to the LLM...")
    completions = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        max_tokens=max_tokens,
        temperature=model_temperature,
        stream=False,
    )

    answer = {
        "answer": completions.choices[0].message.content,
        "relevant_tickets": [r.metadata["key"] for r in docs],
        "sources": [r.metadata["source"] for r in docs],
        "context": context,  # TODO: if this is big consider logging context here and sending some reference id to UI
    }
    print("Received answer: ", answer)
    return answer