# End-User Docs for using Elotl's Question-Answer ChatBot

Elotl's Question-Answer ChatBot is powered by these technologies:

1. Retrieval Augmented Generation
2. Open-Source Large Language Models


## What types of questions can a RAG+LLM Chatbot answer well?

- Chatbots powered with RAG & LLMs are good at answering questions about a small subset of relevant data from the given Knowledge Base. This is because of the way Retrieval Augmented Generation works: User questions are first matched to "chunks" of data from the user's Knowledge Base. The questions are then answered using ONLY the matching subset of the user's private data.

- Let's take for example, a Question-Answer Chatbot working with an engineering team's JIRA ticket dataset. Here are some example questions that can be answered well:


```sh
What was the issue with the Apache Airflow installation?
What type of security issues have been handled? 
Were the SQL issues resolved for the ENG team?
Were any Kubernetes upgrades initiated on the cloud? 
Can you give the description of any upgrade requests that have been received? 
```

## What types of questions is a RAG+LLM Chatbot not intended to answer?

- RAG + LLM technology is not intended to answer knowledge aggregation questions about a large amount of data from a Knowledge Base.

```sh
What is the most frequent Kubernetes issue? 
Can you summarize all the recent upgrade tasks?
Can you find the ticket that's been unresolved for the longest time?
```

