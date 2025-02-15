import re
from openai import BadRequestError

from typing import Any, Dict, List
from typing_extensions import Annotated
from typing_extensions import TypedDict

from sqlalchemy import create_engine

from transformers import AutoTokenizer

from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

from langchain_core.prompts import PromptTemplate

import logging.config


LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
    },
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'elotl-qa-in-a-box.log',
            'formatter': 'default',
        },
        'stdout': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        },
    },
    'loggers': {
        'ElotlQAInABoxLogger': {
            'handlers': ['file', 'stdout'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('ElotlQAInABoxLogger')

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

MODEL_MAX_CONTEXT_LEN = 8192
delta = 50 # value by which we keep the prompt len less than the model context len  


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
        logger.info(f"Label separator: {label_separator} seems to have been included in the generated answer and it has been removed: {answer}")

    return answer.strip()


# Answer user's question via vector search or RAG technique
def get_answer_with_settings(question, retriever, client, model_id, max_tokens, model_temperature, system_prompt):
    docs = retriever.invoke(input=question)

    num_of_docs = len(docs)
    logger.info(f"Number of relevant documents retrieved and that will be used as context for query: {num_of_docs}")
    logger.info(f"Relevant docs retrieved from Vector store: {docs}")

    context = format_context(docs)
    logger.info(f"Length of context after formatting: {len(context)}")
    logger.info(f"Context after formatting: {context}")

    logger.info("Calling chat completions for JSON model...")
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
    except BadRequestError as e:
        if (e.status_code == 400 and
                "Please reduce the length of the messages or completion." in e.body.get("message", "") and
                len(docs) > 1
        ):
            docs = docs[:-1]  # removing last document
            context = format_context(docs)
            logger.info(f"Need to decrease context - length of context after formatting: {len(context)}")
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
                logger.error(f"An unexpected error occurred: {e}")
                errorToUI = {
                    "answer": f"Please try another question. Received error from LLM invocation: {e}",
                    "relevant_tickets": [],
                    "sources": [],
                    "context": context,
                }
                return errorToUI

    except Exception as e:
        # Handle any error
        logger.error(f"An unexpected error occurred: {e}")
        errorToUI = {
            "answer": f"Please try another question. Received error from LLM invocation: {e}",
            "relevant_tickets": [],
            "sources": [],
            "context": context,
        }
        return errorToUI
    
    generated_answer = completions.choices[0].message.content

    answer = postprocess_hallucinations(generated_answer)

    answerToUI = {
        "answer": answer,
        "relevant_tickets": [r.metadata["ticket"] for r in docs],
        "sources": [r.metadata["source"] for r in docs],
        "context": context,  # TODO: if this is big consider logger context here and sending some reference id to UI
    }
    return answerToUI


# Answer user's question via text-to-sql technique
def get_sql_answer(question, model_id, max_tokens, model_temperature, llm_server_url):
    
    logger.info("Invoking text-to-sql question-answer search")
    try:
        llm = ChatOpenAI(
            model=model_id,
            temperature=model_temperature,
            openai_api_base=llm_server_url,
            openai_api_key="unused-for-self-hosted-llms",
            max_tokens=max_tokens
        )        

        logger.info("Loading the pre-created SQL DB")
        engine = create_engine("sqlite:////app/db/zendesk.db")

        logger.info("Check that the SQL data can be accessed from the DB via querying")
        db = SQLDatabase(engine=engine)
        logger.info(f"DB dialect is: {db.dialect}")
        logger.info(f"Usable table names: {db.get_usable_table_names()}")
        logger.info("Table info:")
        print(db.get_table_info(["zendesk"]))
        
        logger.info("Running sanity test SQL query:") 
        db.run("SELECT COUNT(*) FROM zendesk WHERE assignee_name LIKE 'John Doe';")
        
        # Prompt template to convert NL question to SQL
        # This was manually retrieved from langchain hub and customized
        query_prompt_template = prompt_template_for_text_to_sql()

        logger.info("Prompt template for text-to-sql conversion:" 
                     "{query_prompt_template.messages[0]}")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        errorToUI = {
            "answer": f"Please try another question. Received error from LLM invocation: {e}",
            "relevant_tickets": [],
            "sources": [],
            "context": "",
        }
        return errorToUI

    # send SQL query and response to LLM and get a natural language answer
    # TODO avoid calling text to SQL conversion twice
    state: State = {
        "question": question,
        "query": write_query({"question": question}, query_prompt_template, llm, db),
        "result": execute_query(write_query({"question": question}, query_prompt_template, llm, db), db),
    }
    generated_answer = convert_sql_result_to_nl(state, model_id, llm)
    answer = postprocess_hallucinations(generated_answer['answer'])
    
    answerToUI = {
        "answer": answer,
        "relevant_tickets": ["n/a"],
        "sources": ["n/a"],
        "context": "",  # TODO: if this is big consider logger context here and sending some reference id to UI
    }
    return answerToUI

def prompt_template_for_text_to_sql():

    query_prompt_template = PromptTemplate.from_template(
        "Given an input question, create a syntactically correct "
        "{dialect} query to run to help find the answer." 
        "Unless the user specifies in his question a specific " 
        "number of examples they wish to obtain, always limit " 
        "your query to at most {top_k} results. You can order "
        "the results by a relevant column to return the most " 
        "interesting examples in the database. Never query for " 
        "all the columns from a specific table, only ask for a " 
        "few relevant columns given the question. Pay attention " 
        "to use only the column names that you can see in the schema "
        "description. Be careful to not query for columns that do "
        "not exist. Also, pay attention to which column is in " 
        "which table. Only use the following tables: {table_info}."
        "If there is a ticket ID in the question, ensure that you maintain "
        "the exact ticket ID in the query."
        "Question: {input}")
    
    # Alternatively uncomment below to use prompt template from hub directly
    # without customization    
    # query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
    # assert len(query_prompt_template.messages) == 1
        
    return query_prompt_template

def postprocess_hallucinations(generated_answer: str) -> str:

    # Handle common hallucinations observed:
    #    1. Added-Context hallucination
    #    2. Added-Question hallucination
    #    3. Added-Context hallucination just labelled as "Content" (instead of Context like 1)
    logger.info(f"Removing any observed hallucinations in the generated answer: {generated_answer}")
    labels_to_trim = ["<|im_end|>", "Context:", "Question:", "Content:"]
    answer = generated_answer

    for label in labels_to_trim:
        if label in answer:
            answer = trim_answer(answer, label)

    answer = answer.replace("<|im_start|>", "")

    logger.info(f"Answer (after cleanup): {answer}")

    return answer


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


# Post-process LLM output to extract only the SQL query, handles both cases when
# returned output is of these forms:
# a) "sql: select title from employee limit 10"
# b) SELECT subject FROM tickets WHERE ticket_id = 685490;<|im_end|>
#    <|im_start|>user>
#    Question: What are the details of the tickets with the highest priority?<|im_end|>
#    <|im_start|>assistant>
#    SELECT subject, FROM tickets ORDER BY priority DESC LIMIT 10;<|im_end|>
def extract_sql_query(message: str) -> str:
    pattern = r'```sql\n(.*?)\n```'
    match = re.search(pattern, message, re.DOTALL)

    if match:
        return match.group(1) 
    else:
        sql_query = postprocess_hallucinations(message)

    return sql_query


def write_query(state: State, query_prompt_template, llm, db):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    # structured output wasn't implemented for the RUBRA-phi3 model so had to move to the 
    # raw llm invoke. If we can move to a different function-calling LLM, we can uncomment
    # this.
    #structured_llm = llm.with_structured_output(QueryOutput)
    #result = structured_llm.invoke(prompt)

    logger.info(f"Prompt for SQL query generation: {prompt}")
    result = llm.invoke(prompt)

    sql_query = extract_sql_query(result.content)

    logger.info(f"Extracted SQL query: {sql_query}")
    
    return {"query": sql_query}


# Executes a SQL query against the provided database
def execute_query(state: State, db):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDataBaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

# Trims a string to fit within a given token limit using a model-specific tokenizer.
def trim_text_by_tokens(text: str, model_id: str, token_limit: int) -> str:
 
    # Load a tokenizer that is specific to the model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    tokens = tokenizer(text)["input_ids"]
    
    # If within the limit, return the original text
    if len(tokens) <= token_limit:
        return text
    
    # Trim the tokens to the allowed limit
    trimmed_tokens = tokens[:token_limit]
    
    # Decode back into text
    trimmed_text = tokenizer.decode(trimmed_tokens, skip_special_tokens=True)
    
    return trimmed_text

# Answer user's question in natural language using SQL query and SQL query results from the 
# database as context. 
def convert_sql_result_to_nl(state: State, model_id, llm):
    
    domainExpertInstructions = "In the provided SQL table, each entry or row refers to a ticket and not to a customer." 
    " The column titled requester is also referred to as the customer or submitter or client."
    " The column titled all_comments can also be referred to as responses or resolution or details"
    
    prompt = (
        "You are a customer support ticket expert. Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user's question."
        "Do not make any references to the SQL query or the SQL result in your answer." 
        + domainExpertInstructions + 
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    logger.info(f"Prompt for SQL result to NL conversion: {prompt}. Prompt length: {len(prompt)}")

    # Prompt length has to be smaller than model max because of errors like this:
    # This model's maximum context length is 8192 tokens. However, you requested 8220 
    # tokens in the messages, Please reduce the length of the messages.", 
    # 'type': 'BadRequestError',
    # 'param': None, 'code': 400}
    PROMPT_TRIM_LENGTH = MODEL_MAX_CONTEXT_LEN - delta
    trimmed_prompt = trim_text_by_tokens(prompt, model_id, PROMPT_TRIM_LENGTH) 
    logger.info(f"Trimmed prompt: {trimmed_prompt}")

    response = llm.invoke(trimmed_prompt)
    logger.info(f"LLM generated NL answer to user question: {response.content}")

    return {"answer": response.content}
