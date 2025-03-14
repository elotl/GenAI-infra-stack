import os
import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI

data_version = "v1.1_march14"
print("CREATE the SQL DB from the CSV - to be done only once\n")
df = pd.read_csv('customersupport_tickets_' + data_version + '.csv')
engine = create_engine("sqlite:///customersupport.db")
df.to_sql("customersupport", engine, index=False)

print("CREATE the SQL engine to query from...\n")
db = SQLDatabase(engine=engine)
print("DB dialect is:", db.dialect)
print("Usable table names: ", db.get_usable_table_names())
print("SQL DB table info (usually used in text-to-sql prompt): ", db.get_table_info(["customersupport"]))
print("----------------------------------------------------------\n")
print("Running a sample query on the DB\n")
print(db.run("SELECT COUNT(*) FROM customersupport WHERE assignee_name LIKE 'David Levey';"))


# Use OpenAI for testing txt to SQL
#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Use self-hosted model for testing txt to SQL
MODEL_ID="rubra-ai/Phi-3-mini-128k-instruct"
llm = ChatOpenAI(
  model=MODEL_ID,
  temperature=0,
  openai_api_base="http://localhost:9000/v1",
  openai_api_key="dummy_value",
)

from langchain.chains import create_sql_query_chain
chain = create_sql_query_chain(llm, db=db)

response = chain.invoke({"question": "How many total rows are in the customer support table?"})
print("LLM converted text-to-SQL query:", response)

x = response.split(":")
dbresponse = db.run(x[1])
print("Running query against DB, SQL query result is: ", dbresponse)

