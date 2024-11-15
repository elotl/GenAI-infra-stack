# Prepare csv dump of Jira tickets adjusted for embedding
```commandline
uv run csv_to_json.py process_jira_tickets.csv jira_config.ini output.jsonl
```
or
```commandline
python csv_to_json.py process_jira_tickets.csv jira_config.ini output.jsonl
```