# Prepare csv dump of Jira tickets adjusted for embedding

[Install UV](https://docs.astral.sh/uv/getting-started/installation/) than create venv and install deps
```shell
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
set jira_url in jira_config.ini

```shell
uv run process_jira_tickets.py jira_elotl.csv jira_config.ini output_files
```
