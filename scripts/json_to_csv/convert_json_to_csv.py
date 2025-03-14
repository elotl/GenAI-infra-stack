import json
import csv
from datetime import datetime
import io

def clean_text(text):
	"""Clean text by removing extra whitespace and newlines"""
	if text is None:
		return ""
	return ' '.join(str(text).split()).strip()

def	safe_get (obj, *keys, default = ""):
		"""Safely get nested dictionary values with a default if not found"""
		try:
			for	key	in keys:
				if not isinstance(obj, dict):
					return default
				newobj = obj.get(key)
				if newobj is None:
					return default
			return newobj
		except Exception:
			return default

def format_comment(comment):
	""" Format a single comment with proper escaping"""
	if not comment:
		return ""
	try:
		parts =[
		f"ID={safe_get(comment, 'id', default='N/A')}",
		f"Author={safe_get(comment, 'author_id', default='N/A')}",
		f"Time={safe_get(comment, 'created_at', default='N/A')}",
		f"Public={safe_get(comment, 'public', default='N/A')}",
		f"Body={clean_text(safe_get(comment, 'body', default=''))}"
		]
		return "; ".join(parts)
	except Exception as e:
		print (f"Warning: Error formatting comment: {str(e)}")
		return ""


def	format_comments(comments):
	"""Format all comments into a single string with proper escaping """
	if not comments:
		return	""

	formatted=[]
	for comment in comments:
		comment_str = format_comment(comment)
		if comment_str:
			formatted.append(comment_str)
	
	return " || ".join(formatted)

def process_ticket(ticket_data):
	""" Process a single ticket and return a row dictionary """
	if not ticket_data:
		return None

	try:
		# Process tags with proper escaping
		tags = safe_get(ticket_data, 'tags', default =[])
		tags_str = ';'.join(str(tag) for tag in tags if tag is not None)

		return {
				'ticket_id':safe_get(ticket_data, 'id'),
				'ticket_url':safe_get(ticket_data, 'url'),
				'ticket_type':clean_text(safe_get(ticket_data, 'type')), # renaming for clarity
				'subject':clean_text(safe_get(ticket_data, 'subject')),
				'description':clean_text(safe_get(ticket_data, 'description')),
				'details': format_comments(safe_get(ticket_data, 'comments', default =[])), # renaming for clarity
				'created_at':safe_get(ticket_data, 'created_at'),
				'updated_at':safe_get(ticket_data, 'updated_at'),
				'latest_comment_added_at':safe_get(ticket_data, 'latest_comment_added_at'),
				'status':safe_get(ticket_data, 'status'),
				'priority':safe_get(ticket_data, 'priority'),
				'requester_name':safe_get(ticket_data, 'requester', 'name'),
				'requester_email':safe_get(ticket_data, 'requester', 'email'),
				'assignee_name':safe_get(ticket_data, 'assignee', 'name'),
				'assignee_email':safe_get(ticket_data, 'assignee', 'email'),
				'submitter_name':safe_get(ticket_data, 'submitter', 'name'),
				'submitter_email':safe_get(ticket_data, 'submitter', 'email'),
				'organization_name':safe_get(ticket_data, 'organization', 'name'),
				'group_name':safe_get(ticket_data, 'group', 'name'),
				'collaborator_name':safe_get(ticket_data, 'collaborator', 'name'),
				'collaborator_email':safe_get(ticket_data, 'collaborator', 'email'),
				'categories':tags_str,
				'satisfaction_rating_score':safe_get(ticket_data, 'satisfaction_rating', 'score'),
				'number_of_reopens':safe_get(ticket_data, 'metric_set', 'reopens'),
				'number_of_replies':safe_get(ticket_data, 'metric_set', 'replies'),
				'reply_time_in_minutes':safe_get(ticket_data, 'metric_set', 'reply_time_in_minutes', 'business'),
				'full_resolution_time_in_minutes':safe_get(ticket_data, 'metric_set', 'full_resolution_time_in_minutes', 'business'),
				

	    }
	except Exception as e:
		print(f"Warning: Error processing ticket: {str(e)}")
		return None

def convert_customersupport_json_to_csv(json_file_path, csv_file_path):
	# Define CSV headers
	headers=[
	  'ticket_id',
	  'ticket_url',
	  'ticket_type',
	  'subject',
	  'description',
	  'details',
	  'created_at',
	  'updated_at',
	  'latest_comment_added_at',
	  'status',
	  'priority',
	  'requester_name',
	  'requester_email',
	  'assignee_name',
	  'assignee_email',
	  'submitter_name',
	  'submitter_email',
	  'organization_name',
	  'group_name',
	  'collaborator_name',
	  'collaborator_email',
	  'categories',
	  'satisfaction_rating_score',
	  'number_of_reopens',
	  'number_of_replies',
	  'reply_time_in_minutes',
	  'full_resolution_time_in_minutes',
	]
	# Read and process JSON file
	rows =[]
	skipped_tickets = 0
	total_tickets = 0

	with open(json_file_path, 'r', encoding = 'utf-8') as file:
		for line in file:
			total_tickets += 1
			try:
				#Try to parse each line as a separate JSON object
				ticket_data = json.loads(line.strip())
				processed_ticket = process_ticket(ticket_data)
				if processed_ticket:
					rows.append(processed_ticket)
				else:
					skipped_tickets += 1
			except json.JSONDecodeError as e:
				print(f"Warning: Skipping invalid JSON line: {str(e)}")
				skipped_tickets += 1
			continue

	if not rows:
		print("No valid tickets found in the input file")
		return

	# Write to CSV with proper escaping
	with open(csv_file_path, 'w', newline = '', encoding = 'utf-8') as file:
			writer = csv.DictWriter(file,
						fieldnames = headers,
						quoting = csv.QUOTE_ALL,
						#Quote all fields
						escapechar = '\\',
						#Use backslash as escape character
						doublequote = True)
			# Double-quotes within fields
			writer.writeheader()
			writer.writerows(rows)
			print(f"CSV file has been created successfully at: {csv_file_path}")
			print(f"Processed {len(rows)} tickets successfully")
			print(f"Skipped {skipped_tickets} tickets due to errors")
			print(f"Total tickets in input: {total_tickets}")

if __name__ == "__main__":

	json_file_path = "customersupport_tickets.json"
	csv_file_path = "customersupport_tickets.csv"

	try:
		convert_customersupport_json_to_csv(json_file_path, csv_file_path)
	except Exception as e:
			print(f"An error occurred: {str(e)}")
