import os

def file_walker(start_path, file_to_search):

	try:
		for entry in os.listdir(start_path):
			entry_path = os.path.join(start_path, entry)
			if os.path.isdir(entry_path):
				file_walker(entry_path, file_to_search)
			else:
				#print(entry)
				if entry == file_to_search:
					print("File found {}".format(entry_path))
					return
				#print(entry_path)
	except Exception:
		pass
