import requests
import json
from urllib.request import urlopen

url = 'http://gdcb.westeurope.cloudapp.azure.com/upload/' # Set destination URL here

CarIDs = [2]
Codes = ["57"]
Values = [44.1728233]

post_fields = { 'CarID': CarIDs, 'Code': Codes, 'Value': Values }

request = requests.post(url, data=json.dumps(post_fields), headers={'Content-Type': 'application/json'})
data = json.loads(request.text)
print(data)


"""
DELETE FROM Carbox.dbo.RawData
WHERE ID in (select top 1000 ID from Carbox.dbo.RawData order by ID desc);
"""