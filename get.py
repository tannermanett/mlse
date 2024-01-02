import requests
import json
import pandas as pd

# Set up API parameters
url = "https://api.sproutsocial.com/v1/1609711/analytics/topics/a003bca4-2823-41d9-9d8f-b8428458b934/metrics"
search_query = "Auston Matthews"
start_date = "2023-08-01"  # Modify to your desired start date
end_date = "2023-08-10"    # Modify to your desired end date

payload = json.dumps({
    "filters": [
        f"created_time.in({start_date}..{end_date})",
        f"message_text.contains({search_query})"
    ],
    "metrics": [
        "message_count",
        "engagement_total",
        "messages"
    ],
    "dimensions": [
        "sentiment"
    ],
    "timezone": "America/Chicago"
})

headers = {
    'Content-Type': 'application/json',
    'authorization': 'Bearer YOUR_ACCESS_TOKEN',
    'X-Sprout-User-Roles': 'read'
}

# Send API request
response = requests.post(url, headers=headers, data=payload)

# Process the response and create a DataFrame
if response.status_code == 200:
    data = response.json()
    
    results = []
    for item in data.get('data', []):
        flattened_item = {
            'date': item['date'],
            'sentiment': item['dimensions']['sentiment'],
            'message_count': item['metrics']['message_count'],
            'engagement_total': item['metrics']['engagement_total'],
            'messages': item['metrics']['messages']
        }
        results.append(flattened_item)
    
    df = pd.DataFrame(results)
    print(df)
else:
    print(f"Request failed with status code {response.status_code}")
