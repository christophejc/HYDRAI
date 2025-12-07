# Create a new file: app/notifications.py

import requests
import os
from dotenv import load_dotenv

load_dotenv()

# IMPORTANT: Replace 'YOUR_SECRET_NTFY_TOPIC' with your actual, secret topic name
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "YOUR_SECRET_HYDR_AI_TOPIC") 
NTFY_BASE_URL = "https://ntfy.sh/"

def send_notification(title: str, message: str, priority: int = 3, tags: str = None):
    """
    Sends a push notification using ntfy.sh.
    Priority: 1=Min, 3=Default, 5=Max (Emergency)
    """
    if NTFY_TOPIC == "YOUR_SECRET_HYDR_AI_TOPIC":
        print("NTFY topic not configured. Notification skipped.")
        return

    url = f"{NTFY_BASE_URL}{NTFY_TOPIC}"
    
    # Use headers to send rich notification data
    headers = {
        "Title": title,
        "Priority": str(priority),
        "Tags": tags if tags else ""
    }
    
    try:
        # The message body is the content
        response = requests.post(url, data=message.encode('utf-8'), headers=headers) #
        response.raise_for_status()
        print(f"Notification sent to {NTFY_TOPIC}. Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"NTFY API Error: {e}")