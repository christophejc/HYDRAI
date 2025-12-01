import requests
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="app/.env")  # reads .env file

def get_weather(lat, lon):
  API_KEY = os.getenv("API_KEY")
  print("Using API_KEY:", os.getenv("API_KEY"))  # <- add this
  url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={API_KEY}"
  response = requests.get(url)
  print("Raw weather API response:", response.text)  # <- add this
  data = response.json()
  response.close()
  #temp = data["main"]["temp"]
  #description = data["weather"][0]["description"]
  return data