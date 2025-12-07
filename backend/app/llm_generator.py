# app/llm_generator.py
import os
from dotenv import load_dotenv
import fastapi_poe as fp

# Load environment variables
load_dotenv()
POE_API_KEY = os.getenv("POE_API_KEY")

SYSTEM_PROMPT = (
    "You are an AI specialized in health and hydration. Your task is to provide a concise, "
    "friendly, and actionable hydration recommendation based on the user's data. "
    "Keep the response arouns 30 words. Be encouraging."
)
BOT_NAME = 'GPT-4o-Mini'

# Change the function signature to asynchronous
async def generate_llm_response(hydration_label: str, steps: int, calories: float, weather_data: dict) -> str:
    """
    Generates a personalized hydration recommendation using Poe (GPT-4o-Mini).
    """
    if not POE_API_KEY:
        return f"Prediction: {hydration_label}. Poe API key not set."

    # --- Construct the Prompt (Same logic as before) ---
    weather_desc = weather_data.get("description", "unknown weather conditions")
    temp_f = weather_data.get("temp_f", "N/A")

    user_prompt = (
        f"The user's current hydration level is classified as: '{hydration_label}'. "
        f"Recent activity: {steps} steps and {calories:.1f} active calories burned. "
        f"External conditions: {weather_desc} at {temp_f}°F. "
        "Generate a personalized recommendation considering weather conditions and recent activity to inform water intake."
    )

    try:
        # Assemble the message list using the Poe structure
        message = [
            fp.ProtocolMessage(role="system", content=SYSTEM_PROMPT),
            fp.ProtocolMessage(role="user", content=user_prompt)
        ]
        
        full_response = ""
        # Use the asynchronous generator for the response
        async for partial in fp.get_bot_response(
            messages=message,
            bot_name=BOT_NAME,
            api_key=POE_API_KEY
        ):
            full_response += partial.text
            
        return full_response.strip()

    except Exception as e:
        print(f"POE API Error: {e}")
        return f"Prediction: {hydration_label}. Error generating personalized advice."