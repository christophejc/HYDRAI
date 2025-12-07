# app/llm_generator.py

import os
from google.cloud import aiplatform

# Initialize Vertex AI client
# The client automatically uses the Service Account credentials
# provided by the Cloud Run environment.
try:
    aiplatform.init(project=os.getenv("hydr-ai"), location="us-central1")
except Exception as e:
    print(f"Vertex AI initialization failed (running locally?): {e}")

# Define the model to use (e.g., Gemini 2.5 Flash)
MODEL_NAME = "gemini-2.5-flash"

def generate_llm_response(hydration_label: str, steps: int, calories: float, weather_data: dict) -> str:
    """
    Generates a personalized hydration recommendation using Google's Vertex AI (Gemini).
    """
    if not aiplatform.initializer.global_config.project:
        return f"Prediction: {hydration_label}. LLM environment not initialized."

    # --- Construct the Prompt ---
    weather_desc = weather_data.get("description", "unknown weather conditions")
    temp_f = weather_data.get("temp_f", "N/A")

    system_prompt = (
        "You are an AI specialized in health and hydration. Your task is to provide a concise, "
        "friendly, and actionable hydration recommendation based on the user's data. "
        "Keep the response under 50 words. Be encouraging."
    )

    user_prompt = (
        f"The user's current hydration level is classified as: '{hydration_label}'. "
        f"Recent activity: {steps} steps and {calories:.1f} active calories burned. "
        f"External conditions: {weather_desc} at {temp_f}°F. "
        "Generate a personalized, encouraging recommendation."
    )

    try:
        model = aiplatform.get_model_registry().get_model(MODEL_NAME)
        
        # Use the generate_content method
        response = model.generate_content(
            contents=[
                {"role": "user", "parts": [{"text": user_prompt}]},
            ],
            system_instruction=system_prompt,
            config=aiplatform.types.GenerateContentConfig(
                max_output_tokens=100,
                temperature=0.7
            )
        )
        
        llm_text = response.text.strip()
        return llm_text

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return f"Prediction: {hydration_label}. Error generating personalized advice."