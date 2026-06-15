from groq import Groq
import streamlit as st

def generate_health_advice(aqi, category):

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        return "GROQ_API_KEY not found in Streamlit secrets."

    client = Groq(api_key=api_key)

    prompt = f"""
You are an air quality health advisor.

Mumbai AQI forecast:
AQI: {aqi}
Category: {category}

Provide advice in this format:

General Public:
- bullet point
- bullet point

Sensitive Groups:
- bullet point
- bullet point

Keep it concise and practical.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content