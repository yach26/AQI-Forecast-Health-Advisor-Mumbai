"""
LLM utility module for the Mumbai AQI Forecast & AI Health Advisor.

Uses the Groq Cloud API (Llama-3.3-70b-versatile) to generate
actionable health recommendations based on the forecasted AQI.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Fallback health advice (used when the LLM is unavailable)
# ---------------------------------------------------------------------------
_FALLBACK_ADVICE = {
    "Good": (
        "**General Public:**\n"
        "- Air quality is satisfactory. Enjoy outdoor activities.\n"
        "- No special precautions needed.\n\n"
        "**Sensitive Groups:**\n"
        "- No restrictions. Normal outdoor activities are safe."
    ),
    "Moderate": (
        "**General Public:**\n"
        "- Air quality is acceptable. Consider reducing prolonged outdoor exertion.\n\n"
        "**Sensitive Groups:**\n"
        "- People with respiratory conditions should monitor symptoms.\n"
        "- Limit extended outdoor exercise if you feel discomfort."
    ),
    "Unhealthy for Sensitive Groups": (
        "**General Public:**\n"
        "- Reduce prolonged outdoor exertion.\n"
        "- Keep windows closed during peak pollution hours.\n\n"
        "**Sensitive Groups:**\n"
        "- Children and elderly should limit outdoor exposure.\n"
        "- People with asthma or heart disease should avoid outdoor exercise."
    ),
    "Unhealthy": (
        "**General Public:**\n"
        "- Avoid prolonged outdoor activities.\n"
        "- Consider wearing a mask outdoors.\n\n"
        "**Sensitive Groups:**\n"
        "- Stay indoors as much as possible.\n"
        "- Use air purifiers if available."
    ),
    "Very Unhealthy": (
        "**General Public:**\n"
        "- Stay indoors whenever possible.\n"
        "- Avoid all outdoor physical activity.\n\n"
        "**Sensitive Groups:**\n"
        "- Remain indoors with windows and doors closed.\n"
        "- Seek medical attention if you experience symptoms."
    ),
    "Hazardous": (
        "**General Public:**\n"
        "- Avoid ALL outdoor activities.\n"
        "- Follow emergency health precautions.\n\n"
        "**Sensitive Groups:**\n"
        "- Remain indoors. Use air purifiers.\n"
        "- Seek immediate medical attention if symptomatic."
    ),
}


def _get_fallback_advice(aqi, category):
    """Return pre-written advice when the LLM is unavailable."""
    header = f"*⚠️ AI advisor unavailable — showing standard guidance for **{category}** (AQI {aqi}).*\n\n"
    return header + _FALLBACK_ADVICE.get(category, _FALLBACK_ADVICE["Moderate"])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_health_advice(aqi, category):
    """
    Generate health recommendations via Groq LLM.

    Falls back to pre-written advice if:
        - GROQ_API_KEY is not configured
        - The Groq API call fails for any reason
    """
    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        return _get_fallback_advice(aqi, category)

    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        prompt = f"""You are an air quality health advisor.

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

Keep it concise and practical."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content

    except ImportError:
        return _get_fallback_advice(aqi, category)
    except Exception:
        return _get_fallback_advice(aqi, category)