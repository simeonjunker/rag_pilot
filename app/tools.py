from langchain.tools import tool
from app.retrieval import vector_db
from app.config import RETRIEVAL_K
import requests

@tool(parse_docstring=True)
def query_wikipedia_db(query: str) -> str:
    """Search Wikipedia articles and return matching chunks.

    Args:
        query: Natural language search query.

    Returns:
        The retrieved article chunks.
    """
    retrieved_docs = vector_db.similarity_search(query, k=RETRIEVAL_K)

    results = []

    for _, doc in enumerate(retrieved_docs, start=1):
        results.append(
            f"# Source: {doc.metadata.get('source', 'unknown')}\n\n"
            f"{doc.page_content}"
        )
        
    return "\n\n---\n\n".join(results)


@tool(parse_docstring=True)
def get_weather(latitude: float, longitude: float) -> str:
    """Get the current weather for a location.

    Args:
        latitude: Latitude of the location.
        longitude: Longitude of the location.

    Returns:
        Current weather information.
    """
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
        },
    )
    response.raise_for_status()

    data = response.json()["current"]

    return (
        f"Temperature: {data['temperature_2m']}°C\n"
        f"Humidity: {data['relative_humidity_2m']}%\n"
        f"Wind: {data['wind_speed_10m']} km/h\n"
        f"Weather code: {data['weather_code']}"
    )