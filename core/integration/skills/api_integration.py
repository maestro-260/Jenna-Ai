import httpx  # Ensure the httpx library is installed.
from typing import Dict
from utils.config_loader import get_config, cached_config
import asyncio
from google_auth_oauthlib.flow import InstalledAppFlow
# Ensure google-auth-oauthlib.
from google.oauth2.credentials import Credentials
# Ensure google-auth is installed.
from googleapiclient.discovery import build
# Ensure google-api-python-client.
import os
import logging
from datetime import datetime
import ollama  # Ensure ollama is installed.


class APIManager:
    def __init__(self):
        self.client = httpx.AsyncClient()
        self.api_urls = get_config("api_urls.yaml")
        self.services = {
            "calendar": self._handle_google_calendar,
            "email": self._handle_email_service,
            "weather": self._handle_weather_api,
        }
        self.credentials = self._load_or_generate_credentials()
        self.calendar_service = build(
            "calendar", "v3", credentials=self.credentials
        )
        self.logger = logging.getLogger(__name__)

    async def execute(self, service: str, params: Dict) -> Dict:
        handler = self.services.get(service)
        if not handler:
            return {
                "status": "error", "message": f"Service '{service}' not found."
                }
        return await handler(params)

    async def _call_ollama(self, prompt: str) -> str:
        response = await asyncio.to_thread(
            ollama.chat,
            model="mistral",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    def _load_or_generate_credentials(self):
        """Load or generate Google Calendar credentials locally."""
        token_path = "token.json"
        scopes = ["https://www.googleapis.com/auth/calendar"]

        if os.path.exists(token_path):
            # Load existing credentials
            creds = Credentials.from_authorized_user_file(token_path, scopes)
        else:
            # Generate new credentials
            flow = InstalledAppFlow.from_client_secrets_file(
                "client_secrets.json", scopes
            )
            creds = flow.run_local_server(port=0)
            with open(token_path, "w") as token_file:
                token_file.write(creds.to_json())

        if creds.expired and creds.refresh_token:
            creds.refresh(httpx.Request())
            with open(token_path, "w") as token_file:
                token_file.write(creds.to_json())

        return creds

    async def _handle_google_calendar(self, params: Dict) -> Dict:
        """Handle Google Calendar operations and integrate with Ollama."""
        action = params.get("action")
        try:
            if action == "list_events":
                params.get("time_min", datetime.utcnow().isoformat() + "Z")

            elif action == "create_event":
                event = {
                    "summary": params["title"],
                    "start": {
                        "dateTime": params["start_time"],
                        "timeZone": "UTC"
                    },
                    "end": {
                        "dateTime": params["end_time"],
                        "timeZone": "UTC"
                    }
                }
                event_result = await asyncio.to_thread(
                    self.calendar_service.events().insert(
                        calendarId="primary", body=event
                    ).execute
                )
                prompt = (
                    f"I’ve created an event titled '{params['title']}' "
                    f"from {params['start_time']} to {params['end_time']}."
                )
                ollama_response = await self._call_ollama(prompt)
                return {
                    "status": "success",
                    "message": ollama_response,
                    "data": event_result
                }

            elif action == "list_events":
                time_min = params.get("time_min",
                                      "2025-04-03T00:00:00Z")
                events_result = await asyncio.to_thread(
                    self.calendar_service.events().list(
                        calendarId="primary",
                        timeMin=time_min,
                        maxResults=10,
                        singleEvents=True,
                        orderBy="startTime"
                    ).execute
                )
                events = events_result.get("items", [])
                event_list = "\n".join(
                    [
                        f"- {e['summary']} at "
                        f"{e['start'].get('dateTime', e['start'].get('date'))}"
                        for e in events
                    ]
                )
                prompt = (
                    f"Your upcoming events are:\n"
                    f"{event_list if events else 'No events found.'}"
                )
                ollama_response = await self._call_ollama(prompt)
                return {
                    "status": "success",
                    "message": ollama_response,
                    "data": events_result
                }

            elif action == "delete_event":
                event_id = params["event_id"]
                await asyncio.to_thread(
                    self.calendar_service.events().delete(
                        calendarId="primary", eventId=event_id
                    ).execute
                )
                prompt = (
                    f"I’ve deleted the event with ID {event_id} from your "
                    "calendar."
                )
                ollama_response = await self._call_ollama(prompt)
                return {"status": "success", "message": ollama_response}

            else:
                return {
                    "status": "error", "message": "Invalid calendar action"}
        except Exception as e:
            self.logger.error(f"Google Calendar error: {e}")
            return {"status": "error", "message": str(e)}

    async def _handle_email_service(self, params: Dict) -> Dict:
        """Enhanced email service with Ollama integration."""
        message = {
            "to": params["recipient"],
            "subject": params["subject"],
            "body": params["content"]
        }
        response = await self.client.post(
            "https://api.emailservice.com/send",
            json=message
        )
        result = {
            "status": response.status_code,
            "message_id": response.json().get("id")
        }
        prompt = (
            f"Sent an email to {params['recipient']} with subject "
            f"'{params['subject']}'."
        )
        ollama_response = await self._call_ollama(prompt)
        return {
            "status": "success",
            "message": ollama_response,
            "data": result
        }

    async def _handle_weather_api(self, params: Dict) -> Dict:
        """Enhanced weather API with Ollama integration."""
        url = (
            f"https://api.weatherapi.com/v1/current.json"
            f"?key={params['key']}&q={params['location']}"
        )
        response = await self.client.get(url=url)
        weather_data = response.json()
        result = {
            "temp_c": weather_data["current"]["temp_c"],
            "condition": weather_data["current"]["condition"]["text"]
        }
        prompt = (
            f"The weather in {params['location']} is "
            f"{result['condition']} with a temperature of "
            f"{result['temp_c']}°C."
        )
        ollama_response = await self._call_ollama(prompt)
        return {
            "status": "success",
            "message": ollama_response,
            "data": result
        }
