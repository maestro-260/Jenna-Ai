# JENNA AI API Reference

## Core Endpoints

### `POST /api/process`
Process user input and generate response

**Parameters:**
```json
{
  "audio": "base64_encoded_audio",
  "session_id": "string"
}