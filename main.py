"""
Weekly To-Do Backend
--------------------
FastAPI server that:
1. Authenticates with Google Drive via a service account
2. Fetches meeting notes for a given date range
3. Sends them to Claude to extract action items
4. Returns structured JSON to the frontend
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional
import os
import json
import anthropic
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

app = FastAPI(title="Weekly Todo Generator API")

# ─── CORS: allow your Claude artifact / any frontend ──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── CONFIG (set these as environment variables on Railway/Render) ─────────────
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
# Google service account credentials JSON (paste the entire JSON as an env var)
GOOGLE_CREDENTIALS_JSON = os.environ["GOOGLE_CREDENTIALS_JSON"]

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/documents.readonly",
]


# ─── MODELS ───────────────────────────────────────────────────────────────────
class TodoResponse(BaseModel):
    meetings_found: int
    date_range: str
    todo_markdown: str
    sources: list[dict]


# ─── GOOGLE DRIVE HELPERS ─────────────────────────────────────────────────────
def get_google_services():
    creds_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    drive = build("drive", "v3", credentials=creds)
    docs  = build("docs",  "v1", credentials=creds)
    return drive, docs


def fetch_meeting_notes(days_back: int) -> list[dict]:
    drive, docs_service = get_google_services()
    cutoff = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%S")

    # Two passes: Gemini notes + general meeting docs
    queries = [
        f"(name contains 'Notes by Gemini' or fullText contains 'meeting notes') and modifiedTime > '{cutoff}' and mimeType = 'application/vnd.google-apps.document' and trashed = false",
        f"(name contains 'sync' or name contains 'standup' or name contains 'meeting') and modifiedTime > '{cutoff}' and mimeType = 'application/vnd.google-apps.document' and trashed = false",
    ]

    seen_ids = set()
    results = []

    for q in queries:
        try:
            resp = drive.files().list(
                q=q,
                fields="files(id, name, webViewLink, modifiedTime)",
                pageSize=30,
                orderBy="modifiedTime desc",
            ).execute()

            for f in resp.get("files", []):
                if f["id"] in seen_ids:
                    continue
                seen_ids.add(f["id"])

                # Fetch document text via Docs API
                try:
                    doc = docs_service.documents().get(documentId=f["id"]).execute()
                    text = extract_text_from_doc(doc)
                    if len(text.strip()) < 100:
                        continue
                    results.append({
                        "id":    f["id"],
                        "title": f["name"],
                        "url":   f["webViewLink"],
                        "date":  f["modifiedTime"][:10],
                        "text":  text[:8000],  # cap per doc
                    })
                except HttpError:
                    pass  # skip unreadable docs

        except HttpError as e:
            raise HTTPException(status_code=500, detail=f"Drive API error: {e}")

    return results


def extract_text_from_doc(doc: dict) -> str:
    """Extracts plain text from a Google Docs API response."""
    text_parts = []
    body = doc.get("body", {})
    for element in body.get("content", []):
        para = element.get("paragraph")
        if not para:
            continue
        for pe in para.get("elements", []):
            tr = pe.get("textRun")
            if tr:
                text_parts.append(tr.get("content", ""))
    return "".join(text_parts)


# ─── CLAUDE HELPER ────────────────────────────────────────────────────────────
def call_claude(meeting_notes: list[dict], date_range_label: str) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    notes_text = "\n\n".join(
        f"--- {n['title']} ({n['date']}) ---\n{n['text']}"
        for n in meeting_notes
    )

    prompt = f"""You are a highly organized executive assistant. Below are notes from {len(meeting_notes)} meeting(s) covering {date_range_label}.

Extract all action items, follow-ups, commitments, and decisions that require action.

Format your response as follows:
1. One clearly labeled section per meeting with a bullet list of action items
2. Include owner, deadline, and context for each item where available
3. End with a "🔥 Top Priorities" section listing the 5 most urgent cross-meeting items

Be specific and actionable. Use markdown formatting.

Meeting Notes:
{notes_text}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "Weekly Todo Generator"}


@app.get("/generate", response_model=TodoResponse)
def generate_todos(days: int = Query(default=7, ge=1, le=90)):
    """
    Generate a to-do list from meeting notes.
    ?days=7   → last 7 days
    ?days=14  → last 2 weeks
    ?days=30  → last month
    """
    meeting_notes = fetch_meeting_notes(days_back=days)

    if not meeting_notes:
        return TodoResponse(
            meetings_found=0,
            date_range=f"last {days} days",
            todo_markdown="No meeting notes found for this period.",
            sources=[],
        )

    end_date   = datetime.utcnow().strftime("%b %d, %Y")
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%b %d, %Y")
    date_range = f"{start_date} – {end_date}"

    todo_markdown = call_claude(meeting_notes, date_range)

    sources = [{"title": n["title"], "url": n["url"], "date": n["date"]} for n in meeting_notes]

    return TodoResponse(
        meetings_found=len(meeting_notes),
        date_range=date_range,
        todo_markdown=todo_markdown,
        sources=sources,
    )
