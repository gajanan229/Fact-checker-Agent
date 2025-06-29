# VeriFYP: Your AI Research Assistant for the "For You Page"


**VeriFYP is an agentic fact-checking system designed to combat misinformation on TikTok. It empowers users to investigate questionable content and generate constructive, evidence-based responses with the help of a semi-autonomous AI co-pilot.**

---

## The Problem

The TikTok "For You Page" is an incredible source of entertainment and information, but it can also be a breeding ground for misleading or dangerous claims. Fact-checking a 30-second video is a time-consuming process that most users don't have the time for. VeriFYP was built to bridge this gap, providing a fast, transparent, and scalable tool to verify information and foster healthier online discussions.

## How It Works

VeriFYP streamlines the entire fact-checking process into a simple, user-friendly workflow:

1.  **Submit a Link**: A user provides a link to a TikTok video they wish to investigate.
2.  **Data Extraction**: The system automatically extracts a full audio transcript and any visible on-screen text from the video.
3.  **AI Investigation**: A powerful multi-step AI agent, built with LangGraph, analyzes the content to identify verifiable factual claims. It then cross-references these claims against a curated list of trusted sources to gather evidence.
4.  **Response Generation**: The agent drafts a neutral, constructive, and well-sourced comment based on its findings. This response is designed to educate, providing sources to either support or correct the claims.
5.  **Review & Deploy**: The final, polished response is presented to the user in the Analyst's Workbench. They have the final say and can copy the text to post directly into the TikTok comments, armed with facts.

## Core Features

-   **Agentic Workflow**: Utilizes a multi-step agentic system where one AI agent drafts a response and a second "Red Team" agent critiques it for accuracy and neutrality.
-   **Automated Content Ingestion**: Seamlessly processes TikTok videos to extract transcripts and OCR text.
-   **Evidence-Based Responses**: All generated responses are backed by sources found during the investigation phase.
-   **Real-Time Progress**: The frontend provides a real-time view of the agent's progress as it moves through each stage of the analysis.
-   **User-Centric Design**: A clean, intuitive interface that makes the complex process of fact-checking accessible to everyone.

## Tech Stack

-   **Backend**: Python, Flask, LangChain, LangGraph, Google Gemini
-   **Frontend**: React, TypeScript, Vite, Tailwind CSS, Framer Motion
-   **External Services**: Apify (for TikTok data), Tavily (for web search)

## Local Setup & Usage

To run the VeriFYP prototype locally:

### 1. Backend Setup

```bash
# Navigate to the backend directory
cd Backend-Agent

# Create and activate a Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Create a .env file and add your API keys (see .env.example)
# Required keys: GOOGLE_API_KEY, APIFY_API_KEY, TAVILY_API_KEY

# Run the server
python main.py
```

### 2. Frontend Setup

```bash
# Navigate to the frontend directory
cd Frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

The application will be accessible at `http://localhost:5173`.
