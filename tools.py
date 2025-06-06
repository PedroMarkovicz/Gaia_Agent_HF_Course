# Enhanced Tool Implementations for GaiaAgent

import os
import re
import json
import time
import random
import httpx  # For robust HTTP requests
from urllib.parse import urljoin
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from io import StringIO

# Web and API integrations
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader

# Wikipedia and Knowledge Base
import wikipediaapi
import mwclient
from SPARQLWrapper import SPARQLWrapper, JSON

# Multimedia and Transcripts
from youtube_transcript_api import YouTubeTranscriptApi
import whisper

# Structured Data and Math
import sympy
import pandas as pd
import numpy as np

# Image and Chess Analysis
import chess
import chess.engine
import cv2
import pytesseract

# Language and Ontology
import spacy
from nltk.corpus import wordnet
import langdetect
from bs4 import BeautifulSoup

# --- Constants ---
# Base URL for the GAIA evaluation API
DEFAULT_EVAL_API_URL = "https://agents-course-unit4-scoring.hf.space"
EVAL_API_URL = os.getenv("EVAL_API_URL", DEFAULT_EVAL_API_URL)

# Directory for downloaded files
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "/tmp/gaia_agent_downloaded_files")

# API Keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Initialize NLP models
try:
    nlp = spacy.load("en_core_web_sm")
except:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize Whisper model (using base model for efficiency)
whisper_model = whisper.load_model("base")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
]


# --- Helper Functions ---
def _ensure_download_dir() -> Path:
    """Ensures the download directory exists and returns the Path object."""
    path = Path(DOWNLOAD_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _robust_get_request(
    url: str, timeout: int = 20, retries: int = 3, backoff_factor: float = 0.5
) -> httpx.Response:
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    with httpx.Client(
        headers=headers, timeout=timeout, follow_redirects=True
    ) as client:
        for attempt in range(retries):
            try:
                response = client.get(url)
                response.raise_for_status()
                return response
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                print(
                    f"[Tool Helper] Request failed (attempt {attempt + 1}/{retries}): {e}"
                )
                if attempt < retries - 1:
                    time.sleep(backoff_factor * (2**attempt))
                else:
                    raise e


def _format_search_results(docs: list) -> str:
    """Formats a list of Langchain documents into a string for the agent."""
    if not docs:
        return "No results found."
    formatted = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata.get("source", "unknown")}" title="{doc.metadata.get("title", "N/A")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )
    return formatted


def _ocr_from_image_data(image_data: np.ndarray, ocr_preproc_mode: int = 2) -> str:
    gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    if ocr_preproc_mode == 2:
        processed_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        _, processed_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return pytesseract.image_to_string(processed_img).strip()


# --- Tool Functions ---


def get_web_content(url: str) -> str:
    """Fetches and parses text content from a URL using robust requests."""
    try:
        response = _robust_get_request(url)
        content_type = response.headers.get("content-type", "").lower()

        if "html" in content_type or "text" in content_type:
            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(["script", "style", "nav", "footer", "aside"]):
                script.extract()
            text = soup.get_text(separator="\n", strip=True)
            print(
                f"[Tool:get_web_content] Successfully fetched and parsed text from: {url}"
            )
            return text[:8000]
        else:
            return f"Error: Non-text/HTML content type ({content_type}) at URL."
    except Exception as e:
        return f"Error fetching URL {url}: {e}"


def web_search(query: str) -> str:
    """Search the web using Tavily for a query."""
    try:
        if not TAVILY_API_KEY:
            return "Error: Tavily API key not configured."
        search = TavilySearchResults(max_results=3, tavily_api_key=TAVILY_API_KEY)
        results = search.invoke(query)
        return str(results)
    except Exception as e:
        return f"Error during web search: {e}"


def wiki_search(query: str) -> str:
    """Search Wikipedia for a query."""
    try:
        loader = WikipediaLoader(query=query, load_max_docs=2, lang="en")
        docs = loader.load()
        return _format_search_results(docs)
    except Exception as e:
        return f"Error during Wikipedia search: {e}"


def arvix_search(query: str) -> str:
    """Search ArXiv for a query."""
    try:
        loader = ArxivLoader(query=query, load_max_docs=2)
        docs = loader.load()
        return _format_search_results(docs)
    except Exception as e:
        return f"Error during ArXiv search: {e}"


def download_gaia_file(task_id: str) -> str:
    """(Placeholder) Downloads a file for a GAIA task and returns its local path."""
    try:
        download_dir = _ensure_download_dir()
        # In a real scenario, an API call would be made to get the actual filename.
        # This is a placeholder for robust implementation.
        metadata_url = urljoin(EVAL_API_URL, f"/api/tasks/{task_id}/artifacts")
        print(
            f"[Tool:download_gaia_file] Simulating metadata fetch from {metadata_url}"
        )

        # For this review, we'll assume the filename is the task_id for simplicity.
        filename = f"{task_id}.dat"
        file_path = download_dir / filename

        if not file_path.exists():
            print(
                f"[Tool:download_gaia_file] Simulating download for task {task_id} to {file_path}"
            )
            file_path.touch()  # Create an empty file to simulate download

        return str(file_path)
    except Exception as e:
        return f"Error downloading file for task {task_id}: {e}"


def analyze_file_content(file_path: str, analysis_type: str = "summary") -> str:
    """(Placeholder) Analyzes the content of a local file.
    Args:
        file_path: The local path to the file to analyze.
        analysis_type: The type of analysis to perform (e.g., 'summary', 'word_count').
    Returns: The result of the analysis.
    """
    try:
        path = Path(file_path)
        if not path.is_file():
            return f"Error: File not found at {file_path}"

        content = path.read_text(encoding="utf-8", errors="ignore")

        if analysis_type == "summary":
            # In a real implementation, this could be a call to an LLM.
            return f"File content summary (first 200 chars): {content[:200]}..."
        elif analysis_type == "word_count":
            return str(len(content.split()))
        else:
            return f"Error: Unsupported analysis type '{analysis_type}'"
    except Exception as e:
        return f"Error analyzing file {file_path}: {e}"


def scrape_webpage(url: str, mode: str = "text", selector: Optional[str] = None) -> str:
    """Scrapes a webpage for either text or tables, now with more robust requests.
    Args:
        url: URL to scrape
        mode: 'text' or 'table'.
        selector: Optional CSS selector to target specific tables or text elements.
    Returns: Scraped content as text or CSV string.
    """
    try:
        response = _robust_get_request(url)
        soup = BeautifulSoup(response.text, "html.parser")

        if mode == "table":
            tables = soup.select(selector) if selector else soup.find_all("table")
            if not tables:
                return "Error: No tables found on the page or with the given selector."
            df = pd.read_html(str(tables[0]))[0]
            return df.to_csv(index=False)
        else:
            if selector:
                elements = soup.select(selector)
                return "\n".join(elem.get_text(strip=True) for elem in elements)
            for script in soup(["script", "style", "nav", "footer", "aside"]):
                script.extract()
            return soup.get_text(strip=True)
    except Exception as e:
        return f"Error scraping webpage {url}: {e}"


def analyze_image(
    image: Union[str, np.ndarray], operation: str = "ocr", ocr_preproc_mode: int = 2
) -> str:
    """Analyzes an image from a path or numpy array with enhanced pre-processing.
    Args:
        image: Path to the image file or a numpy array of the image.
        operation: 'ocr' for text extraction.
        ocr_preproc_mode: 1 for basic grayscale, 2 for adaptive thresholding.
    Returns: Analysis result or an error message.
    """
    try:
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image

        if img is None:
            return "Error: Image not found or could not be read."

        if operation == "ocr":
            text = _ocr_from_image_data(img, ocr_preproc_mode=ocr_preproc_mode)
            return text if text else "No text detected."
        else:
            return f"Unsupported operation: {operation}"
    except Exception as e:
        return f"Error analyzing image: {e}"


def process_video_frames(video_path: str, frame_interval: int = 30) -> str:
    """Processes a video file by extracting frames and running OCR on them.
    Args:
        video_path: Path to the video file.
        frame_interval: The interval (in frames) at which to process a frame.
    Returns: A JSON string with frame numbers and detected text.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error: Could not open video file."

        results = {}
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                text = analyze_image(frame, operation="ocr")
                if text and len(text) > 5:
                    results[frame_count] = text
            frame_count += 1

        cap.release()
        return (
            json.dumps(results, indent=2)
            if results
            else "No text detected in video frames."
        )
    except Exception as e:
        return f"Error processing video: {e}"


def get_sports_schedule(sport: str, date: str = None) -> str:
    """(Domain Specific) Gets the schedule for a sport from ESPN.
    NOTE: This tool is brittle and may break if ESPN changes its website structure.
    Args:
        sport: The sport to get schedule for (e.g., 'nba', 'nfl', 'mlb').
        date: The date in YYYYMMDD format. If None, gets today's schedule.
    Returns: A JSON string of the schedule, or an error.
    """
    try:
        base_url = f"https://www.espn.com/{sport}/schedule"
        url = f"{base_url}/_/date/{date}" if date else base_url
        response = _robust_get_request(url)
        soup = BeautifulSoup(response.text, "html.parser")

        schedule_tables = soup.find_all("div", class_="ScheduleTables")
        if not schedule_tables:
            return "No schedule found for this sport or date."

        games = []
        for table in schedule_tables:
            rows = table.find_all("tr", class_="Table__TR")
            for row in rows:
                cols = row.find_all("td")
                if len(cols) > 2:
                    teams = [a.text for a in cols[0].find_all("a", class_="AnchorLink")]
                    if len(teams) >= 2:
                        games.append(
                            {
                                "matchup": f"{teams[0]} at {teams[1]}",
                                "time": cols[1].text,
                            }
                        )
        return json.dumps(games, indent=2) if games else "No games found."
    except Exception as e:
        return f"Error getting sports schedule: {e}"


def cross_reference_search(query: str) -> str:
    """Performs a search on multiple platforms (web and Wikipedia) and combines the results.
    Args:
        query: The search query.
    Returns: A JSON object with summarized info from multiple sources.
    """
    try:
        web_results = web_search(query)
        wiki_results = wiki_search(query)

        summary = {
            "query": query,
            "web_summary": web_results
            if not web_results.startswith("Error")
            else "Web search failed.",
            "wiki_summary": wiki_results
            if not wiki_results.startswith("Error")
            else "Wiki search failed.",
        }
        # A more advanced implementation could use an LLM to synthesize these results.
        return json.dumps(summary, indent=2)
    except Exception as e:
        return f"Error during cross-reference search: {e}"


# --- Math Tools ---
def multiply(a: int | float, b: int | float) -> int | float:
    """Multiply two numbers (integers or floats).
    Args: a: first number, b: second number
    Returns: The product of a and b.
    """
    print(f"[Tool:multiply] Calculating {a} * {b}")
    return a * b


def add(a: int | float, b: int | float) -> int | float:
    """Add two numbers (integers or floats).
    Args: a: first number, b: second number
    Returns: The sum of a and b.
    """
    print(f"[Tool:add] Calculating {a} + {b}")
    return a + b


def subtract(a: int | float, b: int | float) -> int | float:
    """Subtract second number from first (integers or floats).
    Args: a: first number, b: second number
    Returns: The result of a - b.
    """
    print(f"[Tool:subtract] Calculating {a} - {b}")
    return a - b


def divide(a: int | float, b: int | float) -> float | str:
    """Divide first number by second (integers or floats).
    Args: a: first number, b: second number
    Returns: The result of a / b as float, or an error string.
    """
    print(f"[Tool:divide] Calculating {a} / {b}")
    if b == 0:
        print("[Tool:divide] Error: Division by zero.")
        return "Error: Cannot divide by zero."
    try:
        return float(a) / float(b)
    except Exception as e:
        return f"Error during division: {e}"


def modulus(a: int | float, b: int | float) -> int | float | str:
    """Get the modulus (remainder) of a divided by b.
    Args: a: first number, b: second number
    Returns: The result of a % b, or an error string.
    """
    print(f"[Tool:modulus] Calculating {a} % {b}")
    if b == 0:
        print("[Tool:modulus] Error: Modulus by zero.")
        return "Error: Cannot take modulus by zero."
    try:
        return a % b
    except Exception as e:
        return f"Error during modulus: {e}"


# --- New Tool Functions ---


# Wikipedia and Knowledge Base Tools
def wiki_advanced_search(query: str, include_revisions: bool = False) -> str:
    """Advanced Wikipedia search using multiple APIs for comprehensive results.

    Args:
        query: Search query
        include_revisions: Whether to include revision history

    Returns:
        Formatted string with comprehensive Wikipedia information
    """
    wiki_wiki = wikipediaapi.Wikipedia("en")
    page = wiki_wiki.page(query)

    if not page.exists():
        return "Error: Page not found"

    result = {
        "title": page.title,
        "summary": page.summary,
        "url": page.fullurl,
        "sections": [sect.title for sect in page.sections],
    }

    if include_revisions:
        site = mwclient.Site("en.wikipedia.org")
        page_mc = site.pages[query]
        revisions = list(page_mc.revisions(limit=5))
        result["recent_revisions"] = revisions

    return json.dumps(result, indent=2)


def wikidata_query(query: str) -> str:
    """Execute SPARQL query on Wikidata.

    Args:
        query: SPARQL query string

    Returns:
        Query results in JSON format
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error executing SPARQL query: {str(e)}"


# Multimedia Tools
def get_youtube_transcript(video_id: str) -> str:
    """Get transcript from YouTube video.

    Args:
        video_id: YouTube video ID

    Returns:
        Video transcript or error message
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return json.dumps(transcript, indent=2)
    except Exception as e:
        return f"Error getting transcript: {str(e)}"


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using Whisper.

    Args:
        audio_path: Path to audio file

    Returns:
        Transcription text or error message
    """
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"


# Math and Symbolic Tools
def solve_equation(equation: str) -> str:
    """Solve mathematical equation using SymPy.

    Args:
        equation: Mathematical equation as string

    Returns:
        Solution or error message
    """
    try:
        result = sympy.solve(equation)
        return str(result)
    except Exception as e:
        return f"Error solving equation: {str(e)}"


def analyze_data(data: str, operation: str) -> str:
    """Analyze data using pandas.

    Args:
        data: CSV string or path to CSV file
        operation: Operation to perform (mean, median, sum, describe, etc.)

    Returns:
        Analysis result or error message
    """
    try:
        df = pd.read_csv(data) if os.path.exists(data) else pd.read_csv(StringIO(data))

        operations = {
            "mean": df.mean,
            "median": df.median,
            "sum": df.sum,
            "describe": df.describe,
        }

        if operation in operations:
            result = operations[operation]()
            # For series or dataframes, convert to a string representation
            if isinstance(result, (pd.Series, pd.DataFrame)):
                return result.to_string()
            return str(result)
        else:
            return f"Unsupported operation: {operation}. Supported operations are: {list(operations.keys())}"
    except Exception as e:
        return f"Error analyzing data: {str(e)}"


# Chess Tools
def analyze_chess_position(fen: str) -> str:
    """Analyze chess position using python-chess and optional Stockfish.

    Args:
        fen: Chess position in FEN notation

    Returns:
        Analysis result or error message
    """
    try:
        board = chess.Board(fen)
        result = {
            "legal_moves": [str(move) for move in board.legal_moves],
            "is_check": board.is_check(),
            "is_checkmate": board.is_checkmate(),
            "is_stalemate": board.is_stalemate(),
        }

        # Try to use Stockfish if available
        try:
            engine = chess.engine.SimpleEngine.popen_uci("stockfish")
            analysis = engine.analyse(board, chess.engine.Limit(time=0.1))
            result["stockfish_score"] = str(analysis["score"])
            engine.quit()
        except:
            result["stockfish_score"] = "Stockfish not available"

        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error analyzing chess position: {str(e)}"


# Language Analysis Tools
def analyze_text(text: str, operation: str) -> str:
    """Analyze text using spaCy and NLTK.

    Args:
        text: Text to analyze
        operation: Operation to perform (ner, pos, sentiment, etc.)

    Returns:
        Analysis result or error message
    """
    try:
        doc = nlp(text)
        if operation == "ner":
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            return json.dumps(entities)
        elif operation == "pos":
            pos_tags = [(token.text, token.pos_) for token in doc]
            return json.dumps(pos_tags)
        elif operation == "detect_language":
            return langdetect.detect(text)
        else:
            return f"Unsupported operation: {operation}"
    except Exception as e:
        return f"Error analyzing text: {str(e)}"


def get_word_info(word: str) -> str:
    """Get word information from WordNet.

    Args:
        word: Word to look up

    Returns:
        Word information or error message
    """
    try:
        synsets = wordnet.synsets(word)
        if not synsets:
            return f"No information found for word: {word}"

        result = {
            "word": word,
            "definitions": [syn.definition() for syn in synsets],
            "examples": [ex for syn in synsets for ex in syn.examples()],
            "synonyms": list(
                set(lemma.name() for syn in synsets for lemma in syn.lemmas())
            ),
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting word information: {str(e)}"


# --- Local Logic Tools ---
def string_operation(text: str, operation: str) -> str:
    """Performs a specified operation on a string.
    Args:
        text: The input string.
        operation: The operation to perform. Supported: 'reverse', 'uppercase', 'lowercase', 'length'.
    Returns: The modified string or an error message.
    """
    if operation == "reverse":
        return text[::-1]
    elif operation == "uppercase":
        return text.upper()
    elif operation == "lowercase":
        return text.lower()
    elif operation == "length":
        return str(len(text))
    else:
        return f"Error: Unsupported string operation '{operation}'."


def get_antonym(word: str) -> str:
    """Finds antonyms for a given word using WordNet.
    Args:
        word: The word to find antonyms for.
    Returns: A comma-separated string of antonyms, or a message if none are found.
    """
    antonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.extend([ant.name() for ant in lemma.antonyms()])
    if not antonyms:
        return f"No antonyms found for '{word}'."
    return ",".join(list(set(antonyms)))


def match_pattern(text: str, pattern: str) -> str:
    """Finds all occurrences of a regex pattern in a text.
    Args:
        text: The text to search within.
        pattern: The regular expression pattern to match.
    Returns: A JSON list of all matches, or an empty list if no matches are found.
    """
    try:
        matches = re.findall(pattern, text)
        return json.dumps(matches)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"


# --- Updated Tool Dictionary ---
AVAILABLE_TOOLS = {
    # Existing Web Tools
    "get_web_content": get_web_content,
    "web_search": web_search,
    "wiki_search": wiki_search,
    "arvix_search": arvix_search,
    # Existing Math Tools
    "multiply": multiply,
    "add": add,
    "subtract": subtract,
    "divide": divide,
    "modulus": modulus,
    # New Wikipedia and Knowledge Base Tools
    "wiki_advanced_search": wiki_advanced_search,
    "wikidata_query": wikidata_query,
    # Multimedia Tools
    "get_youtube_transcript": get_youtube_transcript,
    "transcribe_audio": transcribe_audio,
    # Math and Data Tools
    "solve_equation": solve_equation,
    "analyze_data": analyze_data,
    # File Tools
    "download_gaia_file": download_gaia_file,
    "analyze_file_content": analyze_file_content,
    # Chess Tools
    "analyze_chess_position": analyze_chess_position,
    # Image Analysis Tools
    "analyze_image": analyze_image,
    # Language Analysis Tools
    "analyze_text": analyze_text,
    "get_word_info": get_word_info,
    # Web Scraping Tools
    "scrape_webpage": scrape_webpage,
    "process_video_frames": process_video_frames,
    "get_sports_schedule": get_sports_schedule,
    "cross_reference_search": cross_reference_search,
    # Local Logic Tools
    "string_operation": string_operation,
    "get_antonym": get_antonym,
    "match_pattern": match_pattern,
}

# --- Test Block (Optional) ---
if __name__ == "__main__":
    print("--- Testing Simplified Tools ---")

    # Test Math
    print("\nTesting Math Tools...")
    print(f"add(5, 3) = {add(5, 3)}")
    print(f"multiply(5, 3) = {multiply(5, 3)}")
    print(f"divide(6, 3) = {divide(6, 3)}")
    print(f"divide(5, 0) = {divide(5, 0)}")

    # Test Web Content
    print("\nTesting Web Content...")
    test_url = "https://example.com"
    content = get_web_content(test_url)
    print(f"Content from {test_url}:\n{content[:200]}...")

    # Test Search Tools
    print("\nTesting Search Tools...")
    wiki_results = wiki_search("Large Language Model")
    print(f"\nWikipedia Search Results:\n{wiki_results[:500]}...")

    # Test Tavily Search
    if TAVILY_API_KEY:
        print("\nTesting Tavily Search...")
        search_results = web_search("Latest developments in AI")
        print(f"Tavily Search Results: {search_results[:500]}...")
    else:
        print("\nSkipping Tavily Search (API key not set)")

    print("\n--- Tool Testing Complete ---")

    # Print available tools
    print("\nAvailable Tools:")
    for tool_name in sorted(AVAILABLE_TOOLS.keys()):
        print(f"- {tool_name}")
