"""
GAIA Agent Evaluation Runner - Hugging Face Space Integration
This module provides a Gradio interface for running and evaluating the GAIA agent.
"""

import os
import sys
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path

import gradio as gr
import requests
import pandas as pd
from dotenv import load_dotenv

# Import agent components
from agent import GaiaAgent
import llm_client  # Import to ensure LLM client is available

# --- Constants and Configuration ---
# Base evaluation API URL
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
EVAL_API_URL = os.getenv("EVAL_API_URL", DEFAULT_API_URL)

# Required environment variables/secrets
REQUIRED_SECRETS = {
    "HF_TOKEN": "Hugging Face API token for model access",
    "TAVILY_API_KEY": "Tavily API key for web search functionality",
}


class GaiaEvaluationRunner:
    """
    Handles the execution and evaluation of the GAIA agent.
    Manages agent initialization, question processing, and result submission.
    """

    def __init__(self):
        """Initialize the evaluation runner with configuration checks."""
        self.agent: Optional[GaiaAgent] = None
        self.space_id = os.getenv("SPACE_ID")
        self.space_host = os.getenv("SPACE_HOST")
        self._check_environment()

    def _check_environment(self) -> None:
        """Verify all required environment variables and configurations."""
        missing_secrets = [
            secret for secret, desc in REQUIRED_SECRETS.items() if not os.getenv(secret)
        ]
        if missing_secrets:
            print("\n⚠️ WARNING: Missing required secrets:")
            for secret in missing_secrets:
                print(f"  - {secret}: {REQUIRED_SECRETS[secret]}")
            print("Some functionality may be limited.\n")

        if self.space_host:
            print(f"✅ Running on Space: https://{self.space_host}.hf.space")
        if self.space_id:
            print(f"✅ Space ID: {self.space_id}")

    def _initialize_agent(self) -> Tuple[bool, str]:
        """
        Initialize the GAIA agent instance.

        Returns:
            Tuple[bool, str]: Success status and message
        """
        try:
            if not self.agent:
                print("[Runner] Initializing GAIA agent...")
                self.agent = GaiaAgent()
                print("[Runner] Agent initialized successfully")
            return True, "Agent initialized successfully"
        except Exception as e:
            error_msg = f"Failed to initialize agent: {str(e)}"
            print(f"[Runner] ERROR: {error_msg}")
            return False, error_msg

    def _fetch_questions(self) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Fetch evaluation questions from the API.

        Returns:
            Tuple[bool, str, List]: Success status, message, and questions data
        """
        try:
            response = requests.get(f"{EVAL_API_URL}/questions", timeout=30)
            response.raise_for_status()
            questions = response.json()

            if not isinstance(questions, list) or not questions:
                return False, "Invalid or empty questions data received", []

            return True, f"Successfully fetched {len(questions)} questions", questions
        except requests.exceptions.RequestException as e:
            return False, f"Error fetching questions: {str(e)}", []
        except Exception as e:
            return False, f"Unexpected error fetching questions: {str(e)}", []

    def _submit_answers(
        self, username: str, answers: List[Dict[str, Any]]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Submit agent answers to the evaluation API.

        Args:
            username: The user's Hugging Face username
            answers: List of answer payloads

        Returns:
            Tuple[bool, str, Dict]: Success status, message, and submission results
        """
        try:
            agent_code = (
                f"https://huggingface.co/spaces/{self.space_id}/tree/main"
                if self.space_id
                else "local_development"
            )

            payload = {
                "username": username.strip(),
                "agent_code": agent_code,
                "answers": answers,
            }

            response = requests.post(f"{EVAL_API_URL}/submit", json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            success_msg = (
                f"Submission successful!\n"
                f"Score: {result.get('score', 'N/A')}% "
                f"({result.get('correct_count', '?')}/{result.get('total_attempted', '?')} correct)\n"
                f"Message: {result.get('message', 'No message provided')}"
            )

            return True, success_msg, result
        except requests.exceptions.RequestException as e:
            error_msg = f"Submission failed: {str(e)}"
            return False, error_msg, {}
        except Exception as e:
            error_msg = f"Unexpected error during submission: {str(e)}"
            return False, error_msg, {}

    def process_questions(
        self, questions: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process questions using the GAIA agent.

        Args:
            questions: List of question dictionaries

        Returns:
            Tuple[List, List]: Processed answers and results log
        """
        answers = []
        results_log = []
        total = len(questions)

        for i, item in enumerate(questions, 1):
            task_id = item.get("task_id")
            question = item.get("question")

            if not task_id or not question:
                print(f"[Runner] Skipping invalid question item: {item}")
                continue

            print(f"\n[Runner] Processing question {i}/{total} (Task ID: {task_id})")
            try:
                answer = self.agent(question, task_id=task_id)

                answers.append({"task_id": task_id, "submitted_answer": answer})

                results_log.append(
                    {
                        "Task ID": task_id,
                        "Question": question,
                        "Submitted Answer": answer,
                    }
                )

                print(f"[Runner] Processed answer for {task_id}")
            except Exception as e:
                error_msg = f"Error processing task {task_id}: {str(e)}"
                print(f"[Runner] ERROR: {error_msg}")
                results_log.append(
                    {
                        "Task ID": task_id,
                        "Question": question,
                        "Submitted Answer": f"ERROR: {error_msg}",
                    }
                )

        return answers, results_log

    def run_evaluation(
        self, profile: Optional[gr.OAuthProfile] = None
    ) -> Tuple[str, pd.DataFrame]:
        """
        Run the complete evaluation process.

        Args:
            profile: Gradio OAuth profile for user authentication

        Returns:
            Tuple[str, pd.DataFrame]: Status message and results table
        """
        if not profile:
            return (
                "Please log in to Hugging Face to run the evaluation.",
                pd.DataFrame(),
            )

        username = profile.username
        print(f"[Runner] Starting evaluation for user: {username}")

        # Initialize agent
        success, message = self._initialize_agent()
        if not success:
            return message, pd.DataFrame()

        # Fetch questions
        success, message, questions = self._fetch_questions()
        if not success:
            return message, pd.DataFrame()

        # Process questions
        answers, results_log = self.process_questions(questions)
        if not answers:
            return "No valid answers were generated.", pd.DataFrame(results_log)

        # Submit answers
        success, message, _ = self._submit_answers(username, answers)

        return message, pd.DataFrame(results_log)


# --- Gradio Interface ---
def create_gradio_interface() -> gr.Blocks:
    """Create and configure the Gradio interface."""
    with gr.Blocks() as demo:
        gr.Markdown("# GAIA Agent Evaluation Runner")
        gr.Markdown("""
        ### Instructions
        1. Ensure required secrets are configured in Space settings:
           - `HF_TOKEN`: For model access
           - `TAVILY_API_KEY`: For web search functionality
        2. Log in with your Hugging Face account
        3. Click 'Run Evaluation' to start the process
        
        ### Notes
        - First run includes model download (~5-10 minutes)
        - GPU hardware recommended for optimal performance
        - Full evaluation may take significant time
        """)

        gr.LoginButton()

        # Initialize runner
        runner = GaiaEvaluationRunner()

        with gr.Row():
            run_button = gr.Button("Run Evaluation")

        with gr.Column():
            status_output = gr.Textbox(
                label="Status / Results", lines=5, interactive=False
            )
            results_table = gr.DataFrame(label="Evaluation Results", wrap=True)

        run_button.click(
            fn=runner.run_evaluation,
            outputs=[status_output, results_table],
            api_name="run_evaluation",
        )

    return demo


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Create and launch the interface
    demo = create_gradio_interface()
    demo.launch(debug=False, share=False)
