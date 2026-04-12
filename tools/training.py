"""
Training data collection tools (Section K).

MCP tools for capturing user questions, generating hypotheses,
evaluating them against actual results, and storing training data.
"""

from __future__ import annotations

import os
import json
from typing import Optional

from .config import resolve_path, WORKSPACE_ROOT

# Import training module components
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.session import TrainingSession, get_session_manager
from training.evaluator import (
    HypothesisEvaluator,
    StructuredHypothesis,
    ClaimType,
    parse_hypothesis_response,
)
from training.storage import TrainingDataStorage


# Environment variables for LLM proxy
TRAINING_LLM_PROXY_URL = os.environ.get(
    "TRAINING_LLM_PROXY_URL",
    os.environ.get("LLM_PROXY_URL", "")
)
TRAINING_LLM_TOKEN = os.environ.get(
    "TRAINING_LLM_TOKEN",
    os.environ.get("IDE_LLM_TOKEN", "")
)
TRAINING_MODEL = os.environ.get("TRAINING_MODEL", "anthropic/claude-3-haiku")


HYPOTHESIS_SYSTEM_PROMPT = """You are a geological data analyst generating testable hypotheses.

Given a user's question and the code/operations that will be executed, predict what the results will show.

You must output EXACTLY 10 hypotheses as a JSON array. Each hypothesis must be a JSON object with these fields:
- "text": A clear, specific prediction (string)
- "claim_type": One of: "correlation", "distribution", "cluster", "anomaly", "comparison", "existence", "count", "trend", "general"
- "column_a": Primary column name if applicable (string or null)
- "column_b": Secondary column name if applicable (string or null)  
- "direction": For correlations/trends: "positive", "negative", or "none" (string or null)
- "magnitude_range": Expected numeric range as [low, high] (array of 2 numbers or null)
- "value_range": Expected value range as [low, high] (array of 2 numbers or null)
- "count_range": Expected count range as [low, high] (array of 2 integers or null)
- "reasoning": Brief geological reasoning for this prediction (string)

Be SPECIFIC. Vague predictions score lower. Include concrete numbers, ranges, and geological reasoning.

Example output:
[
  {
    "text": "The Au/Cu correlation is moderate positive (0.4-0.6), consistent with porphyry-style co-precipitation",
    "claim_type": "correlation",
    "column_a": "Au_ppm",
    "column_b": "Cu_ppm",
    "direction": "positive",
    "magnitude_range": [0.4, 0.6],
    "value_range": null,
    "count_range": null,
    "reasoning": "Gold and copper commonly co-precipitate in porphyry systems due to similar fluid chemistry"
  }
]"""


def start_training_session(
    question: str,
    specialist_type: str = None,
) -> dict:
    """
    Start a training data collection session.
    
    Call this at the beginning of an analysis workflow to capture
    the user's question and subsequent tool calls for training data.
    
    Args:
        question: The user's original question being answered
        specialist_type: Optional specialist type (dataops, analytics, etc.)
    
    Returns:
        Session info including session_id for subsequent calls
    """
    manager = get_session_manager()
    
    session = manager.start_session(
        question=question,
        specialist_type=specialist_type,
    )
    
    return {
        "status": "started",
        "session_id": session.session_id,
        "message": f"Training session started. Tool calls will be recorded.",
        "question": question,
    }


def generate_hypotheses(
    session_id: str,
    num_hypotheses: int = 10,
) -> dict:
    """
    Generate hypotheses about the analysis output.
    
    Uses an LLM to predict what the results will show, based on
    the question and code executed (without seeing actual results).
    
    Args:
        session_id: The training session ID from start_training_session
        num_hypotheses: Number of hypotheses to generate (default 10)
    
    Returns:
        List of structured hypotheses
    """
    manager = get_session_manager()
    session = manager.get_session(session_id)
    
    if not session:
        return {"error": f"Session {session_id} not found"}
    
    if not session.tool_calls:
        return {"error": "No tool calls recorded yet. Run analysis first."}
    
    # Build the prompt (code without results)
    code_context = session.get_code_context(include_results=False)
    
    user_prompt = f"""**Question:** {session.question}

**Operations to be executed:**
```python
{code_context}
```

Generate exactly {num_hypotheses} specific, testable hypotheses about what the results will show.
Output as a JSON array."""

    # Call LLM
    hypotheses_response = _call_llm(
        system_prompt=HYPOTHESIS_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    
    if "error" in hypotheses_response:
        return hypotheses_response
    
    # Parse response into structured hypotheses
    response_text = hypotheses_response.get("content", "")
    hypotheses = parse_hypothesis_response(response_text)
    
    if not hypotheses:
        return {
            "error": "Failed to parse hypotheses from LLM response",
            "raw_response": response_text[:1000],
        }
    
    # Store hypotheses in session
    session.hypotheses = [h.to_dict() for h in hypotheses]
    
    return {
        "status": "success",
        "session_id": session_id,
        "hypotheses_count": len(hypotheses),
        "hypotheses": [h.to_dict() for h in hypotheses],
    }


def evaluate_hypotheses(
    session_id: str,
    source_data_path: str = None,
) -> dict:
    """
    Evaluate hypotheses against actual results.
    
    Compares each hypothesis to the actual output from tool execution,
    determining correctness and computing specificity scores.
    
    Args:
        session_id: The training session ID
        source_data_path: Optional path to source data for additional checks
    
    Returns:
        Evaluation results with correct/incorrect hypotheses
    """
    manager = get_session_manager()
    session = manager.get_session(session_id)
    
    if not session:
        return {"error": f"Session {session_id} not found"}
    
    if not session.hypotheses:
        return {"error": "No hypotheses generated yet. Call generate_hypotheses first."}
    
    # Get actual results from tool calls
    actual_results = session.get_final_result()
    if actual_results is None:
        return {"error": "No results to evaluate against"}
    
    # Convert stored dicts back to StructuredHypothesis objects
    hypotheses = [StructuredHypothesis.from_dict(h) for h in session.hypotheses]
    
    # Load source data if provided
    source_data = None
    if source_data_path:
        try:
            from .dataframe_cache import load
            source_data = load(source_data_path)
        except Exception as e:
            pass  # Continue without source data
    
    # Evaluate
    evaluator = HypothesisEvaluator(tolerance=0.1)
    evaluated = evaluator.evaluate_batch(hypotheses, actual_results, source_data)
    
    # Categorize results
    correct = [h for h in evaluated if h.is_correct is True]
    incorrect = [h for h in evaluated if h.is_correct is False]
    unevaluated = [h for h in evaluated if h.is_correct is None]
    
    # Store evaluation results
    session.evaluation_results = {
        "correct_count": len(correct),
        "incorrect_count": len(incorrect),
        "unevaluated_count": len(unevaluated),
        "evaluated_hypotheses": [h.to_dict() for h in evaluated],
    }
    
    return {
        "status": "success",
        "session_id": session_id,
        "correct": [h.to_dict() for h in correct],
        "incorrect": [h.to_dict() for h in incorrect],
        "unevaluated": [h.to_dict() for h in unevaluated],
        "summary": {
            "correct_count": len(correct),
            "incorrect_count": len(incorrect),
            "unevaluated_count": len(unevaluated),
            "avg_specificity_correct": (
                sum(h.specificity_score for h in correct) / len(correct)
                if correct else 0
            ),
            "avg_specificity_incorrect": (
                sum(h.specificity_score for h in incorrect) / len(incorrect)
                if incorrect else 0
            ),
        },
    }


def end_training_session(
    session_id: str,
    save: bool = True,
) -> dict:
    """
    End a training session and optionally save the data.
    
    Finalizes the session, saves correct hypotheses for SFT training,
    and saves rejected hypotheses for reference.
    
    Args:
        session_id: The training session ID
        save: Whether to save the training data (default True)
    
    Returns:
        Summary of saved data
    """
    manager = get_session_manager()
    session = manager.get_session(session_id)
    
    if not session:
        return {"error": f"Session {session_id} not found"}
    
    if not session.evaluation_results:
        return {"error": "No evaluation results. Call evaluate_hypotheses first."}
    
    result = {
        "status": "ended",
        "session_id": session_id,
    }
    
    if save:
        # Get evaluated hypotheses
        evaluated_dicts = session.evaluation_results.get("evaluated_hypotheses", [])
        hypotheses = [StructuredHypothesis.from_dict(h) for h in evaluated_dicts]
        
        # Save to storage
        storage = TrainingDataStorage()
        save_result = storage.save_session(session, hypotheses)
        result.update(save_result)
        result["status"] = "saved"
    
    # Mark session as finalized and remove from active tracking
    session.finalized = True
    manager.end_session(session_id)
    
    return result


def get_training_stats() -> dict:
    """
    Get statistics about collected training data.
    
    Returns:
        Summary of training data collected so far
    """
    storage = TrainingDataStorage()
    return storage.get_stats()


def _call_llm(system_prompt: str, user_prompt: str) -> dict:
    """
    Call the training LLM proxy.
    
    Uses TRAINING_LLM_PROXY_URL and TRAINING_LLM_TOKEN.
    """
    import httpx
    
    if not TRAINING_LLM_PROXY_URL:
        return {"error": "TRAINING_LLM_PROXY_URL not configured"}
    
    if not TRAINING_LLM_TOKEN:
        return {"error": "TRAINING_LLM_TOKEN not configured"}
    
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{TRAINING_LLM_PROXY_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {TRAINING_LLM_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": TRAINING_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4000,
                },
            )
            
            if response.status_code != 200:
                return {
                    "error": f"LLM request failed: {response.status_code}",
                    "detail": response.text[:500],
                }
            
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            return {"content": content}
    
    except httpx.TimeoutException:
        return {"error": "LLM request timed out"}
    except Exception as e:
        return {"error": f"LLM request failed: {str(e)}"}


def record_tool_call(tool: str, args: dict, result) -> bool:
    """
    Record a tool call to the active training session.
    
    This is called automatically by the MCP server middleware
    when a training session is active.
    
    Args:
        tool: Name of the tool called
        args: Arguments passed to the tool
        result: Result returned by the tool
    
    Returns:
        True if recorded, False if no active session
    """
    manager = get_session_manager()
    return manager.record_tool_call(tool, args, result)
