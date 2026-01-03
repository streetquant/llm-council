"""3-stage LLM Council orchestration."""

from typing import List, Dict, Any, Tuple, Optional
from .openrouter import query_models_parallel, query_model
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL


async def stage1_collect_responses(
    user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

    Args:
        user_query: The user's question
        conversation_history: List of previous messages with 'role' and 'content'

    Returns:
        List of dicts with 'model' and 'response' keys
    """
    import asyncio

    conversation_history = conversation_history or []

    # Special system prompts for specific models
    SPECIAL_ROLES = {
        "qwen/qwen3-coder:free": {
            "role": "system",
            "content": "You are a specialized Calculation and Analysis Agent. Your primary role is to provide precise mathematical calculations, data analysis, logical reasoning, and code-based solutions. Focus on accuracy, step-by-step breakdowns, and quantitative insights. When answering questions, prioritize numerical analysis, formulas, algorithms, and structured analytical approaches.",
        }
    }

    # Build tasks with model-specific prompts
    tasks = []
    for model in COUNCIL_MODELS:
        if model in SPECIAL_ROLES:
            messages = [SPECIAL_ROLES[model]]
        else:
            messages = []

        # Add conversation history
        messages.extend(conversation_history)

        # Add current user query
        messages.append({"role": "user", "content": user_query})

        tasks.append((model, query_model(model, messages)))

    # Query all models in parallel
    results = await asyncio.gather(*[task[1] for task in tasks])
    responses = {task[0]: result for task, result in zip(tasks, results)}

    # Format results
    stage1_results = []
    for model, response in responses.items():
        if response is not None:  # Only include successful responses
            stage1_results.append(
                {"model": model, "response": response.get("content", "")}
            )

    return stage1_results


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1
        conversation_history: List of previous messages with 'role' and 'content'

    Returns:
        Tuple of (rankings list, label_to_model mapping)
    """
    conversation_history = conversation_history or []

    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result["model"]
        for label, result in zip(labels, stage1_results)
    }

    # Build context string from conversation history
    context_text = ""
    if conversation_history:
        context_text = "\n\nPrevious conversation:\n" + "\n".join(
            [f"{msg['role'].title()}: {msg['content']}" for msg in conversation_history]
        )

    # Build the ranking prompt
    responses_text = "\n\n".join(
        [
            f"Response {label}:\n{result['response']}"
            for label, result in zip(labels, stage1_results)
        ]
    )

    ranking_prompt = f"""You are evaluating different responses to the following question:
{context_text}

Current Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    # Format results
    stage2_results = []
    for model, response in responses.items():
        if response is not None:
            full_text = response.get("content", "")
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append(
                {"model": model, "ranking": full_text, "parsed_ranking": parsed}
            )

    return stage2_results, label_to_model


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2
        conversation_history: List of previous messages with 'role' and 'content'

    Returns:
        Dict with 'model' and 'response' keys
    """
    conversation_history = conversation_history or []

    # Build comprehensive context for chairman
    stage1_text = "\n\n".join(
        [
            f"Model: {result['model']}\nResponse: {result['response']}"
            for result in stage1_results
        ]
    )

    stage2_text = "\n\n".join(
        [
            f"Model: {result['model']}\nRanking: {result['ranking']}"
            for result in stage2_results
        ]
    )

    # Build context string from conversation history
    context_text = ""
    if conversation_history:
        context_text = "\n\nPrevious conversation:\n" + "\n".join(
            [f"{msg['role'].title()}: {msg['content']}" for msg in conversation_history]
        )

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.
{context_text}

Current Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to provide the FINAL DECISION for the user's question, not merely a summary. Follow this approach:

1. CRITICALLY EVALUATE each individual response:
   - Identify strengths: accuracy, completeness, clarity, evidence provided
   - Identify weaknesses: omissions, errors, poor reasoning, hallucinations
   - Be specific and objective in your criticism

2. JUDGE THE QUALITY of responses:
   - Which responses are best? Why? Cite specific reasons.
   - Which responses are worst? Why? Be direct.
   - Compare responses directly against each other

3. PROVIDE YOUR OWN INDEPENDENT ASSESSMENT:
   - Do not simply parrot the peer rankings
   - Give your own judgment based on your critical evaluation
   - If you disagree with the peer rankings, explain why
   - Highlight insights that peers may have missed

4. DELIVER A FINAL DECISION:
   - Synthesize the best insights from all responses
   - Correct any errors in lower-ranked responses
   - Provide your own authoritative answer to the user's question
   - Be decisive and clear

Your goal is to be a CRITICAL THINKER and ACTIVE JUDGE, not a passive summarizer. The user relies on you to provide the most accurate, well-reasoned answer possible.

Provide your final answer now:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model
    response = await query_model(CHAIRMAN_MODEL, messages)

    if response is None:
        # Fallback if chairman fails
        return {
            "model": CHAIRMAN_MODEL,
            "response": "Error: Unable to generate final synthesis.",
        }

    return {"model": CHAIRMAN_MODEL, "response": response.get("content", "")}


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order
    """
    import re

    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        # Extract everything after "FINAL RANKING:"
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format (e.g., "1. Response A")
            # This pattern looks for: number, period, optional space, "Response X"
            numbered_matches = re.findall(r"\d+\.\s*Response [A-Z]", ranking_section)
            if numbered_matches:
                # Extract just the "Response X" part
                result = []
                for m in numbered_matches:
                    match = re.search(r"Response [A-Z]", m)
                    if match:
                        result.append(match.group())
                return result

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r"Response [A-Z]", ranking_section)
            return matches

    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r"Response [A-Z]", ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]], label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking["ranking"]

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append(
                {
                    "model": model,
                    "average_rank": round(avg_rank, 2),
                    "rankings_count": len(positions),
                }
            )

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x["average_rank"])

    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    # Use gemini-2.5-flash for title generation (fast and cheap)
    response = await query_model("google/gemini-2.5-flash", messages, timeout=30.0)

    if response is None:
        # Fallback to a generic title
        return "New Conversation"

    title = response.get("content", "New Conversation").strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip("\"'")

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(
    user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None
) -> Tuple[List, List, Dict, Dict]:
    """
    Run the complete 3-stage council process.

    Args:
        user_query: The user's question
        conversation_history: List of previous messages with 'role' and 'content'

    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata)
    """
    conversation_history = conversation_history or []

    # Stage 1: Collect individual responses
    stage1_results = await stage1_collect_responses(user_query, conversation_history)

    # If no models responded successfully, return error
    if not stage1_results:
        return (
            [],
            [],
            {
                "model": "error",
                "response": "All models failed to respond. Please try again.",
            },
            {},
        )

    # Stage 2: Collect rankings
    stage2_results, label_to_model = await stage2_collect_rankings(
        user_query, stage1_results, conversation_history
    )

    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Stage 3: Synthesize final answer
    stage3_result = await stage3_synthesize_final(
        user_query, stage1_results, stage2_results, conversation_history
    )

    # Prepare metadata
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
    }

    return stage1_results, stage2_results, stage3_result, metadata
