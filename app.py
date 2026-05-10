"""
Flask application for the Incede contact bot.
Provides REST API endpoints for session management, chat processing,
and log retrieval. Integrates with the LangGraph workflow and SQLite database.
"""

import traceback
from flask import Flask, request, jsonify, render_template
from langchain_core.messages import HumanMessage, AIMessage

import database
from database import (
    init_db,
    create_session,
    end_session,
    get_session,
    get_paginated_sessions,
    save_message,
    get_conversation,
    log_error,
    get_errors_for_session,
    get_contact_detail,
)

app = Flask(__name__)
app.secret_key = "incede_bot_secret_key_2024"

# In-memory storage for active graph states (keyed by session_id)
active_sessions: dict = {}


def _get_initial_state(session_id: str) -> dict:
    """
    Build the initial LangGraph state for a new session.

    Args:
        session_id: The unique session identifier.

    Returns:
        A dict representing the initial ContactBotState.
    """
    return {
        "messages": [],
        "session_id": session_id,
        "name": None,
        "phone": None,
        "email": None,
        "description": None,
        "current_field": "name",
        "is_valid": False,
        "is_complete": False,
        "error_message": None,
        "raw_user_input": None,
    }


# --------------------------------------------------------------------------
# Page routes
# --------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main landing page for Incede."""
    return render_template("index.html")


@app.route("/logs")
def logs_page():
    """Serve the logs page showing session history."""
    return render_template("logs.html")


# --------------------------------------------------------------------------
# API: Session management
# --------------------------------------------------------------------------

@app.route("/api/session/start", methods=["POST"])
def start_session():
    """
    Start a new chat session for the contact bot.
    Calls ask_name which uses the LLM to generate the greeting message.

    Returns:
        JSON with session_id and the LLM-generated greeting from the bot.
    """
    try:
        session_id = create_session()
        state = _get_initial_state(session_id)

        # ask_name generates the greeting via LLM on first call (empty messages list)
        from graph import ask_name
        initial_state = ask_name(state)
        active_sessions[session_id] = initial_state

        greeting = (
            initial_state["messages"][-1].content
            if initial_state["messages"]
            else "Hello! I am here to collect your contact details."
        )

        return jsonify({
            "session_id": session_id,
            "message": greeting,
            "status": "started",
        }), 201

    except Exception as exc:
        tb = traceback.format_exc()
        app.logger.error(f"Error starting session: {exc}\n{tb}")
        return jsonify({"error": "Failed to start session", "detail": str(exc)}), 500


@app.route("/api/session/end", methods=["POST"])
def stop_session():
    """
    End an active chat session.
    Marks the session as ended in the database and removes the in-memory state.

    Request body:
        session_id (str): The session to end.

    Returns:
        JSON confirmation of session end.
    """
    data = request.get_json()
    session_id = data.get("session_id") if data else None

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    try:
        state = active_sessions.pop(session_id, None)
        contact_collected = state.get("is_complete", False) if state else False

        existing = get_session(session_id)
        if existing and existing.get("status") == "active":
            end_session(session_id, contact_collected=contact_collected)

        return jsonify({"status": "ended", "session_id": session_id}), 200

    except Exception as exc:
        tb = traceback.format_exc()
        log_error(session_id or "unknown", type(exc).__name__, str(exc), tb)
        app.logger.error(f"Error ending session: {exc}\n{tb}")
        return jsonify({"error": "Failed to end session", "detail": str(exc)}), 500


# --------------------------------------------------------------------------
# API: Chat processing
# --------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Process a user message through the LangGraph workflow.

    Flow per request:
      1. llm_node  — extracts + validates the user input and generates a reply
                     for every invalid/off-topic/extraction-failure outcome.
      2. route_after_llm — decides whether to advance or stay on current field.
      3. Next ask_* node  — called ONLY when the field was valid, so it can
                            generate the "acknowledged + next question" message.
         complete_node    — called when all fields are done.

    This means the ask_* nodes are NEVER called on an invalid submission,
    preventing duplicate bot messages.

    Request body:
        session_id (str): The active session ID.
        message (str): The user's message text.

    Returns:
        JSON with the bot's reply message and whether the session is complete.
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body is required"}), 400

    session_id = data.get("session_id")
    user_message = data.get("message", "").strip()

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    try:
        session_record = get_session(session_id)
        if not session_record:
            return jsonify({"error": "Session not found"}), 404

        if session_record.get("status") != "active":
            return jsonify({"error": "Session is no longer active"}), 400

        state = active_sessions.get(session_id)
        if not state:
            return jsonify({"error": "Session state not found. Please start a new session."}), 404

        # Persist the user message and add it to the state
        save_message(session_id, "user", user_message)
        state["messages"] = state["messages"] + [HumanMessage(content=user_message)]
        state["raw_user_input"] = user_message

        from graph import (
            llm_node, route_after_llm,
            ask_phone, ask_email, ask_description, complete_node,
        )

        # Step 1: extract, validate, and generate a reply for invalid outcomes
        state = llm_node(state)

        # Step 2: decide the next node
        next_node_name = route_after_llm(state)

        # Step 3: only run the next node when the current field was VALID.
        # On invalid, llm_node already added the error reply — calling the
        # ask_* node again would append a second, redundant bot message.
        if state.get("is_valid"):
            next_node_fn = {
                # ask_name is intentionally excluded: if name re-validation
                # somehow routes back, llm_node's message is sufficient.
                "ask_phone": ask_phone,
                "ask_email": ask_email,
                "ask_description": ask_description,
                "complete": complete_node,
            }.get(next_node_name)

            if next_node_fn:
                state = next_node_fn(state)

        # Persist updated state
        active_sessions[session_id] = state

        # Return the latest bot message
        bot_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
        latest_bot_message = (
            bot_messages[-1].content if bot_messages else "I am processing your request."
        )

        is_complete = state.get("is_complete", False)
        if is_complete:
            active_sessions.pop(session_id, None)

        return jsonify({
            "message": latest_bot_message,
            "is_complete": is_complete,
            "current_field": state.get("current_field"),
        }), 200

    except Exception as exc:
        tb = traceback.format_exc()
        log_error(session_id, type(exc).__name__, str(exc), tb)
        app.logger.error(f"Chat processing error for session {session_id}: {exc}\n{tb}")

        return jsonify({
            "message": "I encountered an issue processing your request. Please try again.",
            "is_complete": False,
            "error": str(exc),
        }), 500


# --------------------------------------------------------------------------
# API: Logs and session details
# --------------------------------------------------------------------------

@app.route("/api/logs", methods=["GET"])
def get_logs():
    """
    Retrieve paginated session logs.

    Query parameters:
        page (int): Page number, default 1.
        per_page (int): Records per page, default 10.

    Returns:
        JSON with paginated session list and pagination metadata.
    """
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 10))

        if page < 1:
            page = 1
        if per_page < 1 or per_page > 100:
            per_page = 10

        result = get_paginated_sessions(page=page, per_page=per_page)
        return jsonify(result), 200

    except Exception as exc:
        app.logger.error(f"Error retrieving logs: {exc}")
        return jsonify({"error": "Failed to retrieve logs", "detail": str(exc)}), 500


@app.route("/api/logs/<session_id>/conversation", methods=["GET"])
def get_session_conversation(session_id: str):
    """
    Retrieve the full conversation history for a specific session.

    Args:
        session_id: The session ID from the URL path.

    Returns:
        JSON with the list of messages and contact details if available.
    """
    try:
        conversation = get_conversation(session_id)
        contact = get_contact_detail(session_id)

        return jsonify({
            "session_id": session_id,
            "conversation": conversation,
            "contact_detail": contact,
        }), 200

    except Exception as exc:
        app.logger.error(f"Error retrieving conversation for {session_id}: {exc}")
        return jsonify({"error": "Failed to retrieve conversation", "detail": str(exc)}), 500


@app.route("/api/logs/<session_id>/errors", methods=["GET"])
def get_session_errors(session_id: str):
    """
    Retrieve all errors logged for a specific session.

    Args:
        session_id: The session ID from the URL path.

    Returns:
        JSON with a list of error records for the session.
    """
    try:
        errors = get_errors_for_session(session_id)
        return jsonify({
            "session_id": session_id,
            "errors": errors,
        }), 200

    except Exception as exc:
        app.logger.error(f"Error retrieving errors for {session_id}: {exc}")
        return jsonify({"error": "Failed to retrieve errors", "detail": str(exc)}), 500


# --------------------------------------------------------------------------
# Application entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)