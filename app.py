
import traceback
from flask import Flask, request, jsonify, render_template
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
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

load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

if not app.secret_key:
    raise ValueError("FLASK_SECRET_KEY environment variable is not set")

def _thread_config(session_id: str) -> dict:
    """Return the LangGraph config dict that identifies a conversation thread."""
    return {"configurable": {"thread_id": session_id}}


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

    Runs the graph from START through ask_name.  The graph pauses before
    llm_node (interrupt_before), so this call returns as soon as ask_name
    has generated and persisted the greeting.  The checkpoint is written to
    SQLite and survives any subsequent restart.

    Returns:
        JSON with session_id and the LLM-generated greeting from the bot.
    """
    try:
        session_id = create_session()
        config = _thread_config(session_id)

        from graph import contact_bot_graph
        initial_state = _get_initial_state(session_id)

        # invoke runs ask_name then pauses at the interrupt_before=["llm_node"] boundary.
        contact_bot_graph.invoke(initial_state, config)

        # Read the persisted snapshot to extract the greeting.
        snapshot = contact_bot_graph.get_state(config)
        messages = snapshot.values.get("messages", [])
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        greeting = (
            ai_messages[-1].content
            if ai_messages
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

    Marks the session as ended in the database.  The checkpointer state is
    left in place (it is read-only from this point and naturally expires).

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
        # Read is_complete from the persisted checkpoint so we correctly mark
        # the session even if the worker that created it has since restarted.
        contact_collected = False
        try:
            from graph import contact_bot_graph
            snapshot = contact_bot_graph.get_state(_thread_config(session_id))
            if snapshot and snapshot.values:
                contact_collected = bool(snapshot.values.get("is_complete", False))
        except Exception:
            # If reading the snapshot fails, fall back to marking as abandoned.
            pass

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
      1. graph.update_state — injects the user's message and raw_user_input
         into the persisted checkpoint so llm_node can read them when resumed.
      2. graph.invoke(None, config) — resumes execution from the interrupt
         point (llm_node), runs validation + reply generation, then runs the
         next ask_* node (or complete_node), and pauses again.
      3. Read the latest AIMessage from the updated snapshot and return it.

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

        from graph import contact_bot_graph
        config = _thread_config(session_id)

        # Verify the checkpoint exists — it is missing only if the session was
        # never started properly (e.g. direct DB insert without /session/start).
        snapshot = contact_bot_graph.get_state(config)
        if not snapshot or not snapshot.values:
            return jsonify({
                "error": "Session state not found. Please start a new session."
            }), 404

        # Persist the user message for the logs page.
        save_message(session_id, "user", user_message)

        # Inject the user input into the checkpoint so llm_node can read it.
        # update_state merges into the current snapshot; the add_messages
        # reducer on ContactBotState.messages appends rather than replaces.
        contact_bot_graph.update_state(config, {
            "raw_user_input": user_message,
            "messages": [HumanMessage(content=user_message)],
        })

        # Resume execution from the interrupt point.  Passing None as input
        # tells LangGraph to continue from where it paused.
        contact_bot_graph.invoke(None, config)

        # Read the updated snapshot to extract the bot's reply.
        updated_snapshot = contact_bot_graph.get_state(config)
        updated_values = updated_snapshot.values if updated_snapshot else {}

        ai_messages = [
            m for m in updated_values.get("messages", [])
            if isinstance(m, AIMessage)
        ]
        latest_bot_message = (
            ai_messages[-1].content
            if ai_messages
            else "I am processing your request."
        )

        is_complete = bool(updated_values.get("is_complete", False))

        return jsonify({
            "message": latest_bot_message,
            "is_complete": is_complete,
            "current_field": updated_values.get("current_field"),
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
    app.run(debug=False, host="127.0.0.1", port=5000)