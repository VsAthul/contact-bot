/**
 * chat.js
 * Handles the contact bot chat UI on the landing page.
 * Manages session lifecycle, message sending/receiving, and UI state.
 */

(function () {
    "use strict";

    // --- DOM References ---
    const chatTrigger    = document.getElementById("chatTrigger");
    const chatWindow     = document.getElementById("chatWindow");
    const chatMessages   = document.getElementById("chatMessages");
    const chatInputArea  = document.getElementById("chatInputArea");
    const chatInput      = document.getElementById("chatInput");
    const chatSendBtn    = document.getElementById("chatSendBtn");
    const chatStatus     = document.getElementById("chatStatus");
    const chatBadge      = document.getElementById("chatBadge");
    const openIcon       = chatTrigger.querySelector(".open-icon");
    const closeIcon      = chatTrigger.querySelector(".close-icon");

    // --- State ---
    let isOpen       = false;
    let sessionId    = null;
    let isProcessing = false;
    let isComplete   = false;

    // --- Toggle the chat window ---
    chatTrigger.addEventListener("click", function () {
        if (isOpen) {
            closeChat();
        } else {
            openChat();
        }
    });

    /**
     * Open the chat window and start a new session if one is not already active.
     */
    function openChat() {
        isOpen = true;
        chatWindow.classList.add("open");
        openIcon.style.display = "none";
        closeIcon.style.display = "block";
        chatBadge.classList.remove("active");

        if (!sessionId && !isComplete) {
            startSession();
        }

        // Focus input after animation
        setTimeout(function () {
            if (chatInput) chatInput.focus();
        }, 300);
    }

    /**
     * Close the chat window and end the session if it is still active.
     */
    function closeChat() {
        isOpen = false;
        chatWindow.classList.remove("open");
        openIcon.style.display = "block";
        closeIcon.style.display = "none";

        if (sessionId && !isComplete) {
            endSession();
        }
    }

    /**
     * Start a new bot session by calling the backend API.
     * Renders the initial greeting from the bot.
     */
    function startSession() {
        setStatus("Connecting...", false);
        clearMessages();

        fetch("/api/session/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
        })
            .then(function (res) {
                if (!res.ok) throw new Error("Failed to start session: " + res.status);
                return res.json();
            })
            .then(function (data) {
                sessionId = data.session_id;
                setStatus("Online", false);
                chatInputArea.style.display = "flex";
                renderBotMessage(data.message);
                chatInput.disabled = false;
                chatSendBtn.disabled = false;
            })
            .catch(function (err) {
                console.error("Session start error:", err);
                setStatus("Offline", false);
                renderSystemMessage("Unable to connect to the bot. Please refresh and try again.");
            });
    }

    /**
     * End the current session by notifying the backend.
     * Called when the user closes the chat window.
     */
    function endSession() {
        if (!sessionId) return;

        navigator.sendBeacon(
            "/api/session/end",
            new Blob(
                [JSON.stringify({ session_id: sessionId })],
                { type: "application/json" }
            )
        );

        sessionId = null;
    }

    /**
     * Send the user's typed message to the backend and display the bot response.
     */
    function sendMessage() {
        const text = chatInput.value.trim();
        if (!text || isProcessing || !sessionId) return;

        renderUserMessage(text);
        chatInput.value = "";
        setProcessing(true);

        const typingEl = showTypingIndicator();

        fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId, message: text }),
        })
            .then(function (res) {
                return res.json();
            })
            .then(function (data) {
                removeTypingIndicator(typingEl);

                if (data.message) {
                    renderBotMessage(data.message);
                }

                if (data.is_complete) {
                    isComplete = true;
                    sessionId = null;
                    chatInput.disabled = true;
                    chatSendBtn.disabled = true;
                    setStatus("Session ended", false);
                    chatBadge.classList.add("active");
                }
            })
            .catch(function (err) {
                console.error("Chat error:", err);
                removeTypingIndicator(typingEl);
                renderSystemMessage("Something went wrong. Please try again.");
            })
            .finally(function () {
                setProcessing(false);
            });
    }

    // --- Event listeners for sending messages ---
    chatSendBtn.addEventListener("click", sendMessage);

    chatInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // --- UI Helper functions ---

    /**
     * Render a bot message bubble in the chat window.
     * @param {string} text - The message text to display.
     */
    function renderBotMessage(text) {
        const el = document.createElement("div");
        el.className = "msg msg-bot";
        el.textContent = text;
        appendMessage(el);
    }

    /**
     * Render a user message bubble in the chat window.
     * @param {string} text - The message text to display.
     */
    function renderUserMessage(text) {
        const el = document.createElement("div");
        el.className = "msg msg-user";
        el.textContent = text;
        appendMessage(el);
    }

    /**
     * Render a system/info message in the chat window.
     * @param {string} text - The system message text to display.
     */
    function renderSystemMessage(text) {
        const el = document.createElement("div");
        el.className = "msg msg-system";
        el.textContent = text;
        appendMessage(el);
    }

    /**
     * Append a message element to the chat container and scroll to the bottom.
     * @param {HTMLElement} el - The message element to append.
     */
    function appendMessage(el) {
        // Remove the start prompt if still visible
        const prompt = chatMessages.querySelector(".chat-start-prompt");
        if (prompt) prompt.remove();

        chatMessages.appendChild(el);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    /**
     * Show the animated typing indicator while waiting for a bot response.
     * @returns {HTMLElement} The typing indicator element (for later removal).
     */
    function showTypingIndicator() {
        const el = document.createElement("div");
        el.className = "typing-indicator";
        el.innerHTML = `
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        `;
        chatMessages.appendChild(el);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        setStatus("Typing...", true);
        return el;
    }

    /**
     * Remove the typing indicator element from the chat.
     * @param {HTMLElement} el - The typing indicator element to remove.
     */
    function removeTypingIndicator(el) {
        if (el && el.parentNode) {
            el.parentNode.removeChild(el);
        }
        setStatus("Online", false);
    }

    /**
     * Update the status text in the chat header.
     * @param {string} text - The status text to display.
     * @param {boolean} isTyping - Whether to apply the typing style.
     */
    function setStatus(text, isTyping) {
        chatStatus.textContent = text;
        chatStatus.className = isTyping ? "chat-status typing" : "chat-status";
    }

    /**
     * Enable or disable input controls during processing.
     * @param {boolean} processing - Whether the bot is currently processing.
     */
    function setProcessing(processing) {
        isProcessing = processing;
        chatInput.disabled = processing;
        chatSendBtn.disabled = processing;
    }

    /**
     * Clear all messages from the chat container and show the start prompt.
     */
    function clearMessages() {
        chatMessages.innerHTML = '<div class="chat-start-prompt"><p>Starting conversation...</p></div>';
        chatInputArea.style.display = "none";
        chatInput.disabled = true;
        chatSendBtn.disabled = true;
    }

    // Handle page unload: end session if still active
    window.addEventListener("beforeunload", function () {
        if (sessionId && !isComplete) {
            navigator.sendBeacon(
                "/api/session/end",
                new Blob(
                    [JSON.stringify({ session_id: sessionId })],
                    { type: "application/json" }
                )
            );
        }
    });
})();
