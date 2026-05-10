/**
 * logs.js
 * Handles the session logs page functionality.
 * Loads and paginates session logs, and shows session detail
 * (conversation, contact details, or errors) in the right-hand drawer.
 */

(function () {
    "use strict";

    // --- DOM References ---
    const logsLoading      = document.getElementById("logsLoading");
    const tableWrapper     = document.getElementById("tableWrapper");
    const logsTableBody    = document.getElementById("logsTableBody");
    const emptyState       = document.getElementById("emptyState");
    const prevPageBtn      = document.getElementById("prevPage");
    const nextPageBtn      = document.getElementById("nextPage");
    const pageInfo         = document.getElementById("pageInfo");

    const drawerPlaceholder = document.getElementById("drawerPlaceholder");
    const drawerContent     = document.getElementById("drawerContent");
    const drawerClose       = document.getElementById("drawerClose");
    const drawerSessionId   = document.getElementById("drawerSessionId");
    const tabConversation   = document.getElementById("tabConversation");
    const tabContact        = document.getElementById("tabContact");
    const tabErrors         = document.getElementById("tabErrors");
    const panelConversation = document.getElementById("panelConversation");
    const panelContact      = document.getElementById("panelContact");
    const panelErrors       = document.getElementById("panelErrors");
    const miniChat          = document.getElementById("miniChat");
    const contactDetail     = document.getElementById("contactDetail");
    const errorList         = document.getElementById("errorList");

    // --- State ---
    let currentPage     = 1;
    const perPage       = 10;
    let selectedRow     = null;
    let activeSessionId = null;
    let activeTab       = "conversation";

    // --- Add # column header to table ---
    const thead = document.querySelector("#logsTable thead tr");
    if (thead) {
        const th = document.createElement("th");
        th.textContent = "#";
        th.style.width = "3rem";
        thead.insertBefore(th, thead.firstChild);
    }

    // --- Initial load ---
    fetchLogs(currentPage);

    /**
     * Fetch the paginated session logs from the backend API.
     * @param {number} page - The page number to fetch.
     */
    function fetchLogs(page) {
        showLoading(true);

        fetch("/api/logs?page=" + page + "&per_page=" + perPage)
            .then(function (res) {
                if (!res.ok) throw new Error("Failed to fetch logs");
                return res.json();
            })
            .then(function (data) {
                showLoading(false);
                // Backend returns "total", not "total_items"
                renderTable(data.items, data.page);
                renderPagination(data.page, data.total_pages);
                currentPage = data.page;
            })
            .catch(function (err) {
                console.error("Error fetching logs:", err);
                showLoading(false);
                showEmpty("Failed to load logs. Please refresh.");
            });
    }

    /**
     * Render the table rows from the fetched session data.
     * @param {Array} sessions - List of session objects from the API.
     * @param {number} page - Current page number (for serial number calculation).
     */
    function renderTable(sessions, page) {
        logsTableBody.innerHTML = "";

        if (!sessions || sessions.length === 0) {
            showEmpty();
            return;
        }

        tableWrapper.style.display = "block";
        emptyState.style.display = "none";

        const startSerial = (page - 1) * perPage + 1;

        sessions.forEach(function (session, index) {
            const tr = document.createElement("tr");
            tr.dataset.sessionId = session.id;

            const serial = startSerial + index;
            const shortId = session.id.substring(0, 8) + "...";
            const startedAt = formatDateTime(session.started_at);
            const endedAt = session.ended_at ? formatDateTime(session.ended_at) : "—";
            const statusBadge = buildStatusBadge(session.status);
            const contactDot = session.contact_collected
                ? '<span class="contact-dot yes" title="Collected"></span>'
                : '<span class="contact-dot no" title="Not collected"></span>';

            tr.innerHTML = `
                <td style="color:var(--gold);font-size:0.8rem;font-weight:600;">${serial}</td>
                <td title="${session.id}">${shortId}</td>
                <td>${startedAt}</td>
                <td>${endedAt}</td>
                <td>${statusBadge}</td>
                <td style="text-align:center">${contactDot}</td>
            `;

            tr.addEventListener("click", function () {
                selectRow(tr, session.id);
            });

            logsTableBody.appendChild(tr);
        });
    }

    /**
     * Handle row selection: highlight the row and open the detail drawer.
     */
    function selectRow(row, sessionId) {
        if (selectedRow) selectedRow.classList.remove("selected");
        row.classList.add("selected");
        selectedRow = row;
        activeSessionId = sessionId;
        openDrawer(sessionId);
    }

    /**
     * Open the detail drawer and load the conversation for the given session.
     */
    function openDrawer(sessionId) {
        drawerPlaceholder.style.display = "none";
        drawerContent.style.display = "flex";
        drawerSessionId.textContent = sessionId;
        // Clear cached contact detail when switching sessions
        delete drawerContent.dataset.contactDetail;
        switchTab("conversation");
        loadConversation(sessionId);
    }

    /**
     * Close the detail drawer and reset its state.
     */
    function closeDrawer() {
        drawerContent.style.display = "none";
        drawerPlaceholder.style.display = "flex";
        if (selectedRow) {
            selectedRow.classList.remove("selected");
            selectedRow = null;
        }
        activeSessionId = null;
    }

    drawerClose.addEventListener("click", closeDrawer);

    /**
     * Switch between tabs: 'conversation', 'contact', or 'errors'.
     */
    function switchTab(tabName) {
        activeTab = tabName;
        tabConversation.classList.toggle("active", tabName === "conversation");
        tabContact.classList.toggle("active",      tabName === "contact");
        tabErrors.classList.toggle("active",       tabName === "errors");
        panelConversation.style.display = tabName === "conversation" ? "block" : "none";
        panelContact.style.display      = tabName === "contact"      ? "block" : "none";
        panelErrors.style.display       = tabName === "errors"       ? "block" : "none";
    }

    tabConversation.addEventListener("click", function () {
        if (activeSessionId) { switchTab("conversation"); loadConversation(activeSessionId); }
    });

    tabContact.addEventListener("click", function () {
        if (activeSessionId) { switchTab("contact"); loadContact(activeSessionId); }
    });

    tabErrors.addEventListener("click", function () {
        if (activeSessionId) { switchTab("errors"); loadErrors(activeSessionId); }
    });

    /**
     * Fetch and render the conversation for a session.
     * Also caches contact_detail from the same API response.
     */
    function loadConversation(sessionId) {
        miniChat.innerHTML = '<p style="color:var(--text-dim);font-size:0.85rem;text-align:center;padding:1rem;">Loading...</p>';

        fetch("/api/logs/" + sessionId + "/conversation")
            .then(function (res) {
                if (!res.ok) throw new Error("Failed to fetch conversation");
                return res.json();
            })
            .then(function (data) {
                renderMiniChat(data.conversation);
                // Cache contact_detail so the Contact tab doesn't need a second fetch
                drawerContent.dataset.contactDetail = JSON.stringify(data.contact_detail || null);
            })
            .catch(function (err) {
                console.error("Error loading conversation:", err);
                miniChat.innerHTML = '<p style="color:var(--red);font-size:0.85rem;text-align:center;padding:1rem;">Failed to load conversation.</p>';
            });
    }

    /**
     * Render the Contact tab using data cached from loadConversation.
     * Falls back to a fresh fetch if cache is missing.
     */
    function loadContact(sessionId) {
        // Use cached data if available
        if ("contactDetail" in drawerContent.dataset) {
            try {
                renderContact(JSON.parse(drawerContent.dataset.contactDetail));
                return;
            } catch (e) { /* fall through */ }
        }

        contactDetail.innerHTML = '<p style="color:var(--text-dim);font-size:0.85rem;text-align:center;padding:1rem;">Loading...</p>';
        fetch("/api/logs/" + sessionId + "/conversation")
            .then(function (res) { return res.json(); })
            .then(function (data) {
                drawerContent.dataset.contactDetail = JSON.stringify(data.contact_detail || null);
                renderContact(data.contact_detail);
            })
            .catch(function () {
                contactDetail.innerHTML = '<p style="color:var(--red);font-size:0.85rem;text-align:center;padding:1rem;">Failed to load contact details.</p>';
            });
    }

    /**
     * Fetch and render the errors for a session.
     */
    function loadErrors(sessionId) {
        errorList.innerHTML = '<p style="color:var(--text-dim);font-size:0.85rem;text-align:center;padding:1rem;">Loading...</p>';

        fetch("/api/logs/" + sessionId + "/errors")
            .then(function (res) {
                if (!res.ok) throw new Error("Failed to fetch errors");
                return res.json();
            })
            .then(function (data) { renderErrors(data.errors); })
            .catch(function (err) {
                console.error("Error loading errors:", err);
                errorList.innerHTML = '<p style="color:var(--red);font-size:0.85rem;text-align:center;padding:1rem;">Failed to load errors.</p>';
            });
    }

    /**
     * Render conversation messages inside the mini-chat panel.
     */
    function renderMiniChat(messages) {
        miniChat.innerHTML = "";

        if (!messages || messages.length === 0) {
            miniChat.innerHTML = '<p style="color:var(--text-dim);font-size:0.85rem;text-align:center;padding:1rem;">No messages found.</p>';
            return;
        }

        messages.forEach(function (msg) {
            const timeEl = document.createElement("div");
            timeEl.className = "mini-msg-time";
            timeEl.textContent = formatDateTime(msg.timestamp);

            const msgEl = document.createElement("div");
            msgEl.className = "mini-msg " + (msg.role === "user" ? "user" : "bot");
            msgEl.textContent = msg.message;

            miniChat.appendChild(timeEl);
            miniChat.appendChild(msgEl);
        });

        panelConversation.scrollTop = panelConversation.scrollHeight;
    }

    /**
     * Render the collected contact details in the Contact tab.
     * @param {Object|null} contact - contact_detail object from the API, or null.
     */
    function renderContact(contact) {
        contactDetail.innerHTML = "";

        if (!contact) {
            contactDetail.innerHTML = `
                <div class="no-contact-msg">
                    <span class="no-contact-icon">⊘</span>
                    <p>No contact details were collected in this session.</p>
                </div>
            `;
            return;
        }

        const fields = [
            { icon: "👤", label: "Full Name",    value: contact.name },
            { icon: "📞", label: "Phone",        value: contact.phone },
            { icon: "✉",  label: "Email",        value: contact.email },
            { icon: "📝", label: "Description",  value: contact.description || "—" },
            { icon: "🕐", label: "Collected At", value: formatDateTime(contact.collected_at) },
        ];

        const card = document.createElement("div");
        card.className = "contact-card";

        fields.forEach(function (f) {
            const row = document.createElement("div");
            row.className = "contact-field";
            row.innerHTML = `
                <div class="cf-icon">${f.icon}</div>
                <div class="cf-body">
                    <span class="cf-label">${escapeHtml(f.label)}</span>
                    <span class="cf-value">${escapeHtml(String(f.value))}</span>
                </div>
            `;
            card.appendChild(row);
        });

        contactDetail.appendChild(card);
    }

    /**
     * Render error entries inside the errors panel.
     */
    function renderErrors(errors) {
        errorList.innerHTML = "";

        if (!errors || errors.length === 0) {
            errorList.innerHTML = `
                <div class="no-errors-msg">
                    <span class="no-errors-icon">✓</span>
                    <p>No errors recorded for this session.</p>
                </div>
            `;
            return;
        }

        errors.forEach(function (err) {
            const entry = document.createElement("div");
            entry.className = "error-entry";

            let tracebackHtml = "";
            if (err.traceback) {
                tracebackHtml = `<div class="error-traceback">${escapeHtml(err.traceback)}</div>`;
            }

            entry.innerHTML = `
                <div class="error-type">${escapeHtml(err.error_type)}</div>
                <div class="error-message">${escapeHtml(err.error_message)}</div>
                ${tracebackHtml}
                <div class="error-time">${formatDateTime(err.timestamp)}</div>
            `;

            errorList.appendChild(entry);
        });
    }

    /**
     * Update pagination controls.
     */
    function renderPagination(page, totalPages) {
        pageInfo.textContent = "Page " + page + " of " + (totalPages || 1);
        prevPageBtn.disabled = page <= 1;
        nextPageBtn.disabled = page >= totalPages;
    }

    prevPageBtn.addEventListener("click", function () {
        if (currentPage > 1) fetchLogs(currentPage - 1);
    });

    nextPageBtn.addEventListener("click", function () {
        fetchLogs(currentPage + 1);
    });

    // --- Utility functions ---

    function showLoading(loading) {
        logsLoading.style.display = loading ? "flex" : "none";
        if (loading) {
            tableWrapper.style.display = "none";
            emptyState.style.display = "none";
        }
    }

    function showEmpty(message) {
        emptyState.style.display = "block";
        tableWrapper.style.display = "none";
        if (message) emptyState.querySelector("p").textContent = message;
    }

    function buildStatusBadge(status) {
        const classMap = {
            active:    "badge-active",
            completed: "badge-completed",
            abandoned: "badge-abandoned",
        };
        const cssClass = classMap[status] || "badge-active";
        return `<span class="badge ${cssClass}">${status}</span>`;
    }

    function formatDateTime(isoString) {
        if (!isoString) return "—";
        try {
            const normalized = /[Z+\-]\d*$/.test(isoString.trim())
                ? isoString
                : isoString + "Z";
            const d = new Date(normalized);
            if (isNaN(d.getTime())) return isoString;
            return d.toLocaleDateString() + " " + d.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
            });
        } catch (e) {
            return isoString;
        }
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }
})();