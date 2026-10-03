const state = {
  models: [],
  history: [],
  pinned: null,
  busy: false,
};

const $ = (id) => document.getElementById(id);

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, reducedMotion ? 0 : ms));

// ---------- Formatting ----------

function splitName(model) {
  const name = (model.name || model.id).replace(/\s*\(free\)\s*$/i, "");
  const [provider, ...rest] = name.split(": ");
  return rest.length ? { provider, title: rest.join(": ") } : { provider: model.id.split("/")[0], title: name };
}

function formatContext(tokens) {
  if (!tokens) return "—";
  if (tokens >= 1_000_000) return `${+(tokens / 1_000_000).toFixed(1)}M`;
  return `${Math.round(tokens / 1000)}K`;
}

function formatDate(value) {
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function timeAgo(iso) {
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  const units = [["day", 86400], ["hour", 3600], ["minute", 60]];
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  for (const [unit, size] of units) {
    if (seconds >= size) return rtf.format(-Math.floor(seconds / size), unit);
  }
  return "just now";
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  node.append(...children.filter((child) => child != null));
  return node;
}

// ---------- Data ----------

async function getJSON(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const error = body?.error || {};
    const message = error.message || `Request failed with HTTP ${response.status}`;
    throw Object.assign(new Error(message), { type: error.type, attempts: error.attempts || [], response });
  }
  return { body, response };
}

async function load() {
  const [models, history] = await Promise.all([getJSON("/api/models"), getJSON("/api/history")]);
  state.models = models.body;
  state.history = history.body;
  if (state.pinned && !state.models.some((model) => model.id === state.pinned)) state.pinned = null;
  render();
}

// ---------- Rendering ----------

function render() {
  renderStatus();
  renderJacks();
  renderHistory();
  renderModels();
  renderRouteMode();
}

function renderStatus() {
  const last = state.history.at(-1);
  const count = `${state.models.length} free model${state.models.length === 1 ? "" : "s"}`;
  $("sync-status").textContent = last ? `${count}, synced ${timeAgo(last.synced_at)}` : `${count}. Sync to fetch the list.`;
}

function latestAdded() {
  // The very first sync "adds" everything, which isn't news.
  if (state.history.length < 2) return new Set();
  return new Set(state.history.at(-1).added);
}

function renderJacks() {
  const list = $("jacks");
  const fresh = latestAdded();

  if (!state.models.length) {
    list.replaceChildren(el("li", { class: "empty" }, "No free models yet. Use Sync now to fetch them from OpenRouter."));
    return;
  }

  list.replaceChildren(
    ...state.models.map((model) => {
      const { provider, title } = splitName(model);
      const pinned = state.pinned === model.id;
      const button = el(
        "button",
        {
          type: "button",
          class: ["jack", fresh.has(model.id) && "jack--new", pinned && "jack--pinned"].filter(Boolean).join(" "),
          "data-id": model.id,
          "aria-pressed": String(pinned),
          title: model.id,
        },
        el("span", { class: "socket", "aria-hidden": "true" }),
        el(
          "span",
          { class: "jack__label" },
          el("span", { class: "jack__provider" }, provider),
          el("span", { class: "jack__name" }, title),
          el("span", { class: "jack__code" }),
        ),
      );
      button.addEventListener("click", () => togglePin(model.id));
      return el("li", {}, button);
    }),
  );
}

function renderRouteMode() {
  const model = state.models.find((m) => m.id === state.pinned);
  $("route-mode").textContent = model ? `Pinned: ${splitName(model).title}` : "Auto: any free line";
  $("board-hint").textContent = model ? "Pick the pinned line again to go back to auto." : "Pick a line to pin it, or leave it on auto.";
}

function renderHistory() {
  const list = $("history");
  const entries = [];

  // Fold runs of no-change syncs into one line so real changes stand out.
  for (const entry of [...state.history].reverse()) {
    const changed = entry.added.length || entry.removed.length;
    const previous = entries.at(-1);
    if (!changed && previous && !previous.changed) {
      previous.count += 1;
    } else {
      entries.push({ entry, changed, count: 1 });
    }
  }

  if (!entries.length) {
    list.replaceChildren(el("li", {}, el("p", {}, "No syncs yet.")));
    return;
  }

  list.replaceChildren(
    ...entries.map(({ entry, changed, count }) => {
      const when = el("time", { datetime: entry.synced_at }, formatDate(entry.synced_at));
      if (!changed) {
        const span = count > 1 ? `${count} syncs, no changes` : "No changes";
        return el("li", {}, when, el("p", {}, `${span}. ${entry.total_free_models} models.`));
      }
      return el(
        "li",
        { class: "has-change" },
        when,
        el("p", {}, `${entry.added.length} added, ${entry.removed.length} removed. ${entry.total_free_models} models.`),
        el(
          "ul",
          { class: "change-list" },
          ...entry.added.map((id) => el("li", { class: "added" }, `+ ${id}`)),
          ...entry.removed.map((id) => el("li", { class: "removed" }, `- ${id}`)),
        ),
      );
    }),
  );
}

function renderModels() {
  const query = $("filter").value.trim().toLowerCase();
  const rows = state.models.filter((model) => `${model.id} ${model.name}`.toLowerCase().includes(query));

  if (!rows.length) {
    $("models").replaceChildren(el("tr", {}, el("td", { colspan: "4", class: "empty" }, query ? "No models match that filter." : "No models yet.")));
    return;
  }

  $("models").replaceChildren(
    ...rows.map((model) => {
      const { provider, title } = splitName(model);
      const inputs = model.architecture?.input_modalities?.join(", ") || "text";
      return el(
        "tr",
        {},
        el("td", {}, `${provider} ${title}`, el("code", {}, model.id)),
        el("td", { class: "num" }, formatContext(model.context_length)),
        el("td", {}, inputs),
        el("td", {}, model.created ? formatDate(model.created) : "—"),
      );
    }),
  );
}

// ---------- Interaction ----------

function togglePin(id) {
  state.pinned = state.pinned === id ? null : id;
  renderJacks();
  renderRouteMode();
}

function jack(id) {
  return document.querySelector(`.jack[data-id="${CSS.escape(id)}"]`);
}

function clearSignals() {
  document.querySelectorAll(".jack").forEach((node) => {
    node.classList.remove("jack--live", "jack--busy", "jack--scan");
    node.querySelector(".jack__code").textContent = "";
  });
}

function startScan() {
  if (reducedMotion) return () => {};
  const nodes = [...document.querySelectorAll(".jack")];
  let index = 0;
  const timer = setInterval(() => {
    nodes.forEach((node) => node.classList.remove("jack--scan"));
    nodes[index % nodes.length]?.classList.add("jack--scan");
    index += 1;
  }, 140);
  return () => {
    clearInterval(timer);
    nodes.forEach((node) => node.classList.remove("jack--scan"));
  };
}

async function replayRoute(attempts, answered) {
  for (const attempt of attempts) {
    const node = jack(attempt.model);
    if (!node) continue;
    node.classList.add("jack--busy");
    node.querySelector(".jack__code").textContent = attempt.status ? `HTTP ${attempt.status}` : "No response";
    await wait(180);
  }
  if (answered) jack(answered)?.classList.add("jack--live");
}

function showReply(meta, text, isError = false) {
  const reply = $("reply");
  reply.hidden = false;
  reply.classList.toggle("reply--error", isError);
  $("reply-meta").replaceChildren(...meta);
  $("reply-text").textContent = text;
}

function describeRoute(model, skipped) {
  const parts = ["Answered by ", el("code", {}, model)];
  if (skipped) parts.push(` after skipping ${skipped} busy line${skipped === 1 ? "" : "s"}.`);
  return parts;
}

async function send(event) {
  event.preventDefault();
  if (state.busy) return;

  const prompt = $("prompt").value.trim();
  if (!prompt) return;

  state.busy = true;
  $("send-button").disabled = true;
  $("send-button").textContent = "Routing…";
  clearSignals();
  const stopScan = startScan();

  try {
    const { body, response } = await getJSON("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: state.pinned || "auto",
        messages: [{ role: "user", content: prompt }],
      }),
    });
    stopScan();
    const attempts = JSON.parse(response.headers.get("X-Router-Attempts") || "[]");
    await replayRoute(attempts, body.model);
    showReply(describeRoute(body.model, attempts.length), body.choices?.[0]?.message?.content || "(Empty reply)");
  } catch (error) {
    stopScan();
    const attempts = error.attempts || [];
    await replayRoute(attempts, null);
    if (error.type === "all_models_failed") {
      const hint = `No line answered (${attempts.length} tried). Try again in a minute or pin a different model.`;
      showReply(["No free model answered."], [hint, attempts.at(-1)?.error].filter(Boolean).join("\n\n"), true);
    } else {
      showReply(["The request didn't go through."], error.message, true);
    }
  } finally {
    state.busy = false;
    $("send-button").disabled = false;
    $("send-button").textContent = "Send";
  }
}

async function sync() {
  const button = $("sync-button");
  button.disabled = true;
  button.textContent = "Syncing…";
  try {
    const { body } = await getJSON("/api/sync", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    await load();
    const changes = body.added.length + body.removed.length;
    $("sync-status").textContent = changes
      ? `Synced: ${body.added.length} added, ${body.removed.length} removed`
      : `Synced: no changes, ${body.total_free_models} free models`;
  } catch (error) {
    $("sync-status").textContent = `Sync failed: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "Sync now";
  }
}

// ---------- Boot ----------

document.querySelectorAll(".origin").forEach((node) => (node.textContent = location.origin));
$("console").addEventListener("submit", send);
$("prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) $("console").requestSubmit();
});
$("sync-button").addEventListener("click", sync);
$("filter").addEventListener("input", renderModels);

load().catch((error) => {
  $("sync-status").textContent = `Couldn't load models: ${error.message}`;
});
