// Full mode: when the demo is served by a local `cinegeist serve` backend, the "A full conversation"
// section stops being a recording and becomes the real conversation, driven over the backend's JSON
// API (plan.md §10, session 8). The public Pages build has no backend — the health probe fails and
// the section falls back to the canned recording (chat.ts). This adds no key and no LLM to the static
// bundle: full mode only ever runs against a backend the visitor set up on their own machine, which
// holds the catalog and the key. The bubbles are the same ones the recording uses (entryHtml).

import { entryHtml, escapeHtml, type ChatEntry, type PickLine } from "./chat.ts";

export interface Health {
  ok: boolean;
  offline?: boolean;
  films?: number;
}

interface ApiPick {
  id: number;
  title: string;
  year: number | null;
  explanation: string;
  wildcard: boolean;
}

interface ApiMessage {
  type: "say" | "picks" | "error";
  text?: string;
  header?: string;
  picks?: ApiPick[];
  wildcard?: ApiPick | null;
}

interface ApiPrompt {
  kind: "text" | "choice";
  text: string;
  options?: { key: string; label: string }[];
}

interface ApiTurn {
  session_id: string;
  messages: ApiMessage[];
  prompt: ApiPrompt | null;
  done: boolean;
  error?: boolean;
}

/** Map the backend's turn messages to chat bubbles. Pure, so it's the part that's unit-tested. */
export function apiMessagesToEntries(messages: ApiMessage[]): ChatEntry[] {
  return messages.map(apiMessageToEntry);
}

function apiMessageToEntry(message: ApiMessage): ChatEntry {
  if (message.type === "picks") {
    const picks: PickLine[] = (message.picks ?? []).map((p) => apiPickToLine(p, false));
    if (message.wildcard) picks.push(apiPickToLine(message.wildcard, true));
    return { from: "them", header: message.header ?? "Your picks", picks };
  }
  // say and error both read as something the recommender said.
  return { from: "them", text: message.text ?? "" };
}

function apiPickToLine(pick: ApiPick, wildcard: boolean): PickLine {
  return {
    title: pick.title,
    year: pick.year ?? undefined,
    reason: pick.explanation,
    wildcard: wildcard || pick.wildcard,
  };
}

/** Probe for a local backend. Returns its health, or ``null`` if there's no backend (the Pages case). */
export async function detectBackend(timeoutMs = 1500): Promise<Health | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch("api/health", { signal: controller.signal });
    if (!response.ok) return null;
    const health = (await response.json()) as Health;
    return health && health.ok ? health : null;
  } catch {
    return null; // no backend, a 404, or a timeout — all mean "run the recording instead"
  } finally {
    clearTimeout(timer);
  }
}

/** The live conversation, driven over the backend API and rendered as the same chat bubbles. */
export class LiveChat {
  private log: ChatEntry[] = [];
  private sessionId: string | null = null;
  private prompt: ApiPrompt | null = null;
  private done = false;
  private busy = false;

  constructor(
    private mount: HTMLElement,
    private health: Health,
  ) {}

  async start(): Promise<void> {
    this.log = [];
    this.sessionId = null;
    this.prompt = null;
    this.done = false;
    this.busy = true;
    this.render();
    try {
      const turn = await this.post("api/session", {});
      this.busy = false;
      this.applyTurn(turn);
    } catch (error) {
      this.busy = false;
      this.fail(error);
    }
  }

  private applyTurn(turn: ApiTurn): void {
    this.sessionId = turn.session_id;
    for (const entry of apiMessagesToEntries(turn.messages)) this.log.push(entry);
    this.prompt = turn.prompt;
    this.done = turn.done;
    if (turn.prompt) this.log.push({ from: "them", text: turn.prompt.text });
    this.render();
  }

  private async answer(key: string, label: string): Promise<void> {
    if (this.busy || !this.sessionId || !this.prompt) return;
    this.busy = true;
    this.log.push({ from: "you", text: label });
    this.prompt = null;
    this.render();
    try {
      const turn = await this.post(`api/session/${this.sessionId}/answer`, { answer: key });
      this.busy = false;
      this.applyTurn(turn);
    } catch (error) {
      this.busy = false;
      this.fail(error);
    }
  }

  private render(): void {
    const bubbles = this.log.map(entryHtml).join("");
    this.mount.innerHTML = `
      <div class="chat-banner live">
        <span class="sq green"></span>
        <span class="mono">Live — the full recommender, running on this machine${
          this.health.offline ? " (offline)" : ""
        }.</span>
      </div>
      <div class="chat-log">${bubbles}</div>
      ${this.footer()}`;
    this.wire();
    const log = this.mount.querySelector<HTMLElement>(".chat-log");
    if (log) log.scrollTop = log.scrollHeight;
  }

  private footer(): string {
    if (this.busy) {
      return `<div class="chat-thinking mono"><span class="dot"></span> thinking…</div>`;
    }
    if (this.done || !this.prompt) {
      return `<div class="chat-controls"><button class="btn btn-accent" data-restart><span class="mono">Start again</span></button></div>`;
    }
    if (this.prompt.kind === "choice") {
      const options = (this.prompt.options ?? [])
        .map(
          (o) =>
            `<button class="chat-option btn" data-key="${escapeHtml(o.key)}">${escapeHtml(o.label)}</button>`,
        )
        .join("");
      return `<div class="chat-options">${options}</div>`;
    }
    return `
      <form class="chat-input" data-textform>
        <input type="text" name="answer" placeholder="Type your answer…" autocomplete="off" />
        <button class="btn btn-accent" type="submit"><span class="mono">Send</span></button>
      </form>`;
  }

  private wire(): void {
    this.mount.querySelectorAll<HTMLElement>(".chat-option").forEach((button) =>
      button.addEventListener("click", () => {
        const key = button.dataset.key ?? "";
        void this.answer(key, button.textContent?.trim() || key);
      }),
    );
    const form = this.mount.querySelector<HTMLFormElement>("[data-textform]");
    form?.addEventListener("submit", (event) => {
      event.preventDefault();
      const input = form.querySelector<HTMLInputElement>("input[name=answer]");
      const value = input?.value.trim() ?? "";
      if (value) void this.answer(value, value);
    });
    this.mount.querySelector("[data-restart]")?.addEventListener("click", () => void this.start());
  }

  private fail(error: unknown): void {
    this.log.push({ from: "them", text: `Something went wrong talking to the backend: ${String(error)}` });
    this.prompt = null;
    this.done = true;
    this.render();
  }

  private async post(path: string, body: unknown): Promise<ApiTurn> {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(`the backend returned ${response.status}`);
    return (await response.json()) as ApiTurn;
  }
}
