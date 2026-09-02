// A message-style chat view for the demo. In the public build it replays a recorded conversation
// the visitor steps through (this file, plus transcript.ts); when a local backend is present, full
// mode drives the same bubbles live (see plan.md §10, session 8). The public demo still makes no
// LLM calls and ships no key — this recording is exactly that: a recording, labelled as one, so the
// browser build can show what the phrased, free-text conversation feels like without faking it live.
//
// The stepping is a pure TranscriptPlayer (unit-tested); CannedChat is the thin DOM shell around it,
// and entryHtml renders one bubble — the piece full mode will reuse.

export interface PickLine {
  title: string;
  year?: number;
  reason: string;
  wildcard?: boolean;
}

/** One turn in the conversation: something they said, something you said, or the final picks. */
export type ChatEntry =
  | { from: "them"; text: string }
  | { from: "you"; text: string }
  | { from: "them"; header: string; picks: PickLine[] };

function isPicks(entry: ChatEntry): entry is { from: "them"; header: string; picks: PickLine[] } {
  return "picks" in entry;
}

/** Progressive reveal of a fixed transcript: a cursor into the entries that steps forward and back. */
export class TranscriptPlayer {
  private cursor: number;

  constructor(private readonly entries: readonly ChatEntry[]) {
    this.cursor = entries.length ? 1 : 0;
  }

  get total(): number {
    return this.entries.length;
  }

  get step(): number {
    return this.cursor;
  }

  get atStart(): boolean {
    return this.cursor <= 1;
  }

  get atEnd(): boolean {
    return this.cursor >= this.entries.length;
  }

  visible(): ChatEntry[] {
    return this.entries.slice(0, this.cursor);
  }

  next(): boolean {
    if (this.atEnd) return false;
    this.cursor++;
    return true;
  }

  back(): boolean {
    if (this.atStart) return false;
    this.cursor--;
    return true;
  }

  restart(): void {
    this.cursor = this.entries.length ? 1 : 0;
  }
}

/** The recorded chat: a labelled recording the visitor steps through, mounted into ``mount``. */
export class CannedChat {
  private player: TranscriptPlayer;

  constructor(
    private mount: HTMLElement,
    entries: readonly ChatEntry[],
  ) {
    this.player = new TranscriptPlayer(entries);
  }

  render(): void {
    const bubbles = this.player.visible().map(entryHtml).join("");
    const controls = this.player.atEnd
      ? `<button class="btn" data-restart><span class="mono">Restart</span></button>`
      : `<button class="btn btn-accent" data-next><span class="mono">Next</span><span class="chev" aria-hidden="true">›</span></button>`;
    this.mount.innerHTML = `
      <div class="chat-banner">
        <span class="sq yellow"></span>
        <span class="mono">A recording — the full version's phrased conversation. The demo above is the live one.</span>
      </div>
      <div class="chat-log">${bubbles}</div>
      <div class="chat-controls">
        <button class="btn" data-back ${this.player.atStart ? "disabled" : ""}>
          <span class="chev" aria-hidden="true">‹</span><span class="mono">Back</span>
        </button>
        <span class="chat-progress mono">${this.player.step} / ${this.player.total}</span>
        ${controls}
      </div>`;

    this.mount.querySelector("[data-next]")?.addEventListener("click", () => this.step(() => this.player.next(), true));
    this.mount.querySelector("[data-back]")?.addEventListener("click", () => this.step(() => this.player.back(), false));
    this.mount.querySelector("[data-restart]")?.addEventListener("click", () => this.step(() => this.player.restart(), false));
  }

  private step(move: () => void, toBottom: boolean): void {
    move();
    this.render();
    if (toBottom) {
      const log = this.mount.querySelector<HTMLElement>(".chat-log");
      if (log) log.scrollTop = log.scrollHeight;
    }
  }
}

/** Render one entry as a chat bubble. Shared with full mode, which feeds live turns through it. */
export function entryHtml(entry: ChatEntry): string {
  if (isPicks(entry)) {
    const cards = entry.picks.map(pickHtml).join("");
    return `
      <div class="chat-row them">
        <div class="chat-bubble picks-bubble">
          <div class="chat-head mono"><span class="sq magenta"></span>${escapeHtml(entry.header)}</div>
          <div class="chat-picks">${cards}</div>
        </div>
      </div>`;
  }
  const who = entry.from === "you" ? "you" : "them";
  return `<div class="chat-row ${who}"><div class="chat-bubble">${escapeHtml(entry.text)}</div></div>`;
}

function pickHtml(pick: PickLine): string {
  const year = pick.year ? ` <span class="year">${pick.year}</span>` : "";
  const flag = pick.wildcard ? ` <span class="tag mono">wildcard</span>` : "";
  return `
    <div class="chat-pick${pick.wildcard ? " wild" : ""}">
      <div class="cp-title">${escapeHtml(pick.title)}${year}${flag}</div>
      <div class="cp-reason">${escapeHtml(pick.reason)}</div>
    </div>`;
}

export function escapeHtml(s: string): string {
  return s.replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!,
  );
}
