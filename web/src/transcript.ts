import type { ChatEntry } from "./chat.ts";

// A REAL recording of the full CLI, not authored prose. It was produced by running `cinegeist serve`
// online against the full ~86,500-film catalog and stepping an inarticulate persona through it: the
// greeting, the two free-text seed questions, the model's own title disambiguation, its phrased pair
// questions, and its picks with explanations are all exactly what the tool emitted (plan.md §8.1,
// §10). The model that phrased and explained is named in TRANSCRIPT_MODEL and credited in the UI.
//
// It is lightly condensed for the page — a second seed-title disambiguation and the runtime/subtitle
// questions are dropped — but every model line shown is verbatim, nothing is reworded.
//
// The persona has a taste they can't articulate: gritty, violent frontier stories — modern crime and
// old westerns alike, No Country and Sicario sitting next to Bone Tomahawk — and a hard bounce off
// slow, pretty, "important" art films. They answer in shrugs and name films without being able to say
// why; the tool reads the through-line (brutal, tense, unsentimental) and returns three picks that
// each lean on a different film they named. It produced no wildcard — a wildcard needs a film far
// from the taste yet still sharing its strong tags, and this focused-but-real taste has none in
// range. That honest absence is the real output, left as it came.
export const TRANSCRIPT_MODEL = "openai/gpt-5.6-luna";

export const TRANSCRIPT: readonly ChatEntry[] = [
  {
    from: "them",
    text: "Let's find you something to watch. I'll ask you to react to a few real films rather than describe what you like — people are much better at the first.",
  },
  { from: "them", text: "Name two or three films you've genuinely loved — any era, any genre." },
  {
    from: "you",
    text: "Ugh, I'm genuinely bad at this. Okay — No Country for Old Men. Sicario, I've rewatched it way too many times. Hell or High Water. And this brutal western, Bone Tomahawk. I honestly couldn't tell you what ties them together — they just get me.",
  },
  { from: "them", text: "Which “Sicario” did you mean?" },
  { from: "you", text: "Sicario (2015)" },
  { from: "them", text: "Anything you started and turned off, or that everyone loves and you just didn't?" },
  {
    from: "you",
    text: "The Tree of Life — everyone calls it a masterpiece and I was bored stupid, gorgeous but nothing happens. And The Assassination of Jesse James, that long pretty Brad Pitt western — turned it off, way too slow.",
  },
  { from: "them", text: "Which would you rather put on tonight: Goodfellas or Unforgiven?" },
  { from: "you", text: "Unforgiven" },
  {
    from: "them",
    text: "Which are you in the mood for tonight — The Major or Wild Wild Country?",
  },
  { from: "you", text: "The Major" },
  {
    from: "them",
    header: "Your picks",
    picks: [
      {
        title: "Blue Ruin",
        year: 2013,
        reason:
          "Blue Ruin fits the brutal, violent strain behind your praise for Bone Tomahawk, with tension that connects to the kind of harshness you respond to there.",
      },
      {
        title: "One False Move",
        year: 1992,
        reason:
          "Given how often you return to Sicario, One False Move’s tense and violent register makes sense as another place to look for that same pressure.",
      },
      {
        title: "The Chaser (Chugyeogja)",
        year: 2008,
        reason:
          "The Chaser lines up with the violent, brutal edge you found in No Country for Old Men, while keeping the recommendation focused on tension rather than atmosphere.",
      },
    ],
  },
];
