import type { ChatEntry } from "./chat.ts";

// A REAL recording of the full CLI, not authored prose. It was produced by running `cinegeist serve`
// online against the full ~86,500-film catalog and stepping an inarticulate persona through it: the
// greeting, the two free-text seed questions, the model's own title disambiguation, its phrased pair
// questions, and its picks with explanations are all exactly what the tool emitted (plan.md §8.1,
// §10). The model that phrased and explained is named in TRANSCRIPT_MODEL and credited in the UI.
//
// It is lightly condensed for the page — the disambiguation of a couple of the seed films and the
// runtime/subtitle questions are dropped — but every line shown is verbatim, nothing is reworded.
//
// The persona has a distinctive taste they can't articulate: slow, atmospheric dread that resolves
// (folk horror, occult, bleak tension), and a hard bounce off beautiful-but-plotless art films. They
// answer the seed questions in vivid shrugs and name films without being able to say why; the tool
// reads the pattern and returns three confident picks. It produced no wildcard — a wildcard needs a
// film far from the taste yet sharing its strong tags, and this focused taste has none in range. That
// honest absence is the real output, left as it came.
export const TRANSCRIPT_MODEL = "openai/gpt-5.6-luna";

export const TRANSCRIPT: readonly ChatEntry[] = [
  {
    from: "them",
    text: "Let's find you something to watch. I'll ask you to react to a few real films rather than describe what you like — people are much better at the first.",
  },
  { from: "them", text: "Name two or three films you've genuinely loved — any era, any genre." },
  {
    from: "you",
    text: "God, I'm so bad at this. Um... The Witch — that one just crawled under my skin and stayed there. And Hereditary, I genuinely had to sleep with a light on after. And Sicario, weirdly, I've watched it like four times? I couldn't tell you what ties them together, they just get to me somewhere I can't explain.",
  },
  { from: "them", text: "Which “The Witch” did you mean?" },
  { from: "you", text: "The Witch (2015)" },
  { from: "them", text: "Anything you started and turned off, or that everyone loves and you just didn't?" },
  {
    from: "you",
    text: "Yeah — I turned off Under the Skin about halfway, I could not tell you what was happening or why. And The Tree of Life, everyone acts like it's a masterpiece and I was just bored out of my mind — it's gorgeous but it felt like a screensaver, nothing grabbed me.",
  },
  { from: "them", text: "Which would you rather put on tonight: The Devil's Candy or Unsane?" },
  { from: "you", text: "The Devil's Candy" },
  {
    from: "them",
    text: "In the mood for eerie chills or darkly quirky western tales tonight — Ringu (Ring) or The Ballad of Buster Scruggs?",
  },
  { from: "you", text: "Ringu (Ring)" },
  {
    from: "them",
    header: "Your picks",
    picks: [
      {
        title: "The Silence of the Lambs",
        year: 1991,
        reason:
          "The Witch — “that one just crawled under my skin and stayed there” — points to the horror and suspenseful tension this shares with your taste.",
      },
      {
        title: "Don't Breathe",
        year: 2016,
        reason:
          "Since you’ve watched Sicario “like four times,” this tense, suspense-driven pick seems suited to the kind of repeatable pressure you return to.",
      },
      {
        title: "Green Room",
        year: 2015,
        reason:
          "This sits most directly on the horror side of the match, while keeping the suspense tense rather than treating it as a purely atmospheric pick.",
      },
    ],
  },
];
