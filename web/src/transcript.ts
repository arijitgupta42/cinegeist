import type { ChatEntry } from "./chat.ts";

// A recorded full-version conversation, shown in the demo's chat view. It is authored, not computed:
// the demo above it is the live recommender with the words stripped out, and this stands in for the
// phrased, free-text experience the browser build deliberately leaves out — no LLM, no key (plan.md
// §8.1, §8.2). It is labelled in the UI as a recording, never presented as live. The films are real.
//
// The persona: someone who likes mounting, atmospheric dread but wants the ending to resolve, and
// cools on films that stay deliberately ambiguous — so the picks cite that back, and the wildcard
// deliberately pushes on the edge they named.
export const TRANSCRIPT: readonly ChatEntry[] = [
  {
    from: "them",
    text: "Let's find you something to watch. I'll show you pairs of real films and learn from which you'd rather put on tonight — no need to describe your taste, just react.",
  },
  { from: "them", text: "Would you rather put on Hereditary or Now You See Me?" },
  { from: "you", text: "Hereditary, easily. I love that slow, creeping dread." },
  { from: "them", text: "Noted — atmospheric and slow-building over slick and plot-driven." },
  { from: "them", text: "Between The Witch and John Wick?" },
  { from: "you", text: "The Witch — though I'll admit its ending left me a little cold." },
  { from: "them", text: "Good to know: dread yes, but you want the ending to land, not dissolve." },
  { from: "them", text: "Last pair — Under the Skin or Prisoners?" },
  { from: "you", text: "Prisoners. Under the Skin is gorgeous but too oblique for me." },
  {
    from: "them",
    text: "That's enough to place you. You like mounting dread with a resolution you can hold onto, and you cool on films that stay deliberately ambiguous.",
  },
  {
    from: "them",
    header: "Your picks",
    picks: [
      {
        title: "The Silence of the Lambs",
        year: 1991,
        reason:
          "The same tightening dread you chose in Hereditary and Prisoners — but it resolves cleanly, with none of the ambiguity you said annoys you.",
      },
      {
        title: "Sinister",
        year: 2012,
        reason:
          "Slow-building supernatural dread like The Witch, with an ending that commits rather than dissolves.",
      },
      {
        title: "Zodiac",
        year: 2007,
        reason:
          "The methodical, oppressive tension you picked in Prisoners — grounded, precise, and paid off.",
      },
      {
        title: "The Wailing",
        year: 2016,
        reason:
          "Further out, a deliberate stretch: it shares the mounting dread you keep choosing, but its ending is divisive — right on the edge you told me about.",
        wildcard: true,
      },
    ],
  },
];
