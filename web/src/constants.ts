// The canonical scoring constants, read from spec/constants.json — the single source of truth both
// the Python CLI and this demo share (plan.md §8.6). The values live in the Python modules at
// runtime and are mirrored into that JSON by `make spec`; the demo imports them here so a constant
// changed on the Python side and regenerated flows into the browser with no second edit. Editing a
// value here instead would just be overwritten on the next regeneration — change it in Python.

import constants from "../../spec/constants.json";
import excludedTags from "../../spec/excluded_tags.json";

export const SCORING = constants.scoring;
export const DECAY = constants.decay;
export const PROBES = constants.probes;
export const COVERAGE = constants.coverage;

// Genome tags that describe a film's reception or a viewer's verdict rather than its content, so
// matching taste on them is disingenuous (spec/excluded_tags.json, the mirror of the Python
// cinegeist.catalog.excluded_tags). The demo drops them from the top tags it shows.
export const EXCLUDED_TAGS: ReadonlySet<string> = new Set(excludedTags.excluded_tags);
