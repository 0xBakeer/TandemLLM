# Design: Simple mode for the Setup Guide

The Setup Guide speaks to two audiences with one machine: the person who wants a working
chat in three pages, and the person who wants every knob. It is one guide with one state
machine, not a second installer: plan, estimate, physics guard, apply job, progress and
review are reused exactly as the advanced flow runs them. No server route, registry field
or engine concept is added.

## The switch

Simple|Advanced sits above the wizard's own steps. The choice is a browser memory
(`localStorage`, key `tandem.setup.simple`) — a fresh browser opens in simple mode; a
browser once switched remembers. Entering a mode re-derives the current step from the
mode being entered: a step the fold holds collapses onto the folded page, the tiles'
twin (the standard intro) opens on the quant cards, and the tiles and the running install
are pages of both modes.

## The simple flow

Three pages, one list: the model tiles, the quant cards, the folded settings page.

1. **The tiles.** The models this box can serve, one card each, recommended first.
2. **The quant cards.** The model's profiles, the recommended one preselected; the card
   says what the choice costs (download, memory) in the same words the advanced page uses.
3. **The settings page.** One fold per topic — components, the chats and their window,
   methods, the review — the
   step's whole body behind its title bar, every fold holding what the advanced step shows.
   The review fold opens by default, so a Finish that waits always says so in the open.

A fourth page lives outside the walk: the install job, reached by Finish and returning to
the folded page when it ends.

A first-opened model with nothing running yet gets the simple answers — the model's full
window, as many chats at it as the card measured it can carry (never past the card's own
measurement, `slots.max`), the deciders and the response cache where the model offers
them. The pool is turned on only when it buys chats the plain reservation cannot carry.
A box that already serves the model keeps its own settings: opening the guide in a new
browser must never change what is running.

A disabled Finish is never silent: the blockers ride beside the button, the plan's own
errors included, and a plan check that has not answered says it is still checking — or
that it did not answer and asks for a Retry.

The Advanced escape hatch shows inside a model — on the quant cards and on the settings
page, not on the tiles page; it is the same switch, so there is one way back and no
orphaned state.

## What advanced keeps

Everything: the rail, all steps, every field and help, Wren, the folded pages. The e2e
files that drive the advanced flow ask for it by name, since a fresh browser now lands in
simple mode; everything else keeps the guide's default.

## Why not a separate simple installer

A second flow means two state machines to keep in step, two places where a stale response
can write, and a wall between the two audiences the same person crosses in one evening.
Here the mode is one input to the reducer — `activeSteps` and `normalizeStep` decide which
pages exist and where a step belongs — and the install job belongs to the model that
started it: opening another model leaves the job on its own model's progress page.

## Tests

`dashboard/tests/setup-state.test.ts` covers the mode switch, the active page list, the
route normalization, resume in the right mode, the job's model scope and the simple
defaults (the card's cap, the pool only when it pays). `dashboard/e2e/setup-simple.spec.ts`
walks the three pages on the mock and finishes on the review fold.
