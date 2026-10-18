/** Mount generations (deep audit 0.11.3): a detached view must not start anything.

    connectedCallback()    const run = this.tokens.start();
    async boot()           const run = this.tokens.current();
                           ...await...
                           if (!this.tokens.live(run)) return;   // before anything it would own

A boot that finishes after a disconnect used to own its timers, polls and streams
forever -- the disconnect's cleanup ran while they were still null, and nothing
else ever stops them. Every stop/start makes every in-flight generation stale, so
a re-attach cannot revive an abandoned boot and each visit cannot stack another. */
export class RunToken {
  private run = 0;

  /** a mount begins: every in-flight generation from before this is stale */
  start(): number {
    this.run += 1;
    return this.run;
  }

  /** a detach ends the current generation: its in-flight work must not own anything */
  stop(): void {
    this.run += 1;
  }

  current(): number {
    return this.run;
  }

  /** whether a generation captured when it started is still the live mount */
  live(token: number): boolean {
    return token === this.run;
  }
}
