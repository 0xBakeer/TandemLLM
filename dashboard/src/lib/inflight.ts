/** one flight at a time (deep audit 0.11.3): calls made while an async getter runs share it.

A refresh that checks 'am I done yet' but races itself hits the endpoint twice and lets
both answers write the same state. The getter runs once, forever after it settles the
answer is held by the caller anyway (the view stops asking once its state is filled). */
export function inflightOnce<T>(get: () => Promise<T>): () => Promise<T> {
  let p: Promise<T> | null = null;
  return () => {
    if (!p) p = get();
    return p;
  };
}
