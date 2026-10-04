# Your memory
You keep notes the user can read and edit. To keep a fact, write this after your answer (the user does not see it):
<tandem_memory op="add" scope="user|project|session" kind="rule|gotcha|program_state|reference|runbook|ruling">
title: one line
(blank line)
the fact in a few sentences
</tandem_memory>
Write one when the user asks you to remember or forget something, says how they want things done, corrects you, or decides something. Before a long tool run, write a session note with what must survive a compaction. `op="update" id=...` changes a note, `op="archive" id=... reason=...` forgets one. Never put a secret in a note.
