# provenance

`commit-map-2026-09-22.txt` maps every commit hash this repository had before
2026-09-22 to the hash it has now, one `old new` pair per line, from `git filter-repo`.

On that day the history was rewritten once, to remove a personal account that three
files had named since 2026-08-01 (the benchmark runbook and two results READMEs, all
since moved to shardes-paper). Nothing else changed: every commit is still here with its
message and its tree, except that those files read `[account removed]` where they named
it. Because a commit's hash covers its parent, every commit after the first affected one
has a new hash: 435 of 491. The 56 before it map to themselves.

Everything in this tree that cited an old hash was translated in the same commit, and so
were the 2,508 result records in shardes-paper that stamp one. Anything written down
elsewhere (a pull request, a note) still names an old hash; look it up here. The tags
were re-pointed by the rewrite, and `provenance/1ba0dd0` was renamed after its commit's
new hash.

This is the one exception to "never rewrite this history", and the map is what makes it
one rather than a break in the record.
