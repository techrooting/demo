---
name: load-trace-demo
description: Show how Deep Agents discovers a skill, reads SKILL.md, then loads references, scripts, and assets.
compatibility: deepagents>=0.7.26
---
# Load Trace Demo

Read these files, then report load order and result:
- `/skills/load-trace-demo/references/loading-contract.md`
- `/skills/load-trace-demo/scripts/summarize_skill_payload.py`
- `/skills/load-trace-demo/assets/load-map.json`

Meaning: metadata comes from scanning `/skills` and parsing `SKILL.md` frontmatter; this file and support files are loaded only after explicit reads.
