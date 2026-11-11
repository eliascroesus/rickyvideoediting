# Notes for Claude: video editing repo

- Raw footage: `raw-videos/` (Git LFS). After cloning, fetch it with `git lfs install && git lfs pull`
  (a fresh clone may only contain LFS pointer files). A note `<clip-name>.md` next to a clip, if present,
  describes the edit the user wants.
- Large raw videos may instead be attached to the GitHub Release tagged `raw-videos`
  (browser upload, up to 2 GB per file). Download with
  `curl -L -o raw-videos/<file> https://github.com/eliascroesus/rickyvideoediting/releases/download/raw-videos/<file>`
  (asset names with spaces become dots on GitHub; list them via the releases API if unsure).
- Finished edits go in `edited-videos/` (also LFS via `.gitattributes`). Next to each edit, list
  the music credit lines it needs (Kevin MacLeod tracks are CC BY 4.0, see `assets/CREDITS.md`).
- Asset catalog with "use it for" notes: `assets/README.md`. Music is MP3, SFX are 48 kHz WAV.
- `ffmpeg`/`ffprobe` are available. Regenerate the cinematic SFX with
  `python tools/synth_cinematic.py assets/sfx/cinematic-generated` (needs numpy + scipy).
- Browser uploads can bypass LFS and land as regular blobs. If `git lfs` warns that files
  "should have been pointers", fix them with `git add --renormalize .` and commit.
