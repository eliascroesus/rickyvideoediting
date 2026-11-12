# how-much-youtube-paid-me_edit.mp4 (v3)

Edit of `This is how much youtubee paid me for 144m views` (raw: GitHub Release `raw-videos`).
**2560x1440 (native 1440p), 30 fps, x264 CRF 14**, about 10:01, audio at -14 LUFS.
Rebuild with `tools/precompute_paid_me.py` (person cutouts + paper stabilization), then
`tools/edit_youtube_paid_me.py` (video chunks + audio).

v3 is deliberately fairly raw: screen shares are full frame with no camera moves, no music and no captions.
The only highlights are the 122.8M zoom, the AdSense strike page, the $17K reveal, the $100 -> $500-$2,000 cards,
"$40,000", the STEP 01-04 cards, the 64M competitor cards and "join the Discord".

## Music credits (paste into the YouTube description)

```text
"Crypto" Kevin MacLeod (incompetech.com)
Licensed under Creative Commons: By Attribution 4.0
http://creativecommons.org/licenses/by/4.0/
```

The intense track is the "Me and the Devil" instrumental you supplied. It is not royalty-free, so
YouTube may flag it with Content ID (usually that means ad revenue goes to the rights holder, or the
track gets muted). It is not stored in this repo.

## Style system

- Cards: rounded rectangle, orange to black gradient, faded grid inside, glowing orange edge, shown over the
  footage with the background darkened, blurred and vignetted.
- Text: Inter Display SemiBold (white) + EB Garamond Bold Italic (gold, deep glow), per-letter write-on.
- Talking head: background blur + darken through a soft feathered ellipse around Ricky (no hard cutout).
- Paper shots: stabilized.
