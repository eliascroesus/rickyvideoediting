# Editing style guide (Ricky's YouTube videos)

This document explains, in plain language, how Ricky's long-form YouTube videos are edited. It was
written so that anyone (a human editor or another AI chat) can produce edits he will approve without
re-learning his preferences. Everything here comes from his direct feedback on real edits.

---

## 1. What his videos look like (the raw material)

- **Talking head:** Ricky at his desk, dark room, YouTube play buttons on the shelves behind him.
- **Paper explainers:** a top-down camera on a notepad where he writes and circles numbers.
- **Screen shares:** screen recordings (Shopify, ChatGPT, Claude, CapCut, YouTube Studio, product
  research tools) with his facecam in the bottom-right corner.
- The raw footage is already cut fairly tight. There is very little dead air to remove.
- The topic is usually e-commerce / dropshipping / YouTube money: the hook is a money number
  (e.g. "$40,000 in 28 days", "$17,000 from AdSense").

## 2. The golden rule: keep it raw, add only what earns its place

Ricky's feedback, again and again, was **"less"**: less camera movement, less text, fewer highlights,
no gimmicky sounds. When in doubt, leave the footage alone.

There are two approved levels. Ask which one he wants if it isn't clear; **default to "Raw"**.

### Level A: Raw (his latest preference, used for "How to make stores")

- Footage plays exactly as recorded: **no zooms, no colour grade, no darkening/vignette, no captions,
  no cards, no music**.
- The **only** additions are the **STEP pages** (see section 4) at the exact moment he says
  "step one / the first step / the next step...", with a soft whoosh in and out.
- Allowed invisible fixes: stabilize the paper camera (it jitters when he writes), trim pauses
  longer than ~0.6 s, remove echo, normalize loudness to -14 LUFS.

### Level B: Light highlights (used for "How much YouTube paid me")

Everything from Level A, plus a **small number** of highlights (roughly one every minute or two, never
stacked). Approved highlights:

- **One zoom on the key number in the hook** (e.g. push into "122.8M views" with the rest of the
  screen dimmed and a thin orange underline drawing under the number). He loved this.
- **Blur a money number until he says it**, then reveal it (unblur + zoom onto it).
- **A full-screen text page** for one key idea (e.g. "all my money comes from / YouTube AdSense",
  then a line strikes through it).
- **Cards** that pop up over the footage for numbers or lists (see section 5).
- **Text beside his head** for a key phrase, white words on one side of his head and the gold word
  on the other (see section 6).
- Screen shares still stay **raw**: no zooms, no captions, no music on them. At most one or two cards
  over a screen share for the most important number (e.g. "64M views").

## 3. Colours and fonts

- **60 / 30 / 10 colour rule:** near-black background (60%), grey/white text (30%), **gold/orange
  accent** (10%).
  - Background black: `#060607` to `#0B0D12`
  - Grey text: `#968C9C`-ish (light slate)
  - White text: `#F2F2F2`
  - Gold accent (special words): `#FAAC30`
  - Card orange: `#E8681A` fading to black
- **Two fonts, always together:**
  - **Inter Display SemiBold** (clean sans) for normal words, in white.
  - **EB Garamond Bold Italic** (serif) for the **one special word or phrase**, in gold with a soft glow.
  - Example: "Find a **_viral product_**", "last month I made **_$40,000_**".
- Never fill the screen with white captions. White text only appears **next to** a gold word.

## 4. STEP pages (approved for both levels, he likes these)

A short full-screen page when he starts each section:

- Black background with a warm **orange glow coming up from the bottom**, and a thin grid whose lines
  **fade out towards the edges** (never a flat grey grid).
- Small grey "STEP 01" label on top, then the title: white sans words + gold serif words,
  e.g. **"Find a _viral product_"**, **"Validate the _product_"**, **"Build the store _with AI_"**.
- A gold underline draws under the gold words; a huge faint outlined "01" sits behind the title.
- Letters write on one by one (each rises in, goes from blurred to sharp and from yellow to its final
  colour, at a slightly choppy 15 fps).
- About 4 seconds long, his voice keeps playing underneath. A soft zoom-blur transition and a quiet
  whoosh go in and out of it.
- **Timing:** the page appears the moment he says the step. Never leave the page empty for more than
  about a second before the text starts.

## 5. Cards

- Rounded rectangle (corner radius 30 to 70 px), **orange to black gradient** (orange in the top-left
  corner fading to black), a **faint grid inside** the card that fades with the gradient, a **glowing
  orange edge** that is brightest at the top-left (light-sweep look), and a soft outer glow.
- **Few words, big:** a number in gold serif ("$34", "87") or a short phrase in white sans
  ("TV commercials"), plus at most one small grey line underneath ("2 Cloudoras + free shipping").
  **No tiny all-caps labels**, they look cheap.
- Optional faded line icon (play button, dollar sign, TV, "AD") clipped inside the right edge.
- Animation: pops up from smaller and slightly tilted back, rises into place fast and settles slowly;
  several cards appear **one at a time**, in sync with the words.
- Behind the cards the footage is **darkened, blurred and vignetted** so the cards stand out
  (only while the cards are on screen).
- Over the talking head, cards sit **to the left of him** while he keeps talking (he stays visible).
- **Not** on a separate grey grid page. Only over the footage.

## 6. Text beside his head (Level B only)

- Split the phrase: white sans words on the **left** of his head, the gold serif word on the **right**
  ("millions of | _eyes_", "join the | _Discord_").
- Text sits at face height, slightly tilted in 3D, and can tuck a little behind the edge of his head,
  but must **never be hidden behind him**. Keep the camera still while text is on screen.
- The background around him softly darkens and blurs through a **wide feathered ellipse** (it fades
  smoothly into the normal-brightness area around him). Never use a hard cut-out edge.
- Use it rarely: 2 to 4 times per video at most.

## 7. Camera

- Default: **no camera movement**. He said "too much camera movement" several times.
- Allowed: one push-in on the hook number, one slow zoom onto a number before a reveal.
- No zoom when his face first comes on screen. No punch-ins on random lines. No constant drifting.
- Cuts between camera / paper / screen stay **straight cuts** (like the raw). Transitions only into
  and out of STEP pages.

## 8. Music and sound

- **No music during screen shares.** Ever.
- **Level A (raw):** no music at all, only a soft whoosh for STEP pages.
- **Level B:** dark, mature music only, kept low under his voice:
  - "Me and the Devil" instrumental (his own file, not in the repo, may get a YouTube copyright claim)
    for the hook and money moments.
  - "Crypto" by Kevin MacLeod (`assets/music/suspense-underscore/crypto.mp3`) under the paper section.
  - **Never** playful/upbeat tracks ("childish"). Digital Lemonade, Investigations and Chill Wave were rejected.
  - Music always **fades in slowly** (5 to 7 s) and **fades out slowly** (3 to 5 s) across cuts. Never
    pop in or cut off. A short silence right before a big reveal is fine.
- **Sound effects:** quiet and sparse. A soft whoosh for transitions, a soft pop when a card appears,
  ticks for a number counting up. **No "drop"/sub-boom/bass-hit at the start**. He called it
  "too AI-ish".
- Check clip playback inside screen shares for a **double echo** (his mic picks up his speakers about
  50 ms late). If present, remove it (inverse comb filter on just those seconds).
- Loudness: -14 LUFS integrated.

## 9. Export / quality

- **Never downgrade quality.** Export at **2560x1440 (1440p)**, 30 fps, H.264 at high quality
  (CRF 14, preset slow), AAC 320 kbps. A 4K source is downscaled to 1440p with Lanczos.
- Put the finished file in `edited-videos/<name>_edit.mp4` with a `<name>_edit.md` next to it
  (breakdown, what was done, music credits to paste in the description).

## 10. Things that were rejected (do not do these)

- Grey grid pages with cheap grey cards and small all-caps labels.
- Thin orange "lemon yellow" text, or text that looks like a kids' video.
- Lots of white captions, or captions over screen shares.
- Text hidden behind his head; hard, haloed cut-out edges around him.
- Spotlights/zooms all over the tutorial part; framing screen shares as small windows.
- Phone mockups or other gimmicks replacing his real screen share.
- Constant zooms, punch-ins, drifting camera.
- Bass drops/booms at the start, upbeat or playful music, music that pops in abruptly.

---

## How it is built (for whoever runs the code)

The edits are rendered with Python (OpenCV, Pillow, numpy) and ffmpeg, no After Effects:

| File | What it does |
|---|---|
| `tools/edit_youtube_paid_me.py` | The full engine + the Level B edit of "How much YouTube paid me": `TextSeq` (letter-by-letter text), `ui_card()` (cards), `chapter_texts()` / `chapter_numeral()` / `grid_bg()` (STEP pages), `card_dim()` (darken behind cards), feathered-ellipse background in `head_scene()`, `Stroke` (underlines), `pop_in()` (card animation), audio mixing and sound effects in `render_audio()`. Each edit's timing lives in its `scene(...)` list, `TEXTS`, `overlays()` and the music section. |
| `tools/edit_how_to_make_stores.py` | Same engine, set up as the Level A (raw) edit of "How to make stores". The best starting point for a new raw edit: copy it, change `END`, `CUTS`, the `scene(...)` list and `CH` (STEP titles and times). |
| `tools/precompute_paid_me.py` | Person cut-outs (for text beside his head) and paper-camera stabilization. Run it before rendering. |

Workflow for a new video:

1. Download the raw clip (GitHub Release), transcribe it with word timestamps, and look at frames to
   map out the sections (talking head / paper / screen share) and where he says each step.
2. Copy `tools/edit_how_to_make_stores.py`, set the scene list and STEP pages, render test frames
   (`... frames OUTDIR t1 t2 ...`), check them, then render the video in parallel chunks
   (`... video T0 T1 out.mp4`, max 3 at once, they use about 2.5 GB RAM each) and the audio
   (`... audio out.wav`), join the chunks and mux.
3. Commit the `.mp4` (Git LFS) and the `.md` notes to `edited-videos/`.
