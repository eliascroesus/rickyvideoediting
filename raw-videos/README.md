# Raw videos

Drop your **unedited footage** in this folder. Claude picks it up from here,
edits it, and puts the finished version in [`../edited-videos/`](../edited-videos/).

Any common video format works: `.mp4`, `.mov` (iPhone), `.mkv`, `.avi`, `.webm`, `.mts`, etc.
You can also add separate audio (`.wav`, `.m4a`, `.mp3`) if you recorded sound on another device.

---

## How to upload

### Option A: Small clips (under 25 MB each) from the GitHub website

1. Open this `raw-videos` folder on github.com.
2. Click **Add file → Upload files**.
3. Drag the clips in, then click **Commit changes**.

GitHub's website won't accept files over **25 MB** here. Most phone videos are bigger than
that, so for those use option B or C.

### Option B: Big files (up to 2 GB each) in the browser, via a Release (easiest)

Releases accept browser uploads up to **2 GB per file**, with no app to install. They also don't
use your Git LFS storage quota.

1. **First time:** on a computer, open
   https://github.com/eliascroesus/rickyvideoediting/releases/new?tag=raw-videos&title=Raw%20videos&prerelease=1
   (the tag `raw-videos` and the title are pre-filled).
2. Drag your videos into the **"Attach binaries by dropping them here"** box at the bottom and
   wait until every upload bar finishes.
3. Click **Publish release**.
4. **Next time**, add more videos to the same release with
   https://github.com/eliascroesus/rickyvideoediting/releases/edit/raw-videos
   → drop files → **Update release**.

Then tell Claude the file names. Claude downloads them from the release and edits them.

> ⚠️ This repository is **public**, so anyone can see and download release files (and anything
> in this folder). For private footage, first make the repo private: **Settings → General →
> Danger Zone → Change visibility**. In that case, use option C, which Claude can
> always download from.

### Option C: Any size up to 2 GB per file, with GitHub Desktop

1. Install **GitHub Desktop** (free, Mac and Windows): https://desktop.github.com and sign in.
2. **File → Clone repository** → pick `eliascroesus/rickyvideoediting`.
3. **Repository → Show in Finder / Show in Explorer**, then copy your videos into the `raw-videos` folder.
4. Back in GitHub Desktop, type a short summary (e.g. `Add gym clips`) → **Commit**.
5. Click **Push origin**.

Video files go through **Git LFS** (large file storage) automatically. This repo's
`.gitattributes` file is already set up for it, so there's nothing to configure.

### Option D: Command line

```bash
git lfs install
```

```bash
cp ~/Movies/my-clip.mov raw-videos/
```

```bash
git add raw-videos && git commit -m "Add my-clip" && git push
```

---

## Tips so the edit comes out the way you want

- **Name clips clearly**, e.g. `2026-11-12_gym-pr-attempt.mov` (no need to rename
  `IMG_1234.MOV` files if the note below explains them).
- **Leave a note**: add a text file with the same name (e.g. `2026-11-12_gym-pr-attempt.md`)
  or just tell Claude in chat what you want: the vibe, length (e.g. 30-second Reel / TikTok
  9:16 or YouTube 16:9), the best moments, captions, music mood.
- **Group a multi-clip project** in a subfolder, e.g. `raw-videos/car-meet-nov/`.

## Storage limits (free GitHub plan)

- Git LFS includes **10 GB of storage** and **10 GB of downloads per month**.
- Raw phone footage fills that fast (about 1 minute of 4K ≈ 350–400 MB). Once a video
  is edited and you're happy with it, you can delete the raw clip to free up space.
  (Deleting from the latest version doesn't instantly reclaim LFS storage. Ask Claude
  if you start running low.)
