# Course photos: CC0 and public-domain examples

These three example photos are bundled locally. They cost nothing to download
or reuse and their file-specific source pages identify them as CC0 or dedicated
to the public domain by their authors. No attribution is required by those rights
statements; credits are recorded here for traceability.

| File | Scene | Author | Rights and source | Example question |
|---|---|---|---|---|
| `photo_01.jpg` | Horses in a meadow | Bethany Legg | [CC0 1.0; source](https://commons.wikimedia.org/wiki/File:Wild_horses_in_a_meadow_(Unsplash).jpg#Licensing) | How many horses are visible? |
| `photo_02.jpg` | A white dog resting outdoors among driftwood | Ravello 99; current version by Soerfm | [Author's public-domain dedication](https://commons.wikimedia.org/wiki/File:BeachDog2.jpg) | Show what you see! |
| `photo_03.jpg` | City buildings with illuminated windows in the mist | Anthony DELANOIX | [CC0 1.0; source](https://commons.wikimedia.org/wiki/File:City_Skyline_Urban_Skyscrapers_(Unsplash).jpg#Licensing) | What is in the foreground, and what is in the background? |

Checked on 2026-11-12. The horse and city images are Wikimedia's 1280-pixel previews.
The dog image is the current Wikimedia file, downloaded without further changes. [sources.json](sources.json) records exact source
revisions, download URLs, rights declarations and SHA-256 checksums. The
Wikimedia footer's CC0 statement about metadata was not used as the photo license.

The notebook `../multimodal_prompting.ipynb` defaults to `photo_02.jpg`
for “Show what you see!”. Change `IMAGE_NAME` to select another photo, then rerun
the selection and encoding cells before asking questions. Only one selected
photo is sent, once per question cell (three API requests per complete run).
The API calls still incur usage; the photos themselves are free.

You can add JPG/JPEG, PNG or WEBP photos; convert HEIC first. Use photos you may
share and send to the API, and remove personal information first. Clear notebook
outputs before sharing.

The original user-supplied photos are preserved locally in `user_supplied/`.
Their reuse rights have not been established, so that folder ignores image files
in Git and they are not part of this publishable example set.
