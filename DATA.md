# Dataset

## Description

About 1,900 images of Indian railway stations and platforms with people present, grouped by how dense the crowd is. The density class is used as a proxy for how severe a crowding situation is.

| Class | Folder | Share of data |
|---|---|---|
| low | `low_new1` | ~46% |
| medium | `medium_new1` | ~30% |
| high | `high_new1` | ~23% |

Total after the split used in all notebooks (stratified, seed 42): 1,332 train / 285 validation / ~285 test. Class indices are inferred from folder names containing `low`, `medium`/`mid` or `high`.

## Expected layout

The notebooks read from `data/data/`:

```
data/
└── data/
    ├── high_new1/     e.g. 1459_2.jpg
    ├── low_new1/      e.g. 97_0.jpg
    └── medium_new1/   e.g. 1089_1.jpg
```

The original notebooks unzip a `data.zip` that holds this structure (`!unzip data.zip -d data`).

## Known issues

- `high_new1/1894_2.png` cannot be opened and is skipped (the notebooks handle this).
- Some images contain on-screen watermarks, which suggests part of the data was captured from video. Do not redistribute the images without checking their license.

## Source and license

_Add the dataset source, how the images were collected, and their license here before publishing. The images are intentionally not committed to this repository (see `.gitignore`)._
