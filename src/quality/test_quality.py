from pathlib import Path

from quality.image_quality import analyze_image_quality


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_IMAGES = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "B. Disease Grading"
    / "1. Original Images"
    / "a. Training Set"
)


# Find first 10 retinal images
images = []

for path in TRAIN_IMAGES.rglob("*"):

    if path.suffix.lower() in [
        ".jpg",
        ".jpeg",
        ".png"
    ]:
        images.append(path)


images = images[:10]


print("=" * 70)
print("FUNDUS IMAGE QUALITY ANALYSIS")
print("=" * 70)


for image_path in images:

    scores = analyze_image_quality(image_path)

    print("\nImage:", image_path.name)

    print(
        f"  Focus:      {scores['focus']:.2f}"
    )

    print(
        f"  Brightness: {scores['brightness']:.2f}"
    )

    print(
        f"  Contrast:   {scores['contrast']:.2f}"
    )

    print(
        f"  FOV ratio:  {scores['fov']:.3f}"
    )