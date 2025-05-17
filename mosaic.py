import os

import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def create_prediction_mosaic(
    image_dir,
    csv_path,
    output_path,
    mosaic_size=(2000, 2000),
    tile_size=(200, 200),
    font_size=16,
    max_images=100,
):
    """
    Create a visual mosaic of predictions sorted by confidence.

    Args:
        image_dir: Directory containing input images
        csv_path: Path to predictions CSV file
        output_path: Where to save the mosaic image
        mosaic_size: Dimensions of output mosaic (width, height)
        tile_size: Size of each image tile (width, height)
        font_size: Font size for labels
        max_images: Maximum number of images to include
    """
    # Load predictions and sort by confidence
    df = pd.read_csv(csv_path)
    df = df.sort_values("max_probability", ascending=False)

    # Limit number of images
    if len(df) > max_images:
        df = df.iloc[:max_images]

    # Calculate mosaic grid dimensions
    cols = mosaic_size[0] // tile_size[0]
    rows = mosaic_size[1] // tile_size[1]

    # Create blank mosaic canvas
    mosaic = Image.new("RGB", mosaic_size, color=(240, 240, 240))
    draw = ImageDraw.Draw(mosaic)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Process each image
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx >= cols * rows:
            break

        # Calculate grid position
        x = (idx % cols) * tile_size[0]
        y = (idx // cols) * tile_size[1]

        try:
            # Load and resize image
            img_path = os.path.join(image_dir, row["image_name"])
            img = Image.open(img_path).convert("RGB")
            img = img.resize(tile_size)

            # Draw border color based on correctness
            true_class = row.get("true_class", None)
            pred_class = row["predicted_class"]
            border_color = (0, 255, 0) if true_class == pred_class else (255, 0, 0)

            # Add border
            bordered_img = Image.new(
                "RGB", (tile_size[0] + 4, tile_size[1] + 4), border_color
            )
            bordered_img.paste(img, (2, 2))

            # Add to mosaic
            mosaic.paste(bordered_img, (x, y))

            # Add text labels
            text = f"{pred_class}\n{row['max_probability']:.2f}"
            text_y = y + tile_size[1] + 5

            # Draw text with background for readability
            text_size = draw.textsize(text, font=font)
            draw.rectangle(
                [x, text_y, x + text_size[0], text_y + text_size[1]],
                fill=(255, 255, 255),
            )
            draw.text((x, text_y), text, font=font, fill=(0, 0, 0))

        except Exception as e:
            print(f"Error processing {row['image_name']}: {str(e)}")
            continue

    # Add header with summary statistics
    header = (
        f"Prediction Mosaic (Sorted by Confidence) | "
        f"Total: {len(df)} | "
        f"Avg Confidence: {df['max_probability'].mean():.2f}"
    )
    header_size = draw.textsize(header, font=font)
    draw.rectangle([0, 0, mosaic_size[0], header_size[1] + 10], fill=(200, 200, 255))
    draw.text((10, 5), header, font=font, fill=(0, 0, 0))

    # Save final mosaic
    mosaic.save(output_path)
    print(f"Mosaic saved to {output_path}")

    # Create and save confidence histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df["max_probability"], bins=20, color="skyblue", edgecolor="black")
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Confidence Score")
    plt.ylabel("Number of Images")
    hist_path = os.path.splitext(output_path)[0] + "_histogram.png"
    plt.savefig(hist_path)
    plt.close()
    print(f"Confidence histogram saved to {hist_path}")
