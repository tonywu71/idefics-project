from PIL import Image


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Converts an image to RGB if it is in mode RGBA.
    """
    
    if image.mode == "RGB":
        return image

    # NOTE: `image.convert("RGB")` would only work for .jpg images, as it creates
    # a wrong background for transparent images. The call to `alpha_composite`
    # handles this case.
    
    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    
    return alpha_composite
