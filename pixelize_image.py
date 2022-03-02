from PIL import Image

def pixelate_image(img_name):
    img = Image.open("place.png")

    # Resize to 100 by 80.
    imgSmall = img.resize((200,160),resample=Image.BILINEAR)

    # Scale back up using NEAREST to original size.
    imgBig = imgSmall.resize(img.size, Image.NEAREST)

    # # Save images.
    imgSmall.save('place_decreased.png')
    imgBig.save('place_increased.png')

