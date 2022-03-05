from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class image():
    def __init__(self, areas, image_name):
        self.image_name = image_name
        self.areas = areas

    def resize_image(self, size):
        self.image = Image.open(self.image_name)

        # Resize to size by size.
        self.image = self.image.resize((size, size), resample=Image.BILINEAR)

    def show_areas(self):
        plt.figure(figsize=(12, 6), dpi=200)
        plt.imshow(self.image)

        # Plot rectangles in red.
        for x_b, x_e, y_b, y_e in self.areas:
            plt.hlines(y_b, x_b, x_e, colors='r')
            plt.hlines(y_e, x_b, x_e, colors='r')
            plt.vlines(x_b, y_b, y_e, colors='r')
            plt.vlines(x_e, y_b, y_e, colors='r')
        plt.show()

    def average_within_area(self, array, area):
        x_b, x_e, y_b, y_e = area

        total = sum(sum(array[y][x_b + 1 : x_e]) for y in range(y_b + 1, y_e))
        length = (x_e - (x_b + 1)) * (y_e - (y_b + 1))
        return total / length

    def RSS(self, x1, x2):
        return sum((x1 - x2) ** 2) / len(x2)

    def classify_pixels(self):
        array = np.array(self.image, dtype=int)
        self.classified_pixels = np.zeros((len(array), len(array[0])))
        avg_per_areas = [self.average_within_area(array, area) for area in self.areas]
        for y in range(len(array)):
            for x in range(len(array[0])):
                best = np.inf
                best_state = None
                for cur_state, avg in enumerate(avg_per_areas):
                    if self.RSS(array[y][x], avg) < best:
                        best_state = cur_state
                        best = self.RSS(array[y][x], avg)
                self.classified_pixels[y][x] = best_state

    def show_classified_pixels(self):
        import matplotlib

        plt.figure(figsize=(12, 6), dpi=200)

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(self.classified_pixels, interpolation='none', vmin=0, vmax=len(self.areas) - 1,
                cmap=matplotlib.cm.binary)
        plt.axis('image')
        plt.show()


if __name__ == '__main__':
    sand = (23, 31, 83, 89)
    gras = (130, 160, 130, 160)
    woods = (75, 120, 110, 155)

    size = 200

    areas = np.array([sand, gras, woods])
    areas = np.array(list(list(map(int, area * size / 200)) for area in areas))

    img = image(areas, 'place.png')

    img.resize_image(size)

    img.classify_pixels()
    img.show_areas()
    img.show_classified_pixels()
