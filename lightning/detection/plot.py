
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

def get_rectangle_edges_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height

def draw_pascal_voc_bboxes(plot_ax, bboxes, get_rectangle_corners_fn=get_rectangle_edges_from_pascal_bbox):
    for bbox in bboxes:
        bottom_left, width, height = get_rectangle_corners_fn(bbox)

        rect_1 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=4,
            edgecolor="black",
            fill=False,
        )
        rect_2 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=2,
            edgecolor="white",
            fill=False,
        )

        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        plot_ax.add_patch(rect_2)


def show_image(image: Image, bboxes=None, draw_bboxes_fn=draw_pascal_voc_bboxes, figsize=(10, 10)):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    if bboxes is not None:
        draw_bboxes_fn(ax, bboxes)
    plt.show()


def display_image(image, bboxes, labels, image_id):
    print(f"image_id: {image_id}")
    show_image(image, bboxes.tolist())
    print(labels)
    plt.savefig(f'./fig/{image_id}_box.png')