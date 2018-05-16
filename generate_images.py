import cairo
import random
import itertools
import math
import argparse
import os
import sys
import shutil
import enum

AREA_LOWER_THRESHOLD = 0.10
AREA_UPPER_THRESHOLD = 0.30

class GeometricShape(enum.Enum):
    TRIANGLE = 0, 'triangle'
    RECTANGLE = 1, 'rectangle'
    CIRCLE = 2, 'circle'

    @staticmethod
    def name(value):
        if value == GeometricShape.TRIANGLE: return 'triangle'
        if value == GeometricShape.RECTANGLE: return 'rectangle'
        if value == GeometricShape.CIRCLE: return 'circle'


def draw_triangle(context):
    while True:
        x1, y1, x2, y2, x3, y3 = [random.random() for x in range(6)]
        area = 0.5 * abs((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))
        if area > AREA_LOWER_THRESHOLD and area < AREA_UPPER_THRESHOLD: break
    context.move_to(x1, y1)
    context.line_to(x2, y2)
    context.line_to(x3, y3)
    context.line_to(x1, y1)
    context.fill()


def draw_rectangle(context):
    while True:
        x1, y1, x2, y2 = [random.random() for x in range(4)]
        area = abs(x1 - x2) * abs(y1 - y2)
        if area > AREA_LOWER_THRESHOLD and area < AREA_UPPER_THRESHOLD: break
    context.move_to(x1, y1)
    context.line_to(x1, y2)
    context.line_to(x2, y2)
    context.line_to(x2, y1)
    context.line_to(x1, y1)
    context.fill()


def draw_circle(context):
    while True:
        x, y, r = [random.random() for x in range(3)]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1: continue
        area = r * r * math.pi
        if area > AREA_LOWER_THRESHOLD and area < AREA_UPPER_THRESHOLD: break
    context.arc(x, y, r, 0, 2 * math.pi)
    context.fill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count',
                        help='Number of images to generate', default=1000)
    parser.add_argument('--size',
                        help='Image dimension (width and height)', default=64)
    parser.add_argument('--output',
                        help='Output folder (will be created if doesn\'t '
                        'exist)', default='output')
    parser.add_argument('--force',
                        help='Delete content of output folder if not empty',
                        action='store_true')
    parser.add_argument('--multiple_colors',
                        help='Generate shapes of different colors',
                        action='store_true')
    parser.add_argument('--multiple_shapes',
                        help='Generate multiple shapes one single image',
                        action='store_true')
    parser.add_argument('--seed', help='Seed used by random')
    parsed = parser.parse_args()

    if parsed.seed:
        random.seed(parsed.seed)
    if not os.path.exists(parsed.output):
        os.mkdir(parsed.output)
    if not os.path.isdir(parsed.output):
        print('--output', parsed.output, 'is not directory', file=sys.stderr)
        sys.exit(1)
    elif os.listdir(parsed.output):
        if parsed.force:
            shutil.rmtree(parsed.output)
            os.mkdir(parsed.output)
        else:
            print('--output', parsed.output, 'is not empty', file=sys.stderr)
            sys.exit(1)

    num_triangles, num_rectangles, num_circles = 0, 0, 0
    for i in range(parsed.count):
        imagesize = (parsed.size, parsed.size)
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, *imagesize)
        context = cairo.Context(surface)
        context.scale(*imagesize)

        context.set_source_rgb(1.0, 1.0, 1.0)
        context.paint()

        num_shapes = random.randint(1, 4) if parsed.multiple_shapes else 1
        shapes = []
        for _ in range(num_shapes):
            COLOR_CHOICES = (("red", (1.0, 0.0, 0.0)),
                             ("green", (0.0, 1.0, 0.0)),
                             ("blue", (0.0, 0.0, 1.0)))
            if parsed.multiple_colors:
                color = random.choice(COLOR_CHOICES)
            else:
                color = COLOR_CHOICES[0]
            color_name, color_vector = color
            context.set_source_rgb(*color_vector)
            geometric_shape = random.choice(list(GeometricShape))
            if geometric_shape == GeometricShape.TRIANGLE:
                draw_triangle(context)
                num_triangles += 1
            if geometric_shape == GeometricShape.RECTANGLE:
                draw_rectangle(context)
                num_rectangles += 1
            if geometric_shape == GeometricShape.CIRCLE:
                draw_circle(context)
                num_circles += 1
            context.fill()
            shapes.append("{} {}".format(color_name,
                    GeometricShape.name(geometric_shape)))

        surface.write_to_png(os.path.join(
                parsed.output, "pic{:05d}.png".format(i)))
        with open(os.path.join(parsed.output, "shape{:05d}".format(i)), 'w') \
                as shape_file:
            print('\t'.join(shapes), file=shape_file)

    print("Generated {} triangles, {} rectangles, {} circles".format(
          num_triangles, num_rectangles, num_circles), file=sys.stderr)


if __name__ == "__main__":
    main()
