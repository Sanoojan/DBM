import Box2D as b2d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon


class MatplotlibDebugDrawer(b2d.b2Draw):
    def __init__(self, ax):
        super(MatplotlibDebugDrawer, self).__init__()
        self.ax = ax
        self.patches = []

    def clear_patches(self):
        for patch in self.patches:
            patch.remove()
        self.patches = []

    def DrawPolygon(self, vertices, color):
        polygon = Polygon(vertices, closed=True, fill=False, edgecolor=(color.r, color.g, color.b), linewidth=1)
        self.ax.add_patch(polygon)
        self.patches.append(polygon)

    def DrawSolidPolygon(self, vertices, color):
        # Use a fixed alpha or pass it in as a parameter if needed
        polygon = Polygon(
            vertices,
            closed=True,
            facecolor=(color.r, color.g, color.b, 0.5),
            edgecolor=(color.r, color.g, color.b),
            linewidth=1,
        )
        self.ax.add_patch(polygon)
        self.patches.append(polygon)

    def DrawCircle(self, center, radius, color):
        circle = Circle(center, radius * self.ppm, fill=False, edgecolor=(color.r, color.g, color.b), linewidth=1)
        self.ax.add_patch(circle)
        self.patches.append(circle)

    def DrawSolidCircle(self, center, radius, axis, color):
        circle = Circle(
            center,
            radius * self.ppm,
            facecolor=(color.r, color.g, color.b, 0.5),
            edgecolor=(color.r, color.g, color.b),
            linewidth=1,
        )
        self.ax.add_patch(circle)
        self.patches.append(circle)

    def DrawSegment(self, p1, p2, color):
        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=(color.r, color.g, color.b), linewidth=1)

    def DrawPoint(self, p, size, color):
        self.ax.plot(p[0], p[1], "o", markersize=size, color=(color.r, color.g, color.b))

    def DrawTransform(self, xf):
        raise NotImplementedError


def save_box2d_debug(world, out_path, hit):
    # Set up Matplotlib figure and axis
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect("equal")
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Box2D Simulation")

    # Instantiate the debug drawer and set it for the world
    debug_drawer = MatplotlibDebugDrawer(ax)
    world.renderer = debug_drawer
    debug_drawer.flags = {"drawShapes": True}

    # Draw the world
    world.DrawDebugData()

    # Draw ego
    if hit:
        ax.plot(0, 0, "o", markersize=5.0, color="red")
    else:
        ax.plot(0, 0, "o", markersize=5.0, color="black")

    plt.savefig(out_path)
    plt.close()
