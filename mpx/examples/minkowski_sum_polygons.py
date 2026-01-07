from shapely.geometry import Polygon
from shapely.ops import triangulate, unary_union
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


def minkowski_sum_polygons(pa, pb):
    """
    % Author: John D'Errico
    % Date: 12/1/2022

    :param pa:
    :param pb:
    :return:
    """
    # Ensure input types are Polygon, convert if needed
    if not isinstance(pa, Polygon):
        pa = Polygon(pa)
    if not isinstance(pb, Polygon):
        pb = Polygon(pb)

    # Check if either polygon is empty or has zero area
    if pa.is_empty or np.isclose(pa.area,  0, atol=1e-6):
        return pb
    if pb.is_empty or np.isclose(pb.area,  0, atol=1e-6):
        return pa


    # Triangulate each polygon
    a_triangles = triangulate(pa)
    b_triangles = triangulate(pb)

    # Minkowski sum of all combinations of triangles
    minkowski_shapes = []
    for a_triangle, b_triangle in product(a_triangles, b_triangles):
        # Get coordinates of the current triangles
        a_coords = np.array(a_triangle.exterior.coords)[:-1]  # Remove the closing point
        b_coords = np.array(b_triangle.exterior.coords)[:-1]

        # Compute Minkowski sum for the triangles by adding each vertex combination
        minkowski_points = [a + b for a, b in product(a_coords, b_coords)]

        # Create convex hull to obtain the final polygon
        minkowski_polygon = Polygon(minkowski_points).convex_hull

        # Append the resulting shape to the list
        minkowski_shapes.append(minkowski_polygon)

    # Union all resulting shapes to form the final Minkowski sum
    minkowski_sum_result = unary_union(minkowski_shapes)

    return minkowski_sum_result


if __name__ == "__main__":
    # Example usage
    pa = Polygon([[-1, 2], [0, 2], [-0.75, 2.25], [-1, 3]])
    pb = Polygon([[1, 0], [2, 0], [3, 1], [4, 3], [2, 0.3]])

    minkowski_sum_result = minkowski_sum_polygons(pa, pb)
    # Plotting the polygons
    fig, ax = plt.subplots()

    # Plot the first polygon (pa)
    x, y = pa.exterior.xy
    ax.plot(x, y, 'b-', label='Polygon PA')

    # Plot the second polygon (pb)
    x, y = pb.exterior.xy
    ax.plot(x, y, 'g-', label='Polygon PB')

    # Plot the resulting Minkowski sum polygon
    x, y = minkowski_sum_result.exterior.xy
    ax.plot(x, y, 'r-', label='Minkowski Sum', linewidth=2)

    # Set plot properties
    ax.set_aspect('equal', 'box')
    ax.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Minkowski Sum of Two Polygons')
    plt.grid(True)
    plt.show()  # Output: POLYGON ((...)) representing the resulting Minkowski sum
