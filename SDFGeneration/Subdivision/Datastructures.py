import open3d as o3d
import numpy as np
from typing import Callable
import yaml

class OctreeNode:
    def __init__(
                self,
                origin: np.array,
                extents: np.array,
                points: np.array,
                path: str,
                ):
        self.origin = origin
        self.extents = extents
        self.points = points
        self.path = path
        self.children = [None for _ in range(8)]
        self.is_leaf = False

    def is_inside(self, points: np.array):
        return np.all(self.origin <= points) & np.all(points <= self.origin + self.extents)
    
    def occupancy(self, points: np.array):
        return np.all(self.origin <= points, axis=-1) & np.all(points <= self.origin + self.extents, axis=-1)

    def get_child_index(self, points: np.array):
        index = np.zeros(points.shape[0]).astype('int8')
        mask = np.array(points <= (self.origin + self.extents / 2.)).astype('int8')

        for i in range(mask.shape[1]):
            index = 2 * index + mask[:,i]
        index[~self.is_inside(points)] = -1

        return index

    def get_child_bounding_box(self, index: int):
        child_extents = self.extents / 2.
        dim = self.points.shape[1]
        child_origin = np.array([self.origin[i] + (0 if ((index >> (dim-i-1)) & 1) == 1 else child_extents[i]) for i in range(dim)])
        return child_origin, child_extents


    def get_populated_child_count(self):
        return sum(1 if x is not None else 0 for x in self.children)

    def get_debug_string(self):
        return f"OctreeNode with {self.points.shape[0]} points and {self.get_populated_child_count()} children."

    def get_debug_dict(self):
        return {
            "index": int(self.path[-1]) if len(self.path) > 0 else "root",
            "points_count": self.points.shape[0],
            "child_count": self.get_populated_child_count(),
            "leaf": self.is_leaf,
            "path": self.path,
            "depth": len(self.path),
            "origin": {i: float(j) for i, j in zip("xyz", self.origin)},
            "extents": {i: float(j) for i, j in zip("xyz", self.extents)},
        }

class Octree:
    def __init__(
                self,
                origin: np.array,
                extents: np.array,
                points: np.array,
                max_depth: int,
                subdivide: Callable[[np.array, int], bool],
                drop: Callable[[np.array, int], bool]
                ):

        self.origin = origin
        self.extents = extents
        self.points = points
        self.max_depth = max_depth
        self.subdivide = subdivide
        self.drop = drop
        self.root_node = OctreeNode(origin=self.origin,
                                    extents=self.extents,
                                    points=self.points,
                                    path="")
        self.build_from_points(self.root_node, self.points)

    def build_from_points(self, current_node: OctreeNode, points: np.array, depth: int = 0):
        assert current_node.is_inside(points)
        if depth == self.max_depth:
            current_node.is_leaf = True
            return
        
        if self.subdivide is not None and not self.subdivide(points, depth):
            current_node.is_leaf = True
            return

        indices = current_node.get_child_index(points)
        if np.unique(indices[indices >= 0]).shape[0] < 2:
            current_node.is_leaf = True
            return

        for i in np.unique(indices):
            if i == -1:
                continue
            if self.drop is not None and self.drop(points[indices == i], depth+1):
                continue
            
            child_origin, child_extents = current_node.get_child_bounding_box(i)
            current_node.children[i] = OctreeNode(child_origin, child_extents, points[indices==i], current_node.path + str(i))
            self.build_from_points(current_node.children[i], current_node.children[i].points, depth+1)
    

    def get_all_leaf_nodes(self, current_node: OctreeNode = None):
        if current_node is None:
            current_node = self.root_node

        if current_node.is_leaf:
            return [current_node]
        
        result = []
        for child_node in current_node.children:
            if child_node is None:
                continue
            result += self.get_all_leaf_nodes(child_node)
        return result

    def visualize(self, show_boxes=True, show_coordinate_frames=False, show_points=False, points_as_spheres=False, additional_geometries: list = []):
        leaf_nodes = self.get_all_leaf_nodes()
        geometries = []
        for node in leaf_nodes:
            if show_boxes:
                box = o3d.geometry.TriangleMesh.create_box(width = node.extents[0],
                                                        height = node.extents[1],
                                                        depth = node.extents[2])
                box.translate(node.origin)
                box.paint_uniform_color(np.random.random(3))
                geometries.append(box)
            if show_points:
                max_balls_per_node = self.root_node.points.shape[0] * 0.001 / len(leaf_nodes)
                if not points_as_spheres or node.points.shape[1] > max_balls_per_node:
                    geometries.append(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(node.points)))
                else:
                    radius = max(0.01, node.extents[0]**3 / node.points.shape[0]) / 2.
                    balls = []
                    for i in range(node.points.shape[0]):
                        ball = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                        ball.paint_uniform_color(np.array([0, 0, 0]))
                        ball.translate(node.points[i])
                        balls.append(ball)
                    geometries += balls
            
            if show_coordinate_frames:
                geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=node.extents[0], origin=node.origin))

        geometries += additional_geometries
        o3d.visualization.draw_geometries(geometries)
    
    def get_debug_string(self, current_node: OctreeNode = None, depth=0):
        if current_node is None:
            current_node = self.root_node

        index = "root" if current_node == self.root_node else current_node.path[-1]
        result = f"{' '*depth*4} [{index}] {current_node.get_debug_string()} {np.unique(current_node.get_child_index(current_node.points))} Leaf: {current_node.is_leaf}\n"
        for child_node in current_node.children:
            if child_node is not None:
                result += self.get_debug_string(child_node, depth+1)
        return result

    def check_integrity(self, current_node: OctreeNode = None):
        if current_node is None:
            current_node = self.root_node
        
        result = True
        for child_node in current_node.children:
            if child_node is None:
                continue
            result = result and (not (current_node.is_leaf and child_node.is_leaf))
            self.check_integrity(child_node)
        return result

    def get_debug_dict(self, current_node: OctreeNode = None):
        if current_node is None:
            current_node = self.root_node
        
        result = current_node.get_debug_dict()
        if current_node.get_populated_child_count() > 0:
            result["children"] = {}

        for child_node in current_node.children:
            if child_node is None:
                continue
            index = int(child_node.path[-1])
            result["children"][index] = self.get_debug_dict(child_node)
        
        return result
