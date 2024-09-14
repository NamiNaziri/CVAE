from collections import deque
from xml.etree import ElementTree as ET

import numpy as np
import pyvista as pv


class XMLVisualDataContainer:

    def __init__(self, xml_path: str):
        tree = ET.parse(xml_path)
        worldbody_tag = tree.find("worldbody")
        self.geoms = []  # sorted in the DFS encounter order
        self.offsets = []

        # A body has multiple geom children optionally.
        body_stack = deque()
        for body_tag in worldbody_tag.findall("body")[::-1]:
            body_stack.append((None, body_tag, np.zeros(3)))

        while body_stack:
            parent, child, global_parent_offset = body_stack.pop()
            local_child_offset = np.array(child.attrib.get("pos", "0 0 0").split(), dtype=float)
            global_child_offset = global_parent_offset + local_child_offset

            for geom in child.findall("geom")[::-1]:
                self.geoms.append(geom)
                geom_pos = np.array(geom.attrib.get("pos", "0 0 0").split(), dtype=float)
                self.offsets.append(local_child_offset)

            for grandchild in child.findall("body")[::-1]:
                body_stack.append((child, grandchild, global_child_offset))

        # self.plane = pv.Cube(center=(0, 0, -0.05), x_length=5, y_length=5, z_length=0.1)
        self.plane = pv.Plane(center=(0, 0, 0), i_size=10, j_size=10)

        # Following the DFS encounter order, create PyVista meshes
        self.meshes = []
        self.axes = []
        self.centers = []
        for geom_tag, offset in zip(self.geoms, self.offsets):
            geom_pos = np.array(geom_tag.attrib.get("pos", "0 0 0").split(), dtype=float)

            if geom_tag.attrib["type"] == "sphere":
                size = float(geom_tag.attrib["size"])
                center = np.zeros(3)
                mesh = pv.Sphere(radius=size, center=geom_pos)
                self.centers.append(center)

            elif geom_tag.attrib["type"] == "box":
                x_length, y_length, z_length = [float(k) * 2 for k in geom_tag.attrib["size"].split()]
                center = np.zeros(3)
                mesh = pv.Cube(center=geom_pos, x_length=x_length, y_length=y_length, z_length=z_length)
                self.centers.append(center)

            elif geom_tag.attrib["type"] == "capsule":
                size = float(geom_tag.attrib["size"])
                fromto = np.array([float(k) for k in geom_tag.attrib["fromto"].split()])
                fro = fromto[:3]
                to = fromto[3:]

                direction = to - fro
                length = np.linalg.norm(direction)
                direction /= length
                center = direction * 0.5 * length
                mesh = pv.Arrow(start=fro, direction=direction, scale=length, shaft_radius=size)

                self.centers.append(center)
            else:
                raise NotImplementedError
            self.meshes.append(mesh)
            ax = pv.Axes()
            self.axes.append(ax)

        self.targets = []
        for _ in range(24):
            sphere = pv.Sphere(radius=0.05)
            self.targets.append(sphere)
