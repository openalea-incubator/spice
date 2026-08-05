import re
from pathlib import Path

from openalea.plantgl.all import (
    Color3,
    Frustum,
    Index3,
    Index3Array,
    Material,
    Scene,
    Shape,
    Tesselator,
    TriangleSet,
    Vector3,
)

from openalea.spice.common.math import cross_vector, denormalize

# This module is read the geometry of each object of fichier RAD


def cylinder_vertices(start, end, rayon):
    """
    Generate the geometry of a cylinder

    Parameters
    ----------
    start: Vec3
        The center of the first circle of cylinder
    end: Vec3
        The center of the second circle of cylinder
    rayon: float
        The radius of these two circles

    Returns
    -------
        A array of vertices of an cylinder

    """

    vert = []
    direction = end - start
    direction.normalize()
    horizontal = cross_vector(direction, Vector3(0, 0, 1))

    v1 = end + rayon * horizontal
    v2 = start + rayon * horizontal
    v3 = start - rayon * horizontal
    v4 = end - rayon * horizontal

    vert.append((v1[0], v1[1], v1[2]))
    vert.append((v2[0], v2[1], v2[2]))
    vert.append((v3[0], v3[1], v3[2]))
    vert.append((v4[0], v4[1], v4[2]))

    return vert


def read_rad(file: Path, scale_factor: float, invert_normals: bool):
    """
    Parse a radiance file (https://radsite.lbl.gov/radiance/framed.html) to a
    make a plantGL Scene

    Parameters
    ----------
    file: Path
        the rad file path
    scale_factor: float
        The size of geometries. The vertices of geometries is recalculated by
        dividing their coordinates by this value
    invert_normals: bool
        whether to invert the normals or not.

    Returns
    -------
        A plantGL Scene.

    """

    with open(file, encoding="UTF8") as f:
        lines = f.readlines()
        materials = {}
        shapes = {}
        sc = Scene()

        i = 0
        for _ in range(len(lines)):
            i += 1
            if i >= len(lines):
                break
            if lines[i].startswith("#"):
                continue
            if lines[i].startswith("void"):  # material
                li = lines[i].split(None)
                type = li[1]
                name = li[2].strip("\n")
                # print("material name: " + str(name))
                if materials.get(name) is None:
                    if type == "plastic":
                        li = lines[i + 4].split(None)
                        color = Color3(
                            denormalize(float(li[0])),
                            denormalize(float(li[1])),
                            denormalize(float(li[2])),
                        )
                        spec = Color3(
                            denormalize(float(li[3])),
                            denormalize(float(li[3])),
                            denormalize(float(li[3])),
                        )
                        roughness = float(li[4])
                        mat = {
                            "name": name,
                            "type": type,
                            "color": color,
                            "spec": spec,
                            "roughness": roughness,
                        }
                        materials[name] = mat
                        i += 5
                    elif type == "metal":
                        li = lines[i + 4].split(None)
                        color = Color3(
                            denormalize(float(li[0])),
                            denormalize(float(li[1])),
                            denormalize(float(li[2])),
                        )
                        spec = Color3(
                            denormalize(float(li[3])),
                            denormalize(float(li[3])),
                            denormalize(float(li[3])),
                        )
                        roughness = float(li[4])
                        mat = {
                            "name": name,
                            "type": type,
                            "color": color,
                            "spec": spec,
                            "roughness": roughness,
                        }
                        materials[name] = mat
                        i += 5
                    elif type == "trans":
                        li = lines[i + 4].split(None)
                        color = Color3(
                            denormalize(float(li[0])),
                            denormalize(float(li[1])),
                            denormalize(float(li[2])),
                        )
                        spec = Color3(
                            denormalize(float(li[3])),
                            denormalize(float(li[3])),
                            denormalize(float(li[3])),
                        )
                        roughness = float(li[4])
                        trans = float(li[5])
                        tspec = float(li[6])
                        mat = {
                            "name": name,
                            "type": type,
                            "color": color,
                            "spec": spec,
                            "roughness": roughness,
                            "trans": trans,
                            "tspec": tspec,
                        }
                        materials[name] = mat
                        i += 5
                    elif type == "light":
                        li = lines[i + 4].split(None)
                        color = Color3(
                            denormalize(float(li[0])),
                            denormalize(float(li[1])),
                            denormalize(float(li[2])),
                        )
                        mat = {"name": name, "type": type, "color": color}
                        materials[name] = mat
                        i += 5
            else:
                keys = materials.keys()
                for k in keys:
                    if lines[i].startswith(k):
                        li = lines[i].split(None)
                        material = li[0]
                        type = li[1]
                        name = li[2].strip("\n")

                        if type in ("point_light", "point"):
                            i += 1
                            vert = []
                            line = re.split(r"\s+|;+", lines[i])
                            vert.append(
                                (
                                    float(line[0]) / scale_factor,
                                    float(line[1]) / scale_factor,
                                    float(line[2]) / scale_factor,
                                )
                            )
                            shapes[name] = {
                                "vertices": vert,
                                "type": type,
                                "size": 1,
                                "material": material,
                            }
                        else:
                            i += 3
                            vert = []
                            size = int(lines[i])
                            tmp = i + 1
                            nbCoords = int(size / 3)
                            if type == "cylinder":
                                l1 = re.split(r"\s+|;+", lines[i + 1])
                                l2 = re.split(r"\s+|;+", lines[i + 2])
                                l3 = re.split(r"\s+|;+", lines[i + 3])
                                r = float(l3[0]) / scale_factor

                                x, y, z = (
                                    float(l1[0]) / scale_factor,
                                    float(l1[1]) / scale_factor,
                                    float(l1[2]) / scale_factor,
                                )
                                x2, y2, z2 = (
                                    float(l2[0]) / scale_factor,
                                    float(l2[1]) / scale_factor,
                                    float(l2[2]) / scale_factor,
                                )

                                vert = cylinder_vertices(
                                    Vector3(x, y, z), Vector3(x2, y2, z2), r
                                )

                                shapes[name] = {
                                    "vertices": vert,
                                    "type": type,
                                    "size": len(vert),
                                    "material": material,
                                }
                                i += 3
                            elif type == "cone":
                                l1 = re.split(r"\s+|;+", lines[i + 1])
                                l2 = re.split(r"\s+|;+", lines[i + 2])
                                l3 = re.split(r"\s+|;+", lines[i + 3])
                                r = float(l3[0]) / scale_factor
                                r2 = float(l3[1]) / scale_factor

                                x, y, z = (
                                    float(l1[0]) / scale_factor,
                                    float(l1[1]) / scale_factor,
                                    float(l1[2]) / scale_factor,
                                )
                                x2, y2, z2 = (
                                    float(l2[0]) / scale_factor,
                                    float(l2[1]) / scale_factor,
                                    float(l2[2]) / scale_factor,
                                )

                                ratio = r / r2

                                pos = [x, y, z]
                                newCone = Frustum()
                                newCone.taper = 1.0 + ratio
                                newCone.height = z2 * 10 - z * 10
                                ts = Tesselator()
                                newCone.apply(ts)
                                mesh = ts.result
                                maxi = 0
                                for i in range(0, mesh.indexListSize()):
                                    index = mesh.indexAt(i)
                                    typeF = mesh.faceSize(i)
                                    for j in range(0, typeF):
                                        if index[j] > maxi:
                                            maxi = index[j]
                                for u in range(0, maxi + 1):
                                    mvector = mesh.pointAt(u)
                                    mesh.pointList[u] = (
                                        mvector[0] / 10 + pos[0],
                                        mvector[1] / 10 + pos[1],
                                        mvector[2] / 10 + pos[2],
                                    )

                                shapes[name] = {
                                    "vertices": vert,
                                    "type": type,
                                    "size": len(vert),
                                    "material": material,
                                    "mesh": mesh,
                                }
                                i += 3
                            else:
                                for j in range(tmp, tmp + nbCoords):
                                    line = re.split(r"\s+|;+", lines[j])
                                    vert.append(
                                        (
                                            float(line[0]) / scale_factor,
                                            float(line[1]) / scale_factor,
                                            float(line[2]) / scale_factor,
                                        )
                                    )
                                    i = j
                                    shapes[name] = {
                                        "vertices": vert,
                                        "type": type,
                                        "size": size,
                                        "material": material,
                                    }

        for sh, val in shapes.items():
            mat_key = val["material"]
            mat = materials[mat_key]
            vert = val["vertices"]
            s = Shape()
            nbCoords = int(val["size"] / 3)
            if val.get("mesh") is not None:
                print("mesh detected")
                s.name = sh
                s.geometry = val["mesh"]
                if mat["type"] == "light":
                    s.name += str(vert[0])
                    s.appearance = Material(
                        name=name,
                        ambient=Color3(mat["color"]),
                        emission=Color3(mat["color"]),
                    )

                elif mat["type"] == "trans":
                    s.appearance = Material(
                        name=mat["name"],
                        ambient=Color3(mat["color"]),
                        specular=Color3(mat["spec"]),
                        shininess=1 - mat["roughness"],
                        transparency=mat["trans"],
                    )
                else:
                    if Color3(mat["spec"]) == Color3(0, 0, 0):
                        s.appearance = Material(
                            name=mat["name"],
                            ambient=Color3(mat["color"]),
                            shininess=1 - mat["roughness"],
                        )
                    else:
                        s.appearance = Material(
                            name=mat["name"],
                            ambient=Color3(mat["color"]),
                            specular=Color3(mat["spec"]),
                            shininess=1 - mat["roughness"],
                        )
                s.appearance.name = mat["name"]
                sc.add(s)

            elif nbCoords % 3 == 0:
                ts = TriangleSet(vert)
                i = 0
                indList = []
                if nbCoords == 9:
                    ind = Index3(i, i + 1, i + 2)
                    indList.append(ind)
                    ind = Index3(i, i + 2, i + 3)
                    indList.append(ind)
                    ind = Index3(i, i + 3, i + 4)
                    indList.append(ind)
                    ind = Index3(i, i + 4, i + 5)
                    indList.append(ind)
                    ind = Index3(i, i + 5, i + 6)
                    indList.append(ind)
                    ind = Index3(i, i + 6, i + 7)
                    indList.append(ind)
                    ind = Index3(i, i + 7, i + 8)
                    indList.append(ind)
                else:
                    while i < nbCoords:
                        if invert_normals:
                            ind = Index3(i, i + 2, i + 1)
                        else:
                            ind = Index3(i, i + 1, i + 2)
                        indList.append(ind)
                        i += 3
                ts.indexList = Index3Array(indList)
                ts.computeNormalList()
                s.geometry = ts
                s.name = sh
                if mat["type"] == "light":
                    s.name += str(vert[0])
                    s.appearance = Material(
                        name=name,
                        ambient=Color3(mat["color"]),
                        emission=Color3(mat["color"]),
                    )

                elif mat["type"] == "trans":
                    s.appearance = Material(
                        name=mat["name"],
                        ambient=Color3(mat["color"]),
                        specular=Color3(mat["spec"]),
                        shininess=1 - mat["roughness"],
                        transparency=mat["trans"],
                    )
                else:
                    if Color3(mat["spec"]) == Color3(0, 0, 0):
                        s.appearance = Material(
                            name=mat["name"],
                            ambient=Color3(mat["color"]),
                            shininess=1 - mat["roughness"],
                        )
                    else:
                        s.appearance = Material(
                            name=mat["name"],
                            ambient=Color3(mat["color"]),
                            specular=Color3(mat["spec"]),
                            shininess=1 - mat["roughness"],
                        )
                s.appearance.name = mat["name"]
                sc.add(s)
            else:
                # triangulate quad
                ts = TriangleSet(vert)
                i = 0
                indList = []
                while i < nbCoords:
                    if invert_normals:
                        ind = Index3(i, i + 2, i + 1)
                        ind2 = Index3(i, i + 3, i + 2)
                    else:
                        ind = Index3(i, i + 1, i + 2)
                        ind2 = Index3(i, i + 2, i + 3)
                    indList.append(ind)
                    indList.append(ind2)
                    i += 4

                ts.indexList = Index3Array(indList)
                ts.computeNormalList()
                s.geometry = ts
                s.name = sh
                if mat["type"] == "light":
                    s.appearance = Material(
                        name=mat["name"],
                        ambient=Color3(mat["color"]),
                        emission=Color3(mat["color"]),
                    )
                elif mat["type"] == "trans":
                    s.appearance = Material(
                        name=mat["name"],
                        ambient=Color3(mat["color"]),
                        specular=Color3(mat["spec"]),
                        shininess=1 - mat["roughness"],
                        transparency=mat["trans"],
                    )
                else:
                    if Color3(mat["spec"]) == Color3(0, 0, 0):
                        s.appearance = Material(
                            name=mat["name"],
                            ambient=Color3(mat["color"]),
                            shininess=1 - mat["roughness"],
                        )
                    else:
                        s.appearance = Material(
                            name=mat["name"],
                            ambient=Color3(mat["color"]),
                            specular=Color3(mat["spec"]),
                            shininess=1 - mat["roughness"],
                        )
                s.appearance.name = mat["name"]
                sc.add(s)
        return sc
