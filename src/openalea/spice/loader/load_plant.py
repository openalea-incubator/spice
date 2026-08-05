try:
    from openalea.lpy import Lsystem
except ImportError:
    pass

from openalea.spice import (
    Vec3,
    VectorFloat,
    VectorUint,
)
from openalea.spice.common.tools import flatten
from openalea.plantgl.all import (
    Color3,
    Material,
    Scene,
    Shape,
    Tesselator,
    TriangleSet,
    Vector3,
)

# Objective of this module is adding plants to the scene of Photon Mapping to calculate the received energy
# Data is located in this directory: ./assets


def add_lpy_file_to_scene(scene, filename, t, tr2shmap, anchor, scale_factor=1):
    """
    Adds the lpy mesh to the photonmapping scene.

    Parameters
    ----------
    scene : spice.Scene
        The photon mapping scene used to run the simulation
    filename : str
        The link to the lpy file
    t : int
        The number of iteration applied
    tr2shmap : dict
        The dictionary of triangles of plant
    anchor : Vec3
        The position of the plant
    scale_factor : int
        The size of geometries. The vertices of geometries is recalculated by dividing their coordinates by this value

    Returns
    -------
        Add all the mesh of plant to the scene and return the list of index of organs

    """
    lsystem = Lsystem(filename)
    lstring = lsystem.derive(lsystem.axiom, t)
    lscene = lsystem.sceneInterpretation(lstring)
    # Adding the model of plant
    return addPlantModel(lscene, Tesselator(), tr2shmap, scene, anchor, scale_factor)


# add plant model to Scene
def addPlantModel(lscene, tr, tr2shmap, sc, anchor, scale_factor):
    """
    Add the PlantGL Shape of plant to the photon mapping scene. This function is calling by the function add_lpy_file_to_scene

    Parameters
    ----------
    lscene : Lscene
        The plantgl scene
    tr : Tesselator
        Tesselator
    sc : spice.Scene
        The photon mapping scene used to run the simulation
    tr2shmap : dict
        The dictionary of triangles of plant
    anchor : Vec3
        The position of the plant
    scale_factor : int
        The size of geometries. The vertices of geometries is recalculated by dividing their coordinates by this value

    """

    ctr = 0
    list_sh_id = set()
    for sh in lscene:
        sh.apply(tr)
        mesh = tr.result
        mesh.computeNormalList()
        index_list_size = mesh.indexListSize()
        vertices = VectorFloat([])
        normals = VectorFloat([])
        ind = []
        maxi = 0
        for i in range(0, index_list_size):
            index = mesh.indexAt(i)
            type_f = mesh.faceSize(i)
            for j in range(0, type_f):
                if index[j] > maxi:
                    maxi = index[j]
        for k in range(0, maxi + 1):
            mvector = mesh.pointAt(k)
            vertices.append(mvector[0] / (scale_factor / 10) + anchor[0])
            vertices.append(mvector[1] / (scale_factor / 10) + anchor[1])
            vertices.append(mvector[2] / (scale_factor / 10) + anchor[2])
        for k in range(0, maxi + 1):
            nvector = mesh.normalAt(k)
            normals.append(nvector[0])
            normals.append(nvector[1])
            normals.append(nvector[2])

        idx = flatten(mesh.indexList)
        for i in idx:
            if len(ind) > 0:
                i += len(ind)
                ctr += 1
        ind.extend(idx)
        indices = VectorUint(ind)
        r = float(sh.appearance.diffuseColor().red) / 255.0
        g = float(sh.appearance.diffuseColor().green) / 255.0
        b = float(sh.appearance.diffuseColor().blue) / 255.0
        diffuse = Vec3(r, g, b)
        ambient_r = float(sh.appearance.ambient.red) / 255.0
        ambient_g = float(sh.appearance.ambient.green) / 255.0
        ambient_b = float(sh.appearance.ambient.blue) / 255.0
        ambient = Vec3(ambient_r, ambient_g, ambient_b)
        shininess = sh.appearance.shininess
        specular_r = float(sh.appearance.specular.red) / 255.0
        specular_g = float(sh.appearance.specular.green) / 255.0
        specular_b = float(sh.appearance.specular.blue) / 255.0
        transparency = sh.appearance.transparency
        illum = 8  # to use the leaf bxdf

        refl = (r + g + b) / 3
        spec = (specular_r + specular_g + specular_b) / 3

        sc.addFaceInfos(
            vertices,
            indices,
            normals,
            diffuse,
            ambient,
            spec,
            shininess,
            transparency,
            illum,
            sh.name,
            1,
            refl,
            transparency,
            1.0 - shininess,
        )

        list_sh_id.add(sh.id)

        for _ in mesh.indexList:
            tr2shmap[ctr] = sh.id
            ctr += 1

    return list_sh_id


# add plant to a scene of PlantGL to visualize
def addPlantModelPgl(lscene, tr, sc, anchor, scale_factor, shenergy: dict):
    """
    Add the plant mesh to the PlantGL scene to visualize the scene

    Parameters
    ----------
    lscene : Lscene
        The plantgl scene
    tr : Tesselator
        Tesselator
    sc : spice.Scene
        The photon mapping scene used to run the simulation
    anchor : Vec3
        The position of the plant
    scale_factor : int
        The size of geometries. The vertices of geometries is recalculated by dividing their coordinates by this value
    shenergy : dict
        The dictionary of received energies in each organs of plant

    Returns
    -------
        A PlantGL Scene with the plant
    """

    pgl_scene = Scene()
    for sh in lscene:
        sh.apply(tr)
        mesh = tr.result
        mesh.computeNormalList()
        index_list_size = mesh.indexListSize()
        vertices = []
        maxi = 0
        for i in range(0, index_list_size):
            index = mesh.indexAt(i)
            type_f = mesh.faceSize(i)
            for j in range(0, type_f):
                if index[j] > maxi:
                    maxi = index[j]
        for k in range(0, maxi + 1):
            mvector = mesh.pointAt(k)
            vertices.append(
                Vector3(
                    (mvector[0] / (scale_factor / 10) + anchor[0]),
                    (mvector[1] / (scale_factor / 10) + anchor[1]),
                    (mvector[2] / (scale_factor / 10) + anchor[2]),
                ),
            )

        idx = mesh.indexList

        tmp_sh = Shape(TriangleSet(vertices, idx, mesh.normalList))
        tmp_sh.appearance = sh.appearance

        # change color of plant follow energy
        if shenergy:
            max_energy = shenergy[max(shenergy, key=shenergy.get)]

            cur_sh_energy = 0
            if sh.id in shenergy:
                cur_sh_energy = shenergy[sh.id]

            ratio = cur_sh_energy / max_energy
            r = int(255 * ratio)
            g = int(255 * ratio)
            b = int(255 * ratio)
            tmp_sh.appearance = Material(
                ambient=Color3(r, g, b), diffuse=sh.appearance.diffuse
            )

        pgl_scene.add(tmp_sh)

    return Scene([pgl_scene, sc])
