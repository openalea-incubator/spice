import pathlib
from openalea.plantgl.scenegraph import QuadSet, Shape

from openalea.plantgl.all import Scene as pglScene
from openalea.spice import Scene
from openalea.spice.loader.load_environment import addEnvironment, addLightDirectionPgl

filepath = pathlib.Path(__file__).parent.resolve() / 'data'

def test_addEnvironment():
    sc = Scene()
    # A list of the coordinates of the square
    points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    # A list of directions for the normals
    normals = [(0, 0, 1) for i in range(4)]
    # A list of indices that set the indices for the quads
    indices = [(0, 1, 2, 3)]
    # Creation of the quadset
    sh = Shape(QuadSet(points, indices, normals, indices))

    addEnvironment(sc, sh, 400, {},{},{})

    addLightDirectionPgl(pglScene([sh]), 1)


def test_load_GLTF():
    sc = Scene()
    sc.loadModelGLTF(str(filepath / 'cornellbox-water2.glb'))

    assert sc.nFaces() > 0

