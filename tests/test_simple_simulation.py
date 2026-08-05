import pathlib

import pytest
from openalea.plantgl.all import Color3, Material, Shape, TriangleSet
from openalea.spice import Vec3

from openalea.spice.configuration import Configuration
from openalea.spice.simulator import Simulator
import shutil

filepath = pathlib.Path(__file__).parent.resolve() / 'data'

def test_configuration():
    simulator = Simulator(config_file=filepath / "simulation.ini")
    assert type(simulator.configuration) is Configuration

    simulator.configuration.read_file(filepath / "simulation_2.ini")
    assert type(simulator.configuration) is Configuration
    assert simulator.configuration.KEEP_ALL == 0

    try:
        conf = Configuration()
        conf.read_file(filepath / "does not exist.txt")
    except FileNotFoundError:
        pass

def test_simple_simulation():
    simulator = Simulator(config_file=filepath / "simulation.ini")

    # setup configuration
    simulator.configuration.nb_photons = 1000000
    simulator.configuration.max_depth = 5
    simulator.configuration.DIVIDED_SPECTRAL_RANGE = [{"start": 655, "end": 665},
                                                      {"start": 600, "end": 655}]
    simulator.resetScene()

    # setup environment
    ground_ts = TriangleSet(
        pointList=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], indexList=[(0, 1, 2)]
    )
    ground_mat = Material(
        name="Ground",
        ambient=Color3(0),
        specular=Color3(127),  # spec = 0.5 = 127/255
        shininess=1,
        transparency=0,
    )
    ground_sh = Shape(ground_ts, ground_mat)

    simulator.addEnvToScene(ground_sh)

    # setup light
    light_ts = TriangleSet(
        pointList=[(0, 0, 5), (1, 0, 5), (0, 1, 5)], indexList=[(0, 1, 2)]
    )
    light_mat = Material(name="Light", emission=Color3(255, 255, 255))
    light_sh = Shape(light_ts, light_mat)
    simulator.addEnvToScene(light_sh)

    # setup sensor
    sensor_ts = TriangleSet(
        pointList=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], indexList=[(0, 2, 1)]
    )
    sensor_mat = Material(
        name="Sensor",
        ambient=Color3(127),
        specular=Color3(127),  # spec = 0.5 = 127/255
        shininess=0.5,
        transparency=0.5,
    )
    sensor_sh = Shape(sensor_ts, sensor_mat, 0)

    # simulator.addFaceSensorToScene(shape=sensor_sh, position=(0,0,1), scale_factor=1)
    simulator.addVirtualSensorToScene(
        shape=sensor_sh, position=(0, 0, 3), scale_factor=1
    )
    simulator.addFaceSensorsFromLpyFile(str(filepath / 'rose-simple4.lpy'))

    simulator.addPointLight(Vec3(0,5,0), 1000)

    # run
    simulator.setup()
    simulator.run()
    simulator.calibrateResults(str(filepath / "chambre1_spectrum"),
                               str(filepath / "point_calibration.csv"))

    simulator.results.writeResults()
    shutil.rmtree('./results')


@pytest.mark.skip(reason="Segfault with pytest")
def test_visualization():
    simulator = Simulator(config_file=filepath / "simulation.ini")
    simulator.run()

    simulator.visualizeScene()
    simulator.visualizePhotons()
    simulator.visualizeRays()