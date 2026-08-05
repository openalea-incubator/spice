import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt

from openalea.spice.libspice import (
    Render,
    visualizeSensorsPhotonMap,
    visualizePhotonMap,
    PhotonMapping,
    UniformSampler,
    Vec3,
    Scene,
    Image,
    Camera,
)
try:
    from openalea.lpy import Lsystem
except ImportError:
    pass

import openalea.plantgl.all as pgl

from openalea.spice.energy import calculate_energy, correct_energy
from openalea.spice.loader import load_sensor, load_environment
from openalea.spice.loader.load_sensor import Sensor
from openalea.spice.reader import read_properties, read_rad_geo
from openalea.spice.common.math import denormalize
from openalea.spice.configuration import Configuration


@dataclass
class SimulationResult:
    """
    A class which contains the result of simulation.

    Attributes
    ----------

    N_sim_virtual_sensor: dict
        Number of received photons on each sensor
    N_sim_face_sensor: dict
        Number of received photons on each organ of plant
    N_mes_virtual_sensor: dict
        The energies after the calibration on each sensor
    N_mes_face_sensor: dict
        The energies after the calibration on each organ of plant
    divided_spectral_range: array
        The list of spectral ranges divided from the base spectral range.
    list_virtual_sensor: array
        The list of virtual sensor in simulation

    """

    # constructor

    def __init__(self, simulator):
        self.N_sim_virtual_sensor = simulator.N_sim_virtual_sensor
        self.N_sim_face_sensor = simulator.N_sim_face_sensor
        self.N_mes_virtual_sensor = simulator.N_mes_virtual_sensor
        self.N_mes_face_sensor = simulator.N_mes_face_sensor

        self.divided_spectral_range = simulator.configuration.DIVIDED_SPECTRAL_RANGE
        self.list_virtual_sensor = simulator.list_virtual_sensor
        self.list_face_sensor = simulator.list_face_sensor

    def writeResults(self, file_prefix=""):
        """
        Write the result of simulation to a file saved in the folder ./results

        Parameters
        ----------
        file_prefix: str
            The prefix of output file
        """

        if len(self.N_sim_virtual_sensor) > 0:
            calculate_energy.write_sensor_energy(
                self.N_sim_virtual_sensor,
                self.N_mes_virtual_sensor,
                self.list_virtual_sensor,
                self.divided_spectral_range,
                file_prefix + "virtual_sensor_res.csv",
            )

        if len(self.N_sim_face_sensor) > 0:
            calculate_energy.write_sensor_energy(
                self.N_sim_face_sensor,
                self.N_mes_face_sensor,
                self.list_face_sensor,
                self.divided_spectral_range,
                file_prefix + "face_sensor_res.csv",
            )

    def display_face_sensor(self):
        """
        Draw a graph with MathPlotlib

        """
        _, ax = plt.subplots(figsize=(12, 8))
        for wavelength_measured in self.N_sim_face_sensor:
            str_keys = [str(key) for key in wavelength_measured.keys()]
            plt.bar(str_keys, wavelength_measured.values(), color="g", width=0.2)
        ax.set_xlabel("Shape id")
        ax.set_ylabel("Number of photons")
        plt.setp(ax.get_xticklabels(), rotation=90)
        plt.tight_layout()
        plt.show()

    def display_virtual_sensor(self):
        """
        Draw a graph with MathPlotlib

        """
        _, ax = plt.subplots()
        for wavelength_measured in self.N_sim_virtual_sensor:
            str_keys = [str(key) for key in wavelength_measured.keys()]
            plt.bar(str_keys, wavelength_measured.values(), color="g", width=0.2)
        ax.set_xlabel("Sensor id")
        ax.set_ylabel("Number of photons")
        plt.setp(ax.get_xticklabels(), rotation=90)
        plt.tight_layout()
        plt.show()


class Simulator:
    """
    A class which include all the tools of simulation.

    Attributes
    ----------
    configuration: Configuration
        The configuration of the engine
    N_sim_virtual_sensor: dict
        Number of received photons on each sensor
    N_sim_face_sensor: dict
        Number of received photons on each organ of plant

    spectrum_file: str
        The link to the file which contains the information of the
        heterogeneity of the spectrum
    points_calibration_file: str
        The link to the file which contains the information of the sensors
        used to calibrate the final result

    list_virtual_sensor: array
        The list of virtual sensor in simulation
    scene_pgl: openalea.plantgl.scenegraph.Scene
        The plantgl scene object used to save the meshes of environment
    po_dir: str
        The link to the folder which contains the optical properties of the room

    """

    # constructor

    def __init__(self, config_file="", configuration=Configuration()):
        self.camera = None
        self.coeffs_calibration = None
        self.integrals = None
        self.points_calibration = None
        self.configuration = configuration
        if config_file != "":
            self.configuration.read_file(Path(config_file))
        self.n_samples = 512
        #
        self.scene_pgl = pgl.Scene()
        self.scene = Scene()
        self.list_virtual_sensor = []
        self.virtual_sensor_triangle_dict = {}
        self.virtual_sensor_energy = {}
        self.list_face_sensor = []
        self.face_sensor_triangle_dict = {}
        self.face_sensor_energy = {}
        self.list_light = []

        # result
        self.N_sim_virtual_sensor = []
        self.N_sim_face_sensor = []
        self.N_mes_virtual_sensor = []
        self.N_mes_face_sensor = []
        self.photonmaps = []
        self.integrators = []
        self.results = None

        self.image_width = 512
        self.image_height = 512

        # input files
        if self.configuration.ENVIRONMENT_FILE.is_file():
            self.addEnvFromFile()
        self.spectrum_file = ""
        self.points_calibration_file = ""

    def resetScene(self):
        """
        Clear list of sensors and list of object in scene
        """
        self.scene_pgl.clear()
        self.list_virtual_sensor.clear()
        self.list_face_sensor.clear()

    def addEnvToScene(self, sh):
        """
        Add an environment's object to scene

        Parameters
        ----------
        sh : plantgl.Shape
            The mesh of object

        Returns
        -------
            The object is added to the scene

        """
        vertices = sh.geometry.pointList
        # apply scale factor
        for i in range(len(vertices)):
            vertices[i] = tuple(
                x / self.configuration.SCALE_FACTOR for x in vertices[i]
            )

        sh.geometry.pointList = vertices
        sh.geometry.computeNormalList()

        # add object to scene
        self.scene_pgl.add(sh)

    def addPointLight(self, position, intensity, color=Vec3(1, 1, 1)):
        self.scene.addPointLight(position, intensity, color)
        self.list_light.append(
            {
                "type": "point",
                "position": position,
                "parameters": {"intensity": intensity, "color": color},
            }
        )

    def addSpotLight(self, position, intensity, direction, angle, color=Vec3(1, 1, 1)):
        self.scene.addSpotLight(position, intensity, color, direction, angle)
        self.list_light.append(
            {
                "type": "Spot",
                "position": position,
                "parameters": {
                    "intensity": intensity,
                    "color": color,
                    "angle": angle,
                    "direction": direction,
                },
            }
        )

    def addFaceSensor(self, shape):
        """
        Add a face sensor object to scene

        Parameters
        ----------
        shape: Shape
            The geometry and material of sensor
        Returns
        -------
            The face sensor is added to the scene

        """
        sensor = Sensor(shape, "FaceSensor")
        self.list_face_sensor.append(sensor)
        load_sensor.addFaceSensors(
            self.scene, self.face_sensor_triangle_dict, self.list_face_sensor
        )

    def addFaceSensors(self, shapes):
        """
        Add face sensors object to scene

        Parameters
        ----------
        shapes: Shape
            The geometry and material of sensor
        Returns
        -------
            The face sensor is added to the scene

        """
        for shape in shapes:
            sensor = Sensor(shape, "FaceSensor")
            self.list_face_sensor.append(sensor)
        load_sensor.addFaceSensors(
            self.scene, self.face_sensor_triangle_dict, self.list_face_sensor
        )

    def addFaceSensorToScene(self, shape, position, scale_factor):
        """
        Add a face sensor object to scene

        Parameters
        ----------
        shape: Shape
            The geometry and material of sensor
        position : tuple
            The position of sensor
        scale_factor: float
            The size of geometries. The vertices of geometries is recalculated
            by dividing their coordinates by this value

        Returns
        -------
            The face sensor is added to the scene

        """
        sensor = Sensor(shape, "FaceSensor", position, scale_factor)
        self.list_face_sensor.append(sensor)

    def addVirtualSensorToScene(self, shape, position, scale_factor):
        """
        Add a virtual sensor object to scene

        Parameters
        ----------
        shape: Shape
            The geometry and material of sensor
        position : tuple(int,int,int)
            The position of sensor
        scale_factor: int
            The size of geometries. The vertices of geometries is recalculated
            by dividing their coordinates by this value

        Returns
        -------
            The virtual sensor is added to the scene

        """
        sensor = Sensor(shape, "VirtualSensor", position, scale_factor)
        self.list_virtual_sensor.append(sensor)

    def addVirtualDiskSensorToScene(self, pos, normal, r, sensor_id):
        """
        Add a virtual disk shaped sensor object to scene

        Parameters
        ----------
        pos : tuple
            The position of sensor
        normal: tuple
            The vector normal of sensor
        r: float
            The radius of sensor
        sensor_id: int
            The id of sensor

        Returns
        -------
            The disk shaped sensor is added to the scene

        """
        sensor = Sensor(
            pgl.Shape(),
            "VirtualSensor",
            (
                pos[0] / self.configuration.SCALE_FACTOR,
                pos[1] / self.configuration.SCALE_FACTOR,
                pos[2] / self.configuration.SCALE_FACTOR,
            ),
        )
        sensor.initVirtualDiskSensor(
            (normal[0], normal[1], normal[2]),
            r,
            sensor_id,
        )

        self.list_virtual_sensor.append(sensor)

    def applyWavelengthProperties(self, current_band, average_wavelength):
        """
        Setup all the material for a simulation in a waveband.

        Parameters
        ----------

        current_band: dict
            Current divided spectral range where the simulation is running
        average_wavelength: Vec3
            The average wavelength of spectral range used to determine the color
            of the light
        """
        materials_r, materials_s, materials_t = read_properties.setup_dataset_materials(
            current_band["start"],
            current_band["end"],
            self.configuration.OPTICAL_PROPERTIES_DIR,
        )

        for sh in self.scene_pgl:
            refl = materials_r.get(sh.id, 0.9)
            trans = materials_t.get(sh.id, 0.7)
            spec = materials_t.get(sh.id, 0.0)
            self.scene.setMatPrimitive(str(sh.id), refl, trans, spec)

    def run(self):
        """
        Run the simulation with the configurations which is determined

        Returns
        -------
            The number of received photons on each sensor and organs of
            plant is saved into the files located in folder ./results

        """

        self.photonmaps.clear()
        self.integrators.clear()
        self.N_sim_face_sensor.clear()
        self.N_sim_virtual_sensor.clear()
        self.N_mes_virtual_sensor.clear()
        self.N_mes_face_sensor.clear()

        n_estimation_global = 100
        final_gathering_depth = 4

        for index in range(len(self.configuration.DIVIDED_SPECTRAL_RANGE)):
            start_time = time.time()
            current_band = self.configuration.DIVIDED_SPECTRAL_RANGE[index]

            print("Wavelength:", current_band["start"], "-", current_band["end"])
            average_wavelength = (current_band["start"] + current_band["end"]) / 2
            # if average_wavelength != 0: # then use actual wavelength sim
            self.applyWavelengthProperties(current_band, average_wavelength)

            # create integrator
            self.scene.tnear = self.configuration.T_MIN
            self.scene.build(self.configuration.BACKFACE_CULLING)
            integrator = PhotonMapping(
                self.configuration.NB_PHOTONS,
                n_estimation_global,
                final_gathering_depth,
                self.configuration.MAXIMUM_DEPTH,
                self.configuration.NB_THREAD,
            )

            self.integrators.append(integrator)
            print("Build photonmap...")

            sampler = UniformSampler(random.randint(1, sys.maxsize))

            # build no kdtree if not rendering
            integrator.build(self.scene, sampler, self.configuration.KEEP_ALL)
            print("Done!")

            # # rendering if declared
            if self.configuration.RENDERING:
                self.render(integrator, self.scene, average_wavelength, sampler)

            # # read energy of virtual sensor
            if len(self.list_virtual_sensor) > 0:
                calculate_energy.sensor_add_energy(
                    self.virtual_sensor_triangle_dict,
                    integrator,
                    self.virtual_sensor_energy,
                )
                self.N_sim_virtual_sensor.append(self.virtual_sensor_energy)

            # read energy of face sensor
            if len(self.list_face_sensor) > 0:
                calculate_energy.sensor_add_energy(
                    self.face_sensor_triangle_dict, integrator, self.face_sensor_energy
                )
                self.N_sim_face_sensor.append(self.face_sensor_energy)

            self.photonmaps.append(integrator.getPhotonMap())
            self.photonmaps.append(integrator.getPhotonMapSensors())
            print("Time taken: " + str(time.time() - start_time))
        self.results = SimulationResult(self)

    def calculateCalibrationCoefficient(
        self, spectrum_file="", points_calibration_file=""
    ):
        """
        Calculate the coefficients which is used to calibrate the final result
        of simulation with the sensors

        Parameters
        ----------
        spectrum_file: str
            The link to the file which contains the information of the
            heterogeneity of the spectrum
        points_calibration_file: str
            The link to the file which contains the information of the
            sensors used to calibrate the final result

        """

        self.coeffs_calibration = []

        if os.path.exists(spectrum_file) and os.path.exists(points_calibration_file):
            self.integrals = correct_energy.get_correct_energy_coeff(
                self.configuration.BASE_SPECTRAL_RANGE,
                self.configuration.DIVIDED_SPECTRAL_RANGE,
                spectrum_file,
            )
            self.points_calibration = correct_energy.get_points_calibration(
                self.list_virtual_sensor,
                points_calibration_file,
                self.configuration.DIVIDED_SPECTRAL_RANGE,
            )
            self.coeffs_calibration = correct_energy.get_calibration_coefficient(
                self.N_sim_virtual_sensor, self.integrals, self.points_calibration
            )
            return True

        return False

    def calibrateResults(self, spectrum_file="", points_calibration_file=""):
        """
        Calibrate the final result of simulation

        Parameters
        ----------
        spectrum_file: str
            The link to the file which contains the information of the
            heterogeneity of the spectrum
        points_calibration_file: str
            The link to the file which contains the information of the sensors
            used to calibrate the final result

        """

        can_calibrate = self.calculateCalibrationCoefficient(
            spectrum_file, points_calibration_file
        )

        if can_calibrate:
            if len(self.N_sim_virtual_sensor) > 0:
                self.N_mes_virtual_sensor = correct_energy.calibrate_sensor_energy(
                    self.N_sim_virtual_sensor,
                    self.integrals,
                    self.points_calibration,
                    self.coeffs_calibration,
                )

            if len(self.coeffs_calibration) > 0 and len(self.N_sim_face_sensor) > 0:
                self.N_mes_face_sensor = correct_energy.calibrate_plant_energy(
                    self.N_sim_face_sensor, self.coeffs_calibration
                )

        return SimulationResult(self)

    def visualizeResults(self, mode="ipython", wavelength_index=0, colormap="jet"):
        """
        Visualize the scene of simulation with the tools of OpenAlea
        To run this function, it has to run these command first:
        -- ipython
        -- %gui qt5

        Parameters
        ----------
        colormap: str
            A matplotlib colormap name to display values with.
        wavelength_index: int
            The wavelength index to visualize.
        mode: str
            This variable define the mode used to visualize the scene.
            There are the supported modes: ipython, oawidgets

        Returns
        -------
            A rendered scene in 3D

        """
        cmap = matplotlib.cm.get_cmap(colormap)

        # Face sensors results
        if len(self.N_sim_face_sensor) > 0:
            faces_values = self.N_sim_face_sensor[wavelength_index].values()

            minimum = 0
            maximum = max(faces_values)

            norm = matplotlib.colors.Normalize(vmin=minimum, vmax=maximum)
            for sh in self.scene_pgl:
                if sh.id in self.N_sim_face_sensor[wavelength_index].keys():
                    color = cmap(norm(self.N_sim_face_sensor[wavelength_index][sh.id]))
                    sh.appearance = pgl.Material(
                        pgl.Color3(
                            denormalize(color[0]),
                            denormalize(color[1]),
                            denormalize(color[2]),
                        )
                    )
        if len(self.N_sim_virtual_sensor) > 0:
            virt_values = self.N_sim_virtual_sensor[wavelength_index].values()

            minimum = 0
            maximum = max(virt_values)
            shape_to_id = {}
            for sensor in self.list_virtual_sensor:
                shape_to_id[sensor.shape.id] = sensor.sensor_id

            norm = matplotlib.colors.Normalize(vmin=minimum, vmax=maximum)
            for sh in self.scene_pgl:
                if sh.id in shape_to_id.keys():
                    if (
                        shape_to_id[sh.id]
                        in self.N_sim_virtual_sensor[wavelength_index].keys()
                    ):
                        color = cmap(
                            norm(
                                self.N_sim_virtual_sensor[wavelength_index][
                                    shape_to_id[sh.id]
                                ]
                            )
                        )
                        sh.appearance = pgl.Material(
                            pgl.Color3(
                                denormalize(color[0]),
                                denormalize(color[1]),
                                denormalize(color[2]),
                            )
                        )

        if mode == "ipython":
            pgl.Viewer.display(self.scene_pgl)
            return None
        elif mode == "oawidgets":
            from openalea.widgets.plantgl import PlantGL
            import k3d

            plot = PlantGL(self.scene_pgl, side="double")
            plot.grid_visible = False
            i = 1
            for light in self.list_light:
                pos = light["position"]
                pos = (pos[0], pos[1], pos[2])
                light_point = k3d.points(
                    [pos],
                    point_size=1 * self.configuration.SCALE_FACTOR,
                    color=0xFFFFFF,
                )
                light_label = k3d.label(text=f"light n°{i}", position=pos)

                plot += light_point + light_label
                i += 1
            plot.camera_reset()

            return plot
        else:
            pgl.Viewer.display(self.scene_pgl)
            return None

    def get_photons_per_triangles(self):
        """
        Returns a counter object with the number of photon per triangle
        Returns
        -------
            a Counter object.
        """
        photons_triangles = Counter()
        for phmap in self.photonmaps:
            for i in range(phmap.nPhotons()):
                triangle_id = phmap.getIthPhoton(i).triId
                photons_triangles[triangle_id] += 1

        return photons_triangles

    def visualizeRays(self, mode="oawidgets", color=0x0000FF, opacity=1.0):
        photons = []

        def r():
            return random.randint(0, 255)

        # creating color per light source
        colors = []
        light_range = range(len(self.list_light))
        n_light = self.scene.nLights()
        for _ in light_range:
            colors.append((r(), r(), r()))
        for _ in range(n_light):
            colors.append((r(), r(), r()))
        for phmap in self.photonmaps:
            for i in range(phmap.nPhotons()):
                photon = phmap.getIthPhoton(i).position
                light_id = phmap.getIthPhoton(i).lightId
                photons.append(((photon[0], photon[1], photon[2]), light_id))
        if mode == "oawidgets":
            import k3d
            from k3d.colormaps import matplotlib_color_maps

            plot = k3d.plot()
            plot.grid_visible = False
            indices, lines, light_ids = [], [], []
            i = 0
            for ph in photons:
                pos = ph[0]
                id = ph[1]

                light = self.list_light[id]["position"]
                light_pos = (light[0], light[1], light[2])
                light_ids.append([id, id])
                lines.append([pos, light_pos])
                indices.append([i, i + 1])
                i += 2
            lines = k3d.lines(
                lines,
                indices,
                indices_type="segment",
                color=color,
                opacity=opacity,
                shader="simple",
                attribute=light_ids,
                color_map=matplotlib_color_maps.Rainbow,
                color_range=[0, n_light],
            )
            plot += lines
            plot.camera_reset()
            return plot

    def visualizePhotons(self, mode="ipython"):
        """
        Visualize the scene of simulation with the tools of OpenAlea
        To run this function, it has to run these command first:
        -- ipython
        -- %gui qt5

        Parameters
        ----------
        mode: str
            This variable define the mode used to visualize the scene.
            There are the supported modes: ipython, oawidgets

        Returns
        -------
            A rendered scene in 3D

        """
        photons = []

        def r():
            return random.randint(0, 255)

        # creating color per light source
        colors = []
        light_range = range(len(self.list_light))
        n_light = self.scene.nLights()
        for _ in light_range:
            colors.append((r(), r(), r()))
        for _ in range(n_light):
            colors.append((r(), r(), r()))

        for phmap in self.photonmaps:
            for i in range(phmap.nPhotons()):
                photon = phmap.getIthPhoton(i).position
                light_id = phmap.getIthPhoton(i).lightId
                photons.append(((photon[0], photon[1], photon[2]), light_id))

        if mode == "ipython":
            ph_sc = pgl.Scene()
            for photon in photons:
                sp = pgl.Sphere(0.1 * self.configuration.SCALE_FACTOR, 4)
                position = photon[0]
                color = colors[photon[1]]
                s2 = pgl.Translated(position[0], position[1], position[2], sp)
                sh = pgl.Shape(
                    s2, pgl.Material(pgl.Color3(color[0], color[1], color[2]))
                )
                ph_sc.add(sh)
            ph_sc.merge(self.scene_pgl)
            pgl.Viewer.display(ph_sc)
            return None
        elif mode == "oawidgets":
            import k3d
            from k3d.colormaps import matplotlib_color_maps

            positions = [photon[0] for photon in photons]
            lights = [photon[1] for photon in photons]

            points = k3d.points(
                positions,
                point_size=1 * self.configuration.SCALE_FACTOR,
                shader="3d",
                attribute=lights,
                color_map=matplotlib_color_maps.Rainbow,
                color_range=[0, n_light],
            )
            plot = self.visualizeScene("oawidgets")
            plot.grid_visible = False
            plot += points
            i = 0
            for light in self.list_light:
                pos = light["position"]
                pos = (pos[0], pos[1], pos[2])
                light_point = k3d.points(
                    [pos],
                    point_size=1 * self.configuration.SCALE_FACTOR,
                    color=0xFFFFFF,
                )
                light_label = k3d.label(text=f"light n°{i}", position=pos)

                plot += light_point + light_label
                i += 1
            plot.camera_reset()
            return plot
        else:
            pgl.Viewer.display(self.scene_pgl)

    def visualizeScene(self, mode="ipython"):
        """
        Visualize the scene of simulation with the tools of OpenAlea
        To run this function, it has to run these command first:
        -- ipython
        -- %gui qt5

        Parameters
        ----------
        mode: str
            This variable define the mode used to visualize the scene.
            There are the supported modes: ipython, oawidgets

        Returns
        -------
            A rendered scene in 3D

        """
        if mode == "ipython":
            pgl.Viewer.display(self.scene_pgl)

        elif mode == "oawidgets":
            from openalea.widgets.plantgl import PlantGL

            plot = PlantGL(self.scene_pgl, side="double")
            plot.camera_reset()
            return plot

        else:
            pgl.Viewer.display(self.scene_pgl)

    def test_t_min(self, nb_photons, start_t, loop, is_only_lamp=False):
        """
        Test the simulation with multiple values of Tmin to avoid the problem
        of auto-intersection

        Parameters
        ----------
        nb_photons: int
            The total number of photons is shooting from the light in the
            simulation
        start_t: float
            The first (smallest) value of Tmin used to run the test
        loop: int
            The number of iteration. At each iteration, the current value Tmin
            is multiply with 10, then run the simulation
        is_only_lamp: bool
            If True, run the test with only the lamps and sensors, If False,
            run the test with all the objects in scene

        Returns
        -------
            A graph is generated to show the connection between the Tmin and
            the results of simulation

        """

        if loop < 1:
            return

        self.scene = Scene()
        n_estimation_global = 100
        final_gathering_depth = 0
        current_band = self.configuration.DIVIDED_SPECTRAL_RANGE[0]
        average_wavelength = (current_band["start"] + current_band["end"]) / 2

        list_tmin = []
        list_res = []
        list_index = []

        (
            self.scene,
            _,
            _,
            _,
            _,
        ) = self.initSimulationScene(
            self.scene, current_band, average_wavelength, is_only_lamp
        )
        # create integrator
        self.scene.setupTriangles()
        self.scene.build(self.configuration.BACKFACE_CULLING)

        for i in range(loop):
            print("---------------------------------")
            print("Test Tmin =", start_t)
            self.scene.tnear = start_t

            integrator = PhotonMapping(
                nb_photons,
                n_estimation_global,
                final_gathering_depth,
                self.configuration.MAXIMUM_DEPTH,
                self.configuration.NB_THREAD,
            )

            sampler = UniformSampler(1)

            # build no kdtree if not rendering
            integrator.build(self.scene, sampler, False)

            res = integrator.getPhotonMapSensors().nPhotons()
            print("Number of photons received in total is", res)

            list_tmin.append(start_t)
            list_res.append(res)
            list_index.append(i)
            start_t = round(start_t * 10, 6)
            print("---------------------------------")

        plt.plot(list_index, list_res, linestyle="--", marker="*")
        plt.title("Number of photons received relative to the change in tmin")
        plt.ylabel("Nb of photon")
        for x, y, text in zip(list_index, list_res, list_tmin):
            plt.text(x, y, text)
        plt.show()

    def initSimulationScene(
        self, scene, current_band, average_wavelength, is_only_lamp=False
    ):
        """
        Setup all the necessary objects (environment, sensor, plant) in the
        simulation

        Parameters
        ----------

        scene: spice.Scene
            The object which contains all the object in the scene of simulation
        current_band: dict
            Current divided spectral range where the simulation is running
        average_wavelength: Vec3
            The average wavelength of spectral range used to determine the color
            of the light
        is_only_lamp: bool
            if True, only the lamps and sensors is added to the scene, if False,
             all the objects is added.


        Returns
        -------
            scene: spice.Scene
                The object which contains all the object in the scene of
                simulation
            has_virtual_sensor: bool
                Return true if the scene has the sensors
            sensor_triangle_dict: dict
                Dictionary of the triangles of the sensors. Using to counting
                the number of photons received in each sensor
            has_plant: bool
                Return true if the scene has the model of plant
            tr2shmap: dict
                Dictionary of the triangles of the plant. Using to counting the
                number of photons received in each organ of plant

        """
        # add env
        materials_r, materials_s, materials_t = read_properties.setup_dataset_materials(
            current_band["start"],
            current_band["end"],
            self.configuration.OPTICAL_PROPERTIES_DIR,
        )

        for sh in self.scene_pgl:
            load_environment.addEnvironment(
                scene,
                sh,
                average_wavelength,
                materials_r,
                materials_s,
                materials_t,
                is_only_lamp,
            )

        # add face sensor
        has_face_sensor = len(self.list_face_sensor) > 0
        face_sensor_triangle_dict = {}

        if has_face_sensor:
            load_sensor.addFaceSensors(
                scene, face_sensor_triangle_dict, self.list_face_sensor
            )

        # add virtual sensor
        has_virtual_sensor = len(self.list_virtual_sensor) > 0
        virtual_sensor_triangle_dict = {}

        if has_virtual_sensor:
            load_sensor.addVirtualSensors(
                scene, virtual_sensor_triangle_dict, self.list_virtual_sensor
            )

        return (
            scene,
            has_virtual_sensor,
            virtual_sensor_triangle_dict,
            has_face_sensor,
            face_sensor_triangle_dict,
        )

    def render(self, integrator, scene, w, sampler):
        """
        Visualize the photon map of the scene and render an image from it.

        Parameters
        ----------
        integrator: spice.PhotonMapping
            The object which handles all the simulation of photon mapping.
        scene: spice.Scene
            The object which contains all the object in the scene of simulation.
        w: Vec3
            The average wavelength of spectral range used to determine the color
             of the light.
        sampler: spice.Sampler
            The generator of the random number.

        Returns
        -------
            Rendered images is saved into the folder ./result

        """

        os.makedirs("results", exist_ok=True)

        if not self.configuration.RENDERING:
            print("Enable rendering first !!!")
            return

        image = Image(self.image_width, self.image_height)
        print("Printing photonmap image...")
        visualizePhotonMap(
            integrator,
            scene,
            image,
            self.image_height,
            self.image_width,
            self.camera,
            self.configuration.NB_PHOTONS,
            self.configuration.MAXIMUM_DEPTH,
            "results/photonmap-" + str(w) + "nm.ppm",
            sampler,
        )
        image.clear()
        print("Done!")

        print("Printing sensor photonmap image...")
        visualizeSensorsPhotonMap(
            scene,
            image,
            self.image_height,
            self.image_width,
            self.camera,
            self.configuration.NB_PHOTONS,
            self.configuration.MAXIMUM_DEPTH,
            "results/photonmap-sensors-" + str(w) + "nm.ppm",
            sampler,
            integrator,
        )
        image.clear()
        print("Done!")

        print("Rendering image...")
        Render(
            sampler,
            image,
            self.image_height,
            self.image_width,
            self.n_samples,
            self.camera,
            integrator,
            scene,
            "results/render-" + str(w) + "nm.ppm",
        )
        image.clear()
        print("Done!")

    def addEnvFromFile(self, flip_normal=False):
        """
        Set up the room/environment of the simulation from file.

        Parameters
        ----------
        flip_normal: bool
            Determine the direction of the vector normal of triangle.
        """

        self.scene_pgl = read_rad_geo.read_rad(
            self.configuration.ENVIRONMENT_FILE,
            self.configuration.SCALE_FACTOR,
            flip_normal,
        )

    def setup(self):
        self.addFaceSensors(self.scene_pgl)

        self.scene_pgl = load_sensor.addSensorPgl(
            self.scene_pgl, self.list_virtual_sensor
        )

        load_sensor.addVirtualSensors(
            self.scene, self.virtual_sensor_triangle_dict, self.list_virtual_sensor
        )

        current_band = self.configuration.DIVIDED_SPECTRAL_RANGE[0]
        average_wavelength = (current_band["start"] + current_band["end"]) / 2

        materials_r, materials_s, materials_t = read_properties.setup_dataset_materials(
            current_band["start"],
            current_band["end"],
            self.configuration.OPTICAL_PROPERTIES_DIR,
        )

        for sh in self.scene_pgl:
            load_environment.addEnvironment(
                self.scene,
                sh,
                average_wavelength,
                materials_r,
                materials_s,
                materials_t,
            )


        self.scene.setupTriangles()

    def setupRender(self, lookfrom=Vec3(0, 0, 0), lookat=Vec3(0, 0, 0), vfov=50.0):
        """
        Enable the capacity to render an image of the scene.

        Parameters
        ----------
        lookfrom: Vec3
            The position of the camera.
        lookat: Vec3
            The point where the camera is looking at
        vfov: float
            The vertical field of view of the camera

        """
        self.configuration.RENDERING = True
        # using for render the results
        self.camera = self.initCameraRender(lookfrom, lookat, vfov)

    def addVirtualDiskSensorsFromFile(self, sensor_file: str):
        """
        Set up the sensors in the simulation. Enable the capacity to run the
        simulation with the circle sensors

        Parameters
        ----------
        sensor_file: str
            The link to the file which contains the information of the sensors
            in the simulation

        """

        if sensor_file != "":
            sensor_id = len(self.list_virtual_sensor)
            with open(sensor_file, encoding="UTF8") as f:
                next(f)
                for line in f:
                    row = line.split(",")
                    x = float(row[0])
                    y = float(row[1])
                    z = float(row[2])
                    r = float(row[3])
                    xnorm = float(row[4])
                    ynorm = float(row[5])
                    znorm = float(row[6])

                    self.addVirtualDiskSensorToScene(
                        (x, y, z), (xnorm, ynorm, znorm), r, sensor_id
                    )
                    sensor_id += 1

    def addFaceSensorsFromLpyFile(
        self, plant_file: str, plant_pos=Vec3(0, 0, 0), derivation_length=None
    ):
        """
        Set up a plant in the simulation. Enable the capacity to run the
        simulation with a model of plant

        Parameters
        ----------
        plant_pos: Vec3
            The position in the scene to put the plant at.
        plant_file: str
            The link to the file of the model of plant. (currently only support
            .lpy file)
        derivation_length: int
            The number of iteration to interpret the plant
        """

        lsystem = Lsystem(plant_file)
        if derivation_length is None:
            derivation_length = lsystem.derivationLength

        lstring = lsystem.derive(lsystem.axiom, derivation_length)
        lscene = lsystem.sceneInterpretation(lstring)

        scale_factor = self.configuration.SCALE_FACTOR / 10
        position = (plant_pos[0] / 10, plant_pos[1] / 10, plant_pos[2] / 10)

        tr = pgl.Tesselator()
        for sh in lscene:
            sh.apply(tr)
            mesh = pgl.Shape(tr.result, sh.appearance, sh.id)
            self.scene_pgl.add(mesh)
            self.addFaceSensorToScene(mesh, position, scale_factor)

    @staticmethod
    def initCameraRender(lookfrom=Vec3(0, 0, 0), lookat=Vec3(0, 0, 0), vfov=50.0):
        """
        Init the camera to render image. Called by the function setupRender

        Parameters
        ----------
        lookfrom: Vec3
            The position of the camera.
        lookat: Vec3
            The point where the camera is looking at
        vfov: float
            The vertical field of view of the camera

        Returns
        -------
        camera: spice.Camera
            An object with all the information of camera
        """

        aspect_ratio = 16.0 / 9.0

        vup = Vec3(0, 0, -1)
        dist_to_focus = 2.0
        aperture = 0.01

        # coordinates must be in meters
        camera = Camera(
            lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus
        )

        return camera
