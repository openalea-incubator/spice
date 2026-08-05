// #define __OUTPUT__

#include "../cpp/include/camera.hpp"
#include "../cpp/include/image.hpp"
#include "../cpp/include/integrator.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/operators.h>


namespace nb = nanobind;

NB_MAKE_OPAQUE(std::vector<float>)
NB_MAKE_OPAQUE(std::vector<uint32_t>)
NB_MAKE_OPAQUE(std::vector<int>)
NB_MAKE_OPAQUE(std::vector<Vec3f>)

void
visualizePhotonMap(const PhotonMapping& integrator,
                   const Scene& scene,
                   Image& image,
                   const unsigned& height,
                   const unsigned& width,
                   const Camera& camera,
                   const unsigned& n_photons,
                   const unsigned& max_depth,
                   const std::string_view& filename,
                   UniformSampler& sampler)
{
        // visualize photon map
        const PhotonMap& photon_map = integrator.getPhotonMapGlobal();

        if (photon_map.nPhotons() > 0)
                for (int i = 0; i < height; ++i) {
#pragma omp parallel for
                        for (int j = 0; j < width; ++j) {
                                const float u = (2.0f * j - width) / height;
                                const float v = (2.0f * i - height) / height;
                                Ray ray;
                                float pdf;
                                if (camera.sampleRay(
                                      Vec2f(v, u), ray, pdf, scene)) {
                                        IntersectInfo info;
                                        if (scene.intersect(ray, info)) {
                                                // query photon map
                                                float r2;
                                                const std::vector<int>
                                                  photon_indices =
                                                    photon_map
                                                      .queryKNearestPhotons(
                                                        info.surfaceInfo
                                                          .position,
                                                        1,
                                                        r2);
                                                const int photon_idx =
                                                  photon_indices[0];

                                                // if distance to the photon is
                                                // small enough, write photon's
                                                // throughput to the image
                                                if (r2 < 1) {
                                                        const Photon& photon =
                                                          photon_map
                                                            .getIthPhoton(
                                                              photon_idx);
                                                        image.setPixel(
                                                          i,
                                                          j,
                                                          photon.throughput);
                                                }
                                        } else {
                                                image.setPixel(i, j, Vec3fZero);
                                        }
                                } else {
                                        image.setPixel(i, j, Vec3fZero);
                                }
                        }
                }

        image.gammaCorrection(2.2f);
        image.writePPM(filename.data());
}

void
visualizeSensorsPhotonMap(const Scene& scene,
                          Image& image,
                          const unsigned& height,
                          const unsigned& width,
                          const Camera& camera,
                          const unsigned& n_photons,
                          const unsigned& max_depth,
                          const std::string_view& filename,
                          UniformSampler& sampler,
                          const PhotonMapping& integrator)
{

        // visualize photon map
        const PhotonMap& photon_map = integrator.getPhotonMapSensors();
        if (photon_map.nPhotons() > 0)
                for (int i = 0; i < height; ++i) {
#pragma omp parallel for
                        for (int j = 0; j < width; ++j) {
                                const float u = (2.0f * j - width) / height;
                                const float v = (2.0f * i - height) / height;
                                Ray ray;
                                float pdf;

                                if (camera.sampleRay(
                                      Vec2f(v, u), ray, pdf, scene)) {
                                        IntersectInfo info;
                                        if (scene.intersect(ray, info)) {
                                                // query photon map
                                                float r2;
                                                const std::vector<int>
                                                  photon_indices =
                                                    photon_map
                                                      .queryKNearestPhotons(
                                                        info.surfaceInfo
                                                          .position,
                                                        1,
                                                        r2);
                                                const int photon_idx =
                                                  photon_indices[0];

                                                // if distance to the photon is
                                                // small enough, write photon's
                                                // throughput to the image
                                                if (r2 < 1) {
                                                        const Photon& photon =
                                                          photon_map
                                                            .getIthPhoton(
                                                              photon_idx);
                                                        image.setPixel(
                                                          i,
                                                          j,
                                                          photon.throughput);
                                                }
                                        } else {
                                                image.setPixel(i, j, Vec3fZero);
                                        }
                                } else {
                                        image.setPixel(i, j, Vec3fZero);
                                }
                        }
                }

        image.gammaCorrection(2.2f);
        image.writePPM(filename.data());
}

void
Render(UniformSampler& sampler,
       Image& image,
       const unsigned& height,
       unsigned& width,
       unsigned& n_samples,
       Camera& camera,
       PhotonMapping& integrator,
       const Scene& scene,
       std::string_view& filename)
{
        if (integrator.getPhotonMapGlobal().nPhotons() <= 0)
                return;
        // #pragma omp parallel for collapse(2) schedule(dynamic, 1)
        for (unsigned int i = 0; i < height; ++i) {

                std::cout << "\033[A\33[2K\r";
                std::cout << "rendering scanline " << i + 1 << "/" << height
                          << "..." << std::endl;
                for (int j = 0; j < width; ++j) {
                        // init sampler
                        sampler = UniformSampler(j + width * i);
#pragma omp parallel for
                        for (int k = 0; k < n_samples; ++k) {
                                const float u =
                                  (2.0f * (j + sampler.getNext1D()) - width) /
                                  height;
                                const float v =
                                  (2.0f * (i + sampler.getNext1D()) - height) /
                                  height;

                                Ray ray;
                                float pdf;
                                if (camera.sampleRay(
                                      Vec2f(v, u), ray, pdf, scene)) {
                                        const Vec3f radiance =
                                          integrator.integrate(
                                            ray, scene, sampler) /
                                          pdf;
#ifdef __OUTPUT__
                                        if (std::isnan(radiance[0]) ||
                                            std::isnan(radiance[1]) ||
                                            std::isnan(radiance[2])) {
                                                std::cerr << "radiance is NaN"
                                                          << std::endl;
                                                continue;
                                        } else if (radiance[0] < 0 ||
                                                   radiance[1] < 0 ||
                                                   radiance[2] < 0) {
                                                std::cerr << "radiance is minus"
                                                          << std::endl;
                                                continue;
                                        }
#endif
                                        image.addPixel(i, j, radiance * 1000);
                                } else {
                                        image.setPixel(
                                          i, j, Vec3f(0.4f, 0.7f, 1.0f));
                                }
                        }
                }
        }
        image.divide(n_samples);
        image.writePPM(filename.data());
}


NB_MODULE(libspice, m)
{
        m.doc() = "nanobind module for photon mapping";

        nb::bind_vector<std::vector<float>>(
          m, "VectorFloat");
        nb::bind_vector<std::vector<unsigned int>>(
          m, "VectorUint");
        nb::bind_vector<std::vector<long>>(
          m, "VectorInt");
        nb::bind_vector<std::vector<unsigned char>>(
          m, "VectorUchar");
        nb::bind_vector<std::vector<Triangle>>(
          m, "VectorTriangle");

        nb::class_<Material>(m, "Material")
          .def(nb::init<>())
          .def_rw("diffuse", &Material::diffuse)
          .def_rw("specular", &Material::specular)
          .def_rw("ambient", &Material::ambient)
          .def_rw("transparency", &Material::transparency)
          .def_rw("illum", &Material::illum)
          .def_rw("shininess", &Material::shininess)
          .def_rw("roughness", &Material::roughness)
          .def_rw("ior", &Material::ior);

        // Photonmap
        nb::class_<Photon>(m, "Photon")
          .def(nb::init<>())
          .def(nb::init<Vec3<float>&, Vec3<float>&, Vec3<float>&, unsigned>())
          .def(nb::init<Vec3<float>&, Vec3<float>&, Vec3<float>&, unsigned, unsigned>())
          .def_rw("throughput", &Photon::throughput)
          .def_rw("position", &Photon::position)
          .def_ro("triId", &Photon::triId)
          .def_ro("lightId", &Photon::lightId)
          .def_rw("wi", &Photon::wi);

        nb::class_<PhotonMap>(m, "PhotonMap")
          .def(nb::init<>())
          .def("getIthPhoton",
               &PhotonMap::getIthPhoton,
               "Returns the ith photon of the photon map",
               nb::arg("i"),
               nb::rv_policy::reference)
          .def("setPhotons",
               &PhotonMap::setPhotons,
               "Sets the photons of the photon map",
               nb::arg("photons"))
          .def("nPhotons",
               &PhotonMap::nPhotons,
               "Returns the size of the photonmap")
          .def("build", &PhotonMap::build, "Builds the photon map")
          .def("queryKNearestPhotons",
               &PhotonMap::queryKNearestPhotons,
               "Returns the k nearest photons of the photon p",
               nb::arg("p"),
               nb::arg("k"),
               nb::arg("max_dist2"));

        // Image
        nb::class_<Image>(m, "Image")
          .def(nb::init<unsigned int, unsigned int>())
          .def("addPixel",
               &Image::addPixel,
               "adds the rgb value to the pixel of coord i j",
               nb::arg("i"),
               nb::arg("j"),
               nb::arg("rgb"))
          .def("getPixel", &Image::getPixel, nb::arg("i"), nb::arg("j"))
          .def("setPixel",
               &Image::setPixel,
               nb::arg("i"),
               nb::arg("j"),
               nb::arg("rgb"))
          .def("divide",
               &Image::divide,
               "Divide all the pixel of the image by k",
               nb::arg("k"))
          .def("gammaCorrection", &Image::gammaCorrection, nb::arg("gamma"))
          .def("writePPM",
               &Image::writePPM,
               "Write the image to the filename (.ppm file)",
               nb::arg("filename"))
          .def("clear", &Image::clear, "Sets the pixels of the image to zero");

        // Camera
        nb::class_<Camera>(m, "Camera")
          .def(nb::init<Vec3<float>,
                        Vec3<float>,
                        Vec3<float>,
                        float,
                        float,
                        float,
                        float>());

        // Integrator
        // PhotonMapping class
        nb::class_<PhotonMapping>(m, "PhotonMapping")
          .def(nb::init<unsigned long long, int, int, int, int>())
          .def(nb::init<unsigned long long, int, float, int, int, int>())
          .def("build",
               &PhotonMapping::build,
               nb::arg("scene"),
               nb::arg("sampler"),
               nb::arg("forRendering") = true)
          .def("integrate",
               &PhotonMapping::integrate,
               nb::arg("ray_in"),
               nb::arg("scene"),
               nb::arg("sampler"))
          .def("getPhotonMap",
               &PhotonMapping::getPhotonMapGlobal,
               "Returns the photon map",
               nb::rv_policy::reference)
          .def("getPhotonMapCaustics",
               &PhotonMapping::getPhotonMapCaustics,
               "Returns the caustics photon map",
               nb::rv_policy::reference)
          .def("getPhotonMapSensors",
               &PhotonMapping::getPhotonMapSensors,
               "Returns the sensor photon map",
               nb::rv_policy::reference);

        // Lights
        nb::enum_<LightType>(m, "LightType")
          .value("Area", Area)
          .value("PointL", PointL)
          .export_values();

        // Area light
        nb::class_<AreaLight>(m, "AreaLight")
          .def(nb::init<Vec3f, Triangle*>())
          .def("Le", &AreaLight::Le)
          .def("samplePoint", &AreaLight::samplePoint)
          .def("sampleDirection", &AreaLight::sampleDirection);

        // Spot light
        nb::class_<SpotLight>(m, "SpotLight")
          .def(nb::init<Vec3f, Vec3f, Vec3f, float>())
          .def("Le", &SpotLight::Le)
          .def("samplePoint", &SpotLight::samplePoint)
          .def("sampleDirection", &SpotLight::sampleDirection);

        // Spot light
        nb::class_<PointLight>(m, "PointLight")
          .def(nb::init<Vec3f, Vec3f>())
          .def("Le", &PointLight::Le)
          .def("samplePoint", &PointLight::samplePoint)
          .def("sampleDirection", &PointLight::sampleDirection);

        nb::class_<Primitive>(m, "Primitive")
          .def(nb::init<Triangle*,
                        std::shared_ptr<BxDF>&,
                        std::string&,
                        const std::shared_ptr<Light>&>())
          .def("hasAreaLight", &Primitive::hasAreaLight)
          .def("Le", &Primitive::Le, nb::arg("surfInfo"), nb::arg("dir"))
          .def("getBxDFType", &Primitive::getBxDFType)
          .def("evaluateBxDF",
               &Primitive::evaluateBxDF,
               nb::arg("wo"),
               nb::arg("wi"),
               nb::arg("surfInfo"),
               nb::arg("mode"))
          .def("sampleBxDF",
               &Primitive::sampleBxDF,
               nb::arg("wo"),
               nb::arg("surfInfo"),
               nb::arg("mode"),
               nb::arg("sampler"),
               nb::arg("wi"),
               nb::arg("pdf"))
          .def("sampleAllBxDF",
               &Primitive::sampleAllBxDF,
               nb::arg("wo"),
               nb::arg("surfInfo"),
               nb::arg("mode"));

        // Sampler
        nb::class_<Sampler>(m, "Sampler");

        nb::class_<UniformSampler, Sampler>(m, "UniformSampler")
          .def(nb::init<>())
          .def(nb::init<uint64_t>())
          .def("clone", &UniformSampler::clone, "Function to clone the sampler")
          .def("getNext1D",
               &UniformSampler::getNext1D,
               nb::rv_policy::reference)
          .def("getNext2D",
               &UniformSampler::getNext2D,
               nb::rv_policy::reference);

        m.def("sampleCosineHemisphere",
              &sampleCosineHemisphere,
              nb::arg("uv"),
              nb::arg("pdf"));

        // Core
        // Constants
        m.attr("PI") = nb::float_(PI);
        m.attr("PI_MUL_2") = nb::float_(PI_MUL_2);
        m.attr("PI_MUL_4") = nb::float_(PI_MUL_4);
        m.attr("PI_INV") = nb::float_(PI_INV);
        m.attr("RAY_EPS") = nb::float_(RAY_EPS);

        // Vec2
        nb::class_<Vec2<float>>(m, "Vec2", nb::dynamic_attr())
          .def(nb::init<>())
          .def(nb::init<float>())
          .def(nb::init<float, float>())
          .def(nb::self + nb::self)
          .def(nb::self + float())
          .def(nb::self += nb::self)
          .def(nb::self *= float())
          .def(float() * nb::self)
          .def(nb::self * float())
          .def(-nb::self)
          .def(
            "__setitem__",
            [](Vec2<float>& self, int index, float val) { self[index] = val; })
          .def("__getitem__",
               [](Vec2<float>& self, int index) { return self[index]; });

        // Vec3
        nb::class_<Vec3<float>>(m, "Vec3", nb::dynamic_attr())
          .def(nb::init<>())
          .def(nb::init<float>())
          .def(nb::init<float, float, float>())
          .def(nb::self + nb::self)
          .def(nb::self + float())
          .def(nb::self += nb::self)
          .def(nb::self - nb::self)
          .def(nb::self *= float())
          .def(float() * nb::self)
          .def(nb::self * float())
          .def(nb::self / float())
          .def(float() / nb::self)
          .def(-nb::self)
          .def(
            "__setitem__",
            [](Vec3<float>& self, int index, float val) { self[index] = val; })
          .def("__getitem__",
               [](Vec3<float>& self, int index) { return self[index]; });

        // Vec3
        nb::class_<Vec3<uint32_t>>(m, "Vec3Ui")
          .def(nb::init<>())
          .def(nb::init<uint32_t>())
          .def(nb::init<uint32_t, uint32_t, uint32_t>())
          .def(nb::self + nb::self)
          .def(nb::self += nb::self)
          .def(nb::self *= float())
          .def(float() * nb::self)
          .def(nb::self * float())
          .def("__setitem__",
               [](Vec3<uint32_t>& self, int index, uint32_t val) {
                       self[index] = val;
               })
          .def("__getitem__",
               [](Vec3<uint32_t>& self, int index) { return self[index]; });

        nb::class_<Ray>(m, "Ray")
          .def(nb::init<>())
          .def(nb::init<Vec3<float>, Vec3<float>>())
          .def_rw("direction", &Ray::direction);

        // SurfaceInfo
        nb::class_<SurfaceInfo>(m, "SurfaceInfo")
          .def_rw("position", &SurfaceInfo::position)
          .def_rw("geometricNormal", &SurfaceInfo::geometricNormal)
          .def_rw("shadingNormal", &SurfaceInfo::shadingNormal)
          .def_rw("dpdu", &SurfaceInfo::dpdu)
          .def_rw("dpdv", &SurfaceInfo::dpdv)
          .def_rw("texcoords", &SurfaceInfo::texcoords)
          .def_rw("barycentric", &SurfaceInfo::barycentric);

        // Scene
        nb::class_<Scene>(m, "Scene")
          .def(nb::init<>())
          .def_rw("triangles", &Scene::triangles)
          .def_rw("vertices", &Scene::vertices)
          .def_rw("indices", &Scene::indices)
          .def_rw("primitives", &Scene::primitives)
          .def_rw("normals", &Scene::normals)
          .def_rw("tnear", &Scene::tnear)
          .def(
            "loadModel",
            &Scene::loadModel,
            "Function to load a model in the scene, must be an .obj file path",
            nb::arg("filepath"))
          .def(
            "loadModelGLTF",
            &Scene::loadModelGLTF,
            "Function to load a model in the scene, must be an .gltf/.glb file path",
            nb::arg("filepath"))
          .def("setupTriangles", &Scene::setupTriangles)
          .def("addFaceInfos",
               &Scene::addFaceInfos,
               nb::arg("vertices"),
               nb::arg("indices"),
               nb::arg("normals"),
               nb::arg("colors"),
               nb::arg("ambient"),
               nb::arg("specular"),
               nb::arg("shininess"),
               nb::arg("transparency"),
               nb::arg("illum"),
               nb::arg("mat_name"),
               nb::arg("ior"),
               nb::arg("reflectance"),
               nb::arg("transmittance"),
               nb::arg("roughness"))
          .def("addLight",
               &Scene::addLight,
               nb::arg("vertices"),
               nb::arg("indices"),
               nb::arg("normals"),
               nb::arg("intensity"),
               nb::arg("color"),
               nb::arg("mat_name"))
          .def("addFaceSensorInfos",
               &Scene::addFaceSensorInfos,
               nb::arg("vertices"),
               nb::arg("indices"),
               nb::arg("normals"),
               nb::arg("mat_name"),
               nb::arg("reflectance"),
               nb::arg("specular"),
               nb::arg("transmittance"),
               nb::arg("roughness"))
          .def("addVirtualSensorInfos",
               &Scene::addVirtualSensorInfos,
               nb::arg("vertices"),
               nb::arg("indices"),
               nb::arg("normals"))
          .def("addPointLight",
               &Scene::addPointLight,
               nb::arg("position"),
               nb::arg("intensity"),
               nb::arg("color"))
          .def("addSpotLight",
               &Scene::addSpotLight,
               nb::arg("position"),
               nb::arg("intensity"),
               nb::arg("color"),
               nb::arg("direction"),
               nb::arg("angle"))
          .def("setMatPrimitive",
               &Scene::setMatPrimitive,
               nb::arg("primName"),
               nb::arg("reflectance"),
               nb::arg("transmittance"),
               nb::arg("specularity") = 0.0)
          .def("build", &Scene::build, nb::arg("back_face_culling") = true)
          .def("getTriangles",
               &Scene::getTriangles,
               "Returns an array with the triangles of the scene")
          .def("intersect", &Scene::intersect, nb::arg("ray"), nb::arg("info"))
          .def("sampleLight",
               nb::overload_cast<Sampler&, float&>(&Scene::sampleLight,
                                                   nb::const_),
               nb::arg("sampler"),
               nb::arg("pdf"))
          .def("sampleLight",
               nb::overload_cast<float&, unsigned int>(&Scene::sampleLight,
                                                       nb::const_),
               nb::arg("pdf"),
               nb::arg("idx"))
          .def("nVertices",
               &Scene::nVertices,
               "Function that returns the number of vertices in the scene")
          .def("nFaces",
               &Scene::nFaces,
               "Function that returns the number of faces in the scene")
          .def("nLights",
               &Scene::nLights,
               "Returns the number of light sources in the scene")
          .def("clear", &Scene::clear, "Clears the elements of the scene");

        // IntersectInfo
        nb::class_<IntersectInfo>(m, "IntersectInfo")
          .def(nb::init<>())
          .def_ro("t", &IntersectInfo::t)
          .def_ro("surfaceInfo", &IntersectInfo::surfaceInfo);

        m.def("Render",
              &Render,
              "Function to render the scene to an image from the camera "
              "perspective",
              nb::arg("sampler"),
              nb::arg("image"),
              nb::arg("height"),
              nb::arg("width"),
              nb::arg("n_samples"),
              nb::arg("camera"),
              nb::arg("integrator"),
              nb::arg("scene"),
              nb::arg("filename"));

        m.def("visualizePhotonMap",
              &visualizePhotonMap,
              "Function to visualize the photonmap as a .ppm image",
              nb::arg("integrator"),
              nb::arg("Scene"),
              nb::arg("image"),
              nb::arg("height"),
              nb::arg("width"),
              nb::arg("camera"),
              nb::arg("n_photons"),
              nb::arg("max_depth"),
              nb::arg("filename"),
              nb::arg("sampler"));

        m.def("visualizeSensorsPhotonMap",
              &visualizeSensorsPhotonMap,
              "Function to visualize the photonmap as a .ppm image",
              nb::arg("Scene"),
              nb::arg("image"),
              nb::arg("height"),
              nb::arg("width"),
              nb::arg("camera"),
              nb::arg("n_photons"),
              nb::arg("max_depth"),
              nb::arg("filename"),
              nb::arg("sampler"),
              nb::arg("integrator"));

        m.def("normalize", &normalize, "Returns the normal of a vector");
}
