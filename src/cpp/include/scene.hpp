#ifndef SCENE_H
#define SCENE_H

#include <embree4/rtcore.h>
#include <filesystem>
#include <optional>
#include <string>
#include <omp.h>

#include <memory>
#include <vector>
#include <cassert>

#include <tiny_obj_loader.h>
#include <tiny_gltf.h>
#include "core.hpp"
#include "primitive.hpp"

void
intersectionFilter(const RTCFilterFunctionNArguments* args)
{
        if (args->context == nullptr)
                return;

        assert(args->N == 1);
        int* valid = args->valid;

        if (valid[0] != -1)
                return;

        RTCRay* ray = (RTCRay*)args->ray;
        RTCHit* hit = (RTCHit*)args->hit;

        // back culling
        Vec3f normal(hit->Ng_x, hit->Ng_y, hit->Ng_z);
        Vec3f direction(ray->dir_x, ray->dir_y, ray->dir_z);
        float denominateur = dot(normalize(direction), normalize(normal));

        // test front face
        if (denominateur >= 0.00001) {
                valid[0] = 0;
        }

        // test bound UV
        if (hit->u < 0 || hit->u > 1) {
                valid[0] = 0;
        }

        if (hit->v < 0 || (hit->v + hit->u) > 1) {
                valid[0] = 0;
        }
}

/**
 * @brief Creates a default Lambert BxDF.
 * @return a pointer towards this bxdf.
 */
std::shared_ptr<BxDF>
createDefaultBxDF()
{
        return std::make_shared<Lambert>(Vec3f(0.9f));
}

/**
 * @brief Creates a BxDF with material data.
 * @param material The obj material data.
 * @param reflectance A reflectance.
 * @param transmittance A transmittance.
 * @param roughness The roughness info for light scattering.
 * @return a pointer towards this bxdf.
 */
std::shared_ptr<BxDF>
createBxDF(tinyobj::material_t& material,
           float reflectance = 0.0f,
           float transmittance = 0.0f,
           float roughness = 0.5f)
{

        // calculate diffuse and specular with reflectance
        const Vec3f kd = Vec3f(material.diffuse[0],
                               material.diffuse[1],
                               material.diffuse[2]);
        const Vec3f ks = Vec3f(material.specular[0],
                               material.specular[1],
                               material.specular[2]) *
                         reflectance;

        if (material.illum == 2 && (ks == Vec3fZero)) {
                material.illum = 1;
        }

        switch (material.illum) {
                case 2:
                        return std::make_shared<Mirror>(ks);
                case 3:
                        return std::make_shared<Lambert>(kd);
                case 5:
                        // mirror
                        return std::make_shared<Mirror>(Vec3f(1.0f));
                case 6:
                        // leaf
                        material.ior = 1.425f; // Source:
                        // https://opg.optica.org/ao/abstract.cfm?uri=ao-13-1-109
                        return std::make_shared<Leaf>(
                          kd, material.ior, roughness);
                case 7:
                        // Transparent
                        return std::make_shared<Transparent>(kd, material.ior);
                case 8:
                        // Phong for plant
                        return std::make_shared<PhongSensor>(
                          kd, ks, roughness, transmittance);

                case 9:
                        return std::make_shared<Refltr>(
                          kd, reflectance, transmittance, roughness);
                default:
                        return std::make_shared<Phong>(
                          kd, ks, roughness, transmittance);
        }
}

/**
 * @brief create AreaLight from tinyobj material
 * @param material The obj material data.
 * @param tri The triangles
 * @return A pointer towards the area light.
 */
std::shared_ptr<AreaLight>
createAreaLight(tinyobj::material_t& material, Triangle* tri)
{
        if (material.emission[0] > 0 || material.emission[1] > 0 ||
            material.emission[2] > 0) {
                Vec3f le = Vec3f(material.emission[0],
                                 material.emission[1],
                                 material.emission[2]);
                return std::make_shared<AreaLight>(le, tri);
        } else {
                return nullptr;
        }
}

/**
 * @brief Creates a point light at the specified position
 * @param emission The emission power of the point light.
 * @param position The position of the point light.
 * @return A pointer towards the point light.
 */
std::shared_ptr<PointLight>
createPointLight(Vec3f emission, Vec3f position)
{
        if (emission[0] > 0 || emission[1] > 0 || emission[2] > 0) {
                Vec3f le = Vec3f(emission[0], emission[1], emission[2]);
                return std::make_shared<PointLight>(le, position);
        } else {
                return nullptr;
        }
}

/**
 * @brief Creates a spot light at the specified position
 * @param emission The emission power of the spot light.
 * @param position The position of the spot light.
 * @param direction The direction of emission of the spot light.
 * @param angle The angle of diffusion of the spot light.
 * @return A pointer towards the spot light.
 */
std::shared_ptr<SpotLight>
createSpotLight(Vec3f emission, Vec3f position, Vec3f direction, float angle)
{
        if (emission[0] > 0 || emission[1] > 0 || emission[2] > 0) {
                Vec3f le = Vec3f(emission[0], emission[1], emission[2]);
                return std::make_shared<SpotLight>(
                  le, position, direction, angle);
        } else {
                return nullptr;
        }
}

std::shared_ptr<TubeLight>
createTubeLight(Vec3f emission, Triangle* tri, Vec3f direction, float angle)
{
        if (emission[0] > 0 || emission[1] > 0 || emission[2] > 0) {
                Vec3f le = Vec3f(emission[0], emission[1], emission[2]);
                return std::make_shared<TubeLight>(le, tri, direction, angle);
        } else {
                return nullptr;
        }
}

/**
 * Class representing a 3D scene.
 * @class Scene
 */
class Scene
{
      private:
        // embree
        RTCDevice device{}; ///< The embree device to use
        RTCScene scene{};   ///< The embree scene

        /**
         * @brief Whether a face is light source or not
         * @param faceID the id of the face
         * @return true if the face emits light else false.
         */
        [[nodiscard]] bool hasLight(uint32_t faceID) const
        {
                return lights[faceID] != nullptr;
        }

      public:
        // mesh data
        // NOTE: assuming size of normals, texcoords == size of vertices
        std::vector<float> vertices;   ///< The vertices of the scene.
        std::vector<uint32_t> indices; ///< The indices of the scene.
        std::vector<float> normals;    ///< The normals of the scene.

        std::vector<std::optional<tinyobj::material_t>>
          materials; ///< The materials of the scene.

        std::vector<Triangle>
          triangles; ///< The triangles of the scene per face.

        std::vector<std::shared_ptr<BxDF>>
          bxdfs; ///< The bxdfs of the scene per face.

        std::vector<std::shared_ptr<Light>>
          lights; ///< The lights of the scene per face.

        std::vector<Primitive>
          primitives; ///< The primitives of the scene per face.

        float tnear = 0.1;
        /**
         * @brief Constructor
         */
        Scene() = default;

        /**
         * @brief Destructor
         */
        ~Scene()
        {
                clear();
                rtcReleaseScene(scene);
                rtcReleaseDevice(device);
        }

        /**
         * @fn void clear()
         * @brief clears the objects in the scene.
         */
        void clear()
        {
                vertices.clear();
                indices.clear();
                normals.clear();

                triangles.clear();
                bxdfs.clear();
                lights.clear();
                primitives.clear();
        }

        /**
         * @fn void addFaceInfos(std::vector<float> &vertices,
                          std::vector<uint32_t> &indices, std::vector<float>
         &normals, Vec3f &colors, Vec3f &ambient, Vec3f &specular, float
         &shininess, float &transparency, int &illum, float ior = 0.0f, float
         reflectance = 0.0f, float transmittance = 0.0f, float roughness = 0.0f)
         * @brief Adds a face to the scene with the provided info.
         * @param vertices
         * @param indices
         * @param normals
         * @param colors
         * @param ambient
         * @param specular
         * @param shininess
         * @param transparency
         * @param illum
         * @param ior
         * @param reflectance
         * @param transmittance
         * @param roughness
         */
        void addFaceInfos(std::vector<float>& newVertices,
                          std::vector<uint32_t>& newIndices,
                          std::vector<float>& newNormals,
                          Vec3f& colors,
                          Vec3f& ambient,
                          float& specular,
                          float& shininess,
                          float& transparency,
                          int& illum,
                          std::string& mat_name,
                          float ior = 0.0f,
                          float reflectance = 0.0f,
                          float transmittance = 0.0f,
                          float roughness = 0.0f)
        {
                omp_set_num_threads(omp_get_max_threads() / 2);
//#pragma omp parallel for
                for (uint32_t& i : newIndices) {
                        i += nVertices();
                }

                this->vertices.insert(std::end(this->vertices),
                                      std::begin(newVertices),
                                      std::end(newVertices));
                this->indices.insert(std::end(this->indices),
                                     std::begin(newIndices),
                                     std::end(newIndices));
                this->normals.insert(std::end(this->normals),
                                     std::begin(newNormals),
                                     std::end(newNormals));

                // populate materials
                for (size_t faceID = nFaces() - (newIndices.size() / 3);
                     faceID < nFaces();
                     ++faceID) {
                        tinyobj::material_t m;

                        m.name = mat_name;
                        m.diffuse[0] = colors[0];
                        m.diffuse[1] = colors[1];
                        m.diffuse[2] = colors[2];
                        m.ambient[0] = ambient[0];
                        m.ambient[1] = ambient[1];
                        m.ambient[2] = ambient[2];
                        m.emission[0] = 0.00;
                        m.emission[1] = 0.00;
                        m.emission[2] = 0.00;
                        m.specular[0] = specular;
                        m.specular[1] = specular;
                        m.specular[2] = specular;
                        m.shininess = shininess;
                        m.dissolve = 1 - transparency;
                        m.ior = ior;
                        if (transparency > 0)
                                m.illum = 1;
                        else
                                m.illum = illum;
                        this->materials.emplace_back(m);

                        // populate BxDF
                        const auto material = this->materials[faceID];
                        if (material) {
                                tinyobj::material_t m = material.value();
                                this->bxdfs.push_back(createBxDF(
                                  m, reflectance, transmittance, roughness));
                        }
                        // default material
                        else {
                                this->bxdfs.push_back(createDefaultBxDF());
                        }
                }
        }

        /**
         * @fn void setupTriangles()
         * @brief Setup the triangles of the scene with the data provided thus
         * far. Must be used after adding all the faces of the desired scene.
         * @warning Should not be used after loading an obj file.
         */
        void setupTriangles()
        {
                // populate  triangles
                for (size_t faceID = 0; faceID < nFaces(); ++faceID) {
                        // add triangle
                        this->triangles.emplace_back(this->vertices.data(),
                                                     this->indices.data(),
                                                     this->normals.data(),
                                                     faceID);
                }

                // populate lights, primitives
                for (size_t faceID = 0; faceID < nFaces(); ++faceID) {
                        // add light
                        std::shared_ptr<Light> light = nullptr;
                        const auto material = this->materials[faceID];
                        // std::cout << "material check" << std::endl;
                        std::string sh_name = "";
                        if (material) {

                                tinyobj::material_t m = material.value();
                                sh_name = m.name;
                                light =
                                  createAreaLight(m, &this->triangles[faceID]);
                                if (light != nullptr) {
                                        lights.push_back(light);
                                }
                        }

                        // add primitive
                        // std::cout << "Adding primitives" << std::endl;
                        primitives.emplace_back(&this->triangles[faceID],
                                                this->bxdfs[faceID],
                                                sh_name,
                                                light);
                }
                // std::cout << "Triangles setup! " << std::endl;

#ifdef __OUTPUT__
                std::cout << "[Scene] vertices: " << nVertices() << std::endl;
                std::cout << "[Scene] faces: " << nFaces() << std::endl;
                std::cout << "[Scene] lights: " << lights.size() << std::endl;
#endif
        }

        /**
         * 
         */
        void setMatPrimitive(std::string& primName, float reflectance, float transmittance, float specularity = 0.0)
        {
                for (Primitive& prim: primitives)
                {
                        if (prim.name == primName)
                        {
                                tinyobj::material_t m ;
                                m.specular[0] = specularity;
                                m.specular[1] = specularity;
                                m.specular[2] = specularity;
                                m.illum = 1;
                                prim.bxdf = createBxDF(m, reflectance, transmittance);
                        }
                } 
        }

        /**
         * @fn void addLight(std::vector<float> newVertices,
         * std::vector<uint32_t> newIndices, std::vector<float> newNormals,
         * float intensity, Vec3f color)
         * @brief adds the given mesh as a light source.
         * @param newVertices The vertices of the mesh
         * @param newIndices The indices of the mesh
         * @param newNormals The normals of the mesh
         * @param intensity The intensity of the light emission of the mesh
         * @param color The color of the light.
         */
        void addLight(std::vector<float> newVertices,
                      std::vector<uint32_t> newIndices,
                      std::vector<float> newNormals,
                      float intensity,
                      Vec3f color,
                      std::string mat_name)
        {
                for (uint32_t& i : newIndices) {
                        i += nVertices();
                }
                this->vertices.insert(std::end(this->vertices),
                                      std::begin(newVertices),
                                      std::end(newVertices));
                this->indices.insert(std::end(this->indices),
                                     std::begin(newIndices),
                                     std::end(newIndices));
                this->normals.insert(std::end(this->normals),
                                     std::begin(newNormals),
                                     std::end(newNormals));

                // populate  triangles
                for (size_t faceID = nFaces() - (newIndices.size() / 3);
                     faceID < nFaces();
                     ++faceID) {
                        tinyobj::material_t m;

                        m.name = mat_name;
                        m.diffuse[0] = color[0];
                        m.diffuse[1] = color[1];
                        m.diffuse[2] = color[2];
                        m.ambient[0] = 0;
                        m.ambient[1] = 0;
                        m.ambient[2] = 0;
                        m.emission[0] = color[0] * (intensity);
                        m.emission[1] = color[1] * (intensity);
                        m.emission[2] = color[2] * (intensity);
                        m.specular[0] = 0.00;
                        m.specular[1] = 0.00;
                        m.specular[2] = 0.00;
                        m.dissolve = 1.0;
                        m.illum = 1;

                        this->materials.emplace_back(m);

                        // populate BxDF
                        const auto material = this->materials[faceID];
                        if (material) {
                                tinyobj::material_t m = material.value();
                                this->bxdfs.push_back(createBxDF(m));
                        }
                        // default material
                        else {
                                this->bxdfs.push_back(createDefaultBxDF());
                        }
                }
        }

        /**
         * @brief Adds the specified mesh as a sensor.
         * @param newVertices The vertices of the mesh
         * @param newIndices The indices of the triangles of the mesh
         * @param newNormals The normals of the mesh.
         */
        void addVirtualSensorInfos(std::vector<float> newVertices,
                                   std::vector<uint32_t> newIndices,
                                   std::vector<float> newNormals)
        {
                for (uint32_t& i : newIndices) {
                        i += nVertices();
                }
                this->vertices.insert(std::end(this->vertices),
                                      std::begin(newVertices),
                                      std::end(newVertices));
                this->indices.insert(std::end(this->indices),
                                     std::begin(newIndices),
                                     std::end(newIndices));
                this->normals.insert(std::end(this->normals),
                                     std::begin(newNormals),
                                     std::end(newNormals));

                for (size_t faceID = nFaces() - (newIndices.size() / 3);
                     faceID < nFaces();
                     ++faceID) {

                        tinyobj::material_t m;

                        m.diffuse[0] = 0;
                        m.diffuse[1] = 0;
                        m.diffuse[2] = 0;
                        m.ambient[0] = 0;
                        m.ambient[1] = 0;
                        m.ambient[2] = 0;
                        m.emission[0] = 0;
                        m.emission[1] = 0;
                        m.emission[2] = 0;
                        m.specular[0] = 0;
                        m.specular[1] = 0;
                        m.specular[2] = 0;
                        m.dissolve = 1.0;
                        m.illum = 1;

                        this->materials.emplace_back(m);
                        this->bxdfs.emplace_back(
                          std::make_shared<Sensor>(Vec3f(1, 0, 1)));
                }
        }

        /**
         * @brief Adds the specified mesh as a sensor.
         * @param newVertices The vertices of the mesh
         * @param newIndices The indices of the triangles of the mesh
         * @param newNormals The normals of the mesh.
         */
        void addFaceSensorInfos(std::vector<float> newVertices,
                                std::vector<uint32_t> newIndices,
                                std::vector<float> newNormals,
                                std::string& mat_name,
                                float reflectance = 0.0f,
                                float specular = 0.0f,
                                float transmittance = 0.0f,
                                float roughness = 0.0f)
        {
                for (uint32_t& i : newIndices) {
                        i += nVertices();
                }
                this->vertices.insert(std::end(this->vertices),
                                      std::begin(newVertices),
                                      std::end(newVertices));
                this->indices.insert(std::end(this->indices),
                                     std::begin(newIndices),
                                     std::end(newIndices));
                this->normals.insert(std::end(this->normals),
                                     std::begin(newNormals),
                                     std::end(newNormals));

                float diffuse_rate = reflectance * (1 - specular);
                float specular_rate = reflectance * specular;

                Vec3f kd = Vec3f(diffuse_rate, diffuse_rate, diffuse_rate);
                Vec3f ks = Vec3f(specular_rate, specular_rate, specular_rate);

                for (size_t faceID = nFaces() - (newIndices.size() / 3);
                     faceID < nFaces();
                     ++faceID) {
                        tinyobj::material_t m;

                        m.diffuse[0] = diffuse_rate;
                        m.diffuse[1] = diffuse_rate;
                        m.diffuse[2] = diffuse_rate;
                        m.ambient[0] = 0;
                        m.ambient[1] = 0;
                        m.ambient[2] = 0;
                        m.emission[0] = 0;
                        m.emission[1] = 0;
                        m.emission[2] = 0;
                        m.specular[0] = specular_rate;
                        m.specular[1] = specular_rate;
                        m.specular[2] = specular_rate;
                        m.dissolve = 1.0;
                        m.illum = 1;

                        this->materials.emplace_back(m);
                        this->bxdfs.emplace_back(std::make_shared<PhongSensor>(
                          kd, ks, roughness, transmittance));
                }
        }

        /**
         * @brief Put a point light at the specified position.
         * @param position The 3d position.
         * @param intensity The intensity of the light
         * @param color The color of the light
         */
        void addPointLight(Vec3f position, float intensity, Vec3f color)
        {
                std::shared_ptr<Light> light;
                light = createPointLight(color * intensity, position);
                if (light != nullptr) {
                        lights.push_back(light);
                }
        }

        /**
         * @brief Put a spot light at the specified position.
         * @param position The 3d position.
         * @param intensity The intensity of the light
         * @param color The color of the light
         * @param direction The direction of the spot light
         * @param angle The angle of diffusion of the spot light
         */
        void addSpotLight(Vec3f position,
                          float intensity,
                          Vec3f color,
                          Vec3f direction,
                          float angle)
        {
                std::shared_ptr<Light> light;
                light = createSpotLight(
                  color * intensity, position, direction, angle);
                if (light != nullptr) {
                        lights.push_back(light);
                }
        }

        void addTubeLight(Triangle* tri,
                          float intensity,
                          Vec3f color,
                          Vec3f direction,
                          float angle)
        {
                std::shared_ptr<Light> light;
                light =
                  createTubeLight(color * intensity, tri, direction, angle);
                if (light != nullptr) {
                        lights.push_back(light);
                }
        }

        // TODO: remove vertex duplication
        /**
         * @fn void loadModel(const std::string &filename)
         * @brief Loads an obj file as the scene.
         *
         * @param filename the filename of the .obj scene.
         */
        void loadModel(const std::string& filename)
        {
                std::filesystem::path filepath(filename);
                clear();
                std::string mtl_folder;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
                const size_t last_slash_idx = filename.rfind('\\');
#else
                const size_t last_slash_idx = filename.rfind('/');
#endif
                if (std::string::npos != last_slash_idx) {
                        mtl_folder = filename.substr(0, last_slash_idx);
                }

                tinyobj::ObjReaderConfig reader_config;
                reader_config.mtl_search_path = mtl_folder;
                reader_config.triangulate = true;

                tinyobj::ObjReader reader;
                if (!reader.ParseFromFile(filepath.generic_string(),
                                          reader_config)) {
                        if (!reader.Error().empty()) {
                                std::cout << "[Scene] failed to load "
                                          << filepath.generic_string() << ": "
                                          << reader.Error();
                        }
                        return;
                }

                if (!reader.Warning().empty()) {
                        std::cout << "[Scene] " << reader.Warning()
                                  << std::endl;
                }

                const auto& attrib = reader.GetAttrib();
                const auto& shapes = reader.GetShapes();
                const auto& materials = reader.GetMaterials();

                // loop over shapes
                for (const auto& shape : shapes) {
                        size_t index_offset = 0;
                        // loop over faces
                        for (size_t f = 0;
                             f < shape.mesh.num_face_vertices.size();
                             ++f) {
                                const size_t fv = static_cast<size_t>(
                                  shape.mesh.num_face_vertices[f]);

                                std::vector<Vec3f> vertices;
                                std::vector<Vec3f> normals;

                                // loop over vertices
                                // get vertices, normals, texcoords of a
                                // triangle
                                for (size_t v = 0; v < fv; ++v) {
                                        const tinyobj::index_t idx =
                                          shape.mesh.indices[index_offset + v];

                                        const tinyobj::real_t vx =
                                          attrib
                                            .vertices[3 * static_cast<size_t>(
                                                            idx.vertex_index) +
                                                      0];
                                        const tinyobj::real_t vy =
                                          attrib
                                            .vertices[3 * static_cast<size_t>(
                                                            idx.vertex_index) +
                                                      1];
                                        const tinyobj::real_t vz =
                                          attrib
                                            .vertices[3 * static_cast<size_t>(
                                                            idx.vertex_index) +
                                                      2];
                                        vertices.push_back(Vec3f(vx, vy, vz));

                                        if (idx.normal_index >= 0) {
                                                const tinyobj::real_t nx =
                                                  attrib.normals
                                                    [3 * static_cast<size_t>(
                                                           idx.normal_index) +
                                                     0];
                                                const tinyobj::real_t ny =
                                                  attrib.normals
                                                    [3 * static_cast<size_t>(
                                                           idx.normal_index) +
                                                     1];
                                                const tinyobj::real_t nz =
                                                  attrib.normals
                                                    [3 * static_cast<size_t>(
                                                           idx.normal_index) +
                                                     2];
                                                normals.push_back(
                                                  Vec3f(nx, ny, nz));
                                        }
                                }

                                // if normals is empty, add geometric normal
                                if (normals.empty()) {
                                        const Vec3f v1 =
                                          normalize(vertices[1] - vertices[0]);
                                        const Vec3f v2 =
                                          normalize(vertices[2] - vertices[0]);
                                        const Vec3f n =
                                          normalize(cross(v1, v2));
                                        normals.push_back(n);
                                        normals.push_back(n);
                                        normals.push_back(n);
                                }

                                // populate vertices, indices, normals,
                                // texcoords
                                for (int i = 0; i < 3; ++i) {
                                        this->vertices.push_back(
                                          vertices[i][0]);
                                        this->vertices.push_back(
                                          vertices[i][1]);
                                        this->vertices.push_back(
                                          vertices[i][2]);

                                        this->normals.push_back(normals[i][0]);
                                        this->normals.push_back(normals[i][1]);
                                        this->normals.push_back(normals[i][2]);

                                        this->indices.push_back(
                                          this->indices.size());
                                }

                                // populate materials
                                const int materialID =
                                  shape.mesh.material_ids[f];
                                std::optional<tinyobj::material_t> material =
                                  std::nullopt;
                                if (materialID != -1) {
                                        material = materials[materialID];
                                }
                                this->materials.push_back(material);

                                index_offset += fv;
                        }
                }

                // populate  triangles
                for (size_t faceID = 0; faceID < nFaces(); ++faceID) {
                        // add triangle
                        this->triangles.emplace_back(vertices.data(),
                                                     indices.data(),
                                                     normals.data(),
                                                     faceID);
                }

                // populate bxdfs
                for (size_t faceID = 0; faceID < nFaces(); ++faceID) {
                        // add bxdf
                        // TODO: remove duplicate
                        const auto material = this->materials[faceID];
                        if (material) {
                                tinyobj::material_t m = material.value();
                                if (m.dissolve != 0.0f)
                                        this->bxdfs.push_back(createBxDF(
                                          m, m.dissolve, 1 - m.dissolve, 0.5f));
                                else
                                        this->bxdfs.push_back(createBxDF(m));
                        }
                        // default material
                        else {
                                this->bxdfs.push_back(createDefaultBxDF());
                        }
                }

                // populate lights, primitives
                for (size_t faceID = 0; faceID < nFaces(); ++faceID) {
                        // add light
                        std::shared_ptr<Light> light = nullptr;
                        const auto material = this->materials[faceID];
                        std::string sh_name = "";
                        if (material) {
                                tinyobj::material_t m = material.value();
                                sh_name = m.name;
                                light =
                                  createAreaLight(m, &this->triangles[faceID]);
                                if (light != nullptr) {
                                        lights.push_back(light);
                                }
                        }

                        // add primitive
                        primitives.emplace_back(&this->triangles[faceID],
                                                this->bxdfs[faceID],
                                                sh_name,
                                                light);
                }
        }

        const unsigned char* getBufferData(const tinygltf::Model& model, int accessorIdx, size_t& byteStride, size_t& count) {
                if (accessorIdx < 0) return nullptr;
                const auto& accessor = model.accessors[accessorIdx];
                const auto& bufferView = model.bufferViews[accessor.bufferView];
                const auto& buffer = model.buffers[bufferView.buffer];
                
                count = accessor.count;
                byteStride = accessor.ByteStride(bufferView);
                return &(buffer.data[accessor.byteOffset + bufferView.byteOffset]);
        }

        void loadModelGLTF(const std::string& filename)
        {
                std::filesystem::path filepath(filename);
                clear();

                tinygltf::Model model;
                tinygltf::TinyGLTF loader;
                std::string err;
                std::string warn;

                bool ret = false;
                if (filepath.extension() == ".glb") {
                        ret = loader.LoadBinaryFromFile(&model, &err, &warn, filepath.string());
                } else {
                        ret = loader.LoadASCIIFromFile(&model, &err, &warn, filepath.string());
                }

                if (!warn.empty()) std::cout << "[Scene] " << warn << std::endl;
                if (!ret) {
                        if (!err.empty()) std::cout << "[Scene] failed to load " << filepath.string() << ": " << err << std::endl;
                        return;
                }

                // map glTF materials to tinyobj::material_t
                std::vector<tinyobj::material_t> convertedMaterials;
                for (const auto& gtMat : model.materials) {
                        tinyobj::material_t m;
                        m.name = gtMat.name;
                        
                        m.diffuse[0] = static_cast<float>(gtMat.pbrMetallicRoughness.baseColorFactor[0]);
                        m.diffuse[1] = static_cast<float>(gtMat.pbrMetallicRoughness.baseColorFactor[1]);
                        m.diffuse[2] = static_cast<float>(gtMat.pbrMetallicRoughness.baseColorFactor[2]);
                        
                        m.dissolve = static_cast<float>(gtMat.pbrMetallicRoughness.baseColorFactor[3]);

                        m.emission[0] = static_cast<float>(gtMat.emissiveFactor[0]);
                        m.emission[1] = static_cast<float>(gtMat.emissiveFactor[1]);
                        m.emission[2] = static_cast<float>(gtMat.emissiveFactor[2]);

                        m.roughness = static_cast<float>(gtMat.pbrMetallicRoughness.roughnessFactor);
                        m.metallic = static_cast<float>(gtMat.pbrMetallicRoughness.metallicFactor);

                        convertedMaterials.push_back(m);
                }

                for (const auto& mesh : model.meshes) {
                        for (const auto& primitive : mesh.primitives) {
                        // only triangles
                        if (primitive.mode != TINYGLTF_MODE_TRIANGLES) continue;

                        // get positions
                        size_t posStride = 0, posCount = 0;
                        auto posIt = primitive.attributes.find("POSITION");
                        if (posIt == primitive.attributes.end()) continue;
                        const unsigned char* posBuffer = getBufferData(model, posIt->second, posStride, posCount);
                        
                        // get normals (optional)
                        size_t normStride = 0, normCount = 0;
                        auto normIt = primitive.attributes.find("NORMAL");
                        const unsigned char* normBuffer = (normIt != primitive.attributes.end()) ? 
                                getBufferData(model, normIt->second, normStride, normCount) : nullptr;

                        // get indices
                        std::vector<uint32_t> primitiveIndices;
                        if (primitive.indices >= 0) {
                                const auto& indexAccessor = model.accessors[primitive.indices];
                                size_t idxStride = 0, idxCount = 0;
                                const unsigned char* idxBuffer = getBufferData(model, primitive.indices, idxStride, idxCount);
                                
                                for (size_t i = 0; i < idxCount; ++i) {
                                uint32_t idx = 0;
                                if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                                        idx = *reinterpret_cast<const uint32_t*>(idxBuffer + i * idxStride);
                                } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                                        idx = *reinterpret_cast<const uint16_t*>(idxBuffer + i * idxStride);
                                } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                                        idx = *reinterpret_cast<const uint8_t*>(idxBuffer + i * idxStride);
                                }
                                primitiveIndices.push_back(idx);
                                }
                        } else {
                                // Unindexed primitives: generate sequential indices
                                for (uint32_t i = 0; i < posCount; ++i) primitiveIndices.push_back(i);
                        }

                        for (size_t f = 0; f < primitiveIndices.size(); f += 3) {
                                std::vector<Vec3f> faceVertices;
                                std::vector<Vec3f> faceNormals;

                                for (size_t v = 0; v < 3; ++v) {
                                uint32_t idx = primitiveIndices[f + v];

                                // read position
                                const float* pPtr = reinterpret_cast<const float*>(posBuffer + idx * posStride);
                                faceVertices.push_back(Vec3f(pPtr[0], pPtr[1], pPtr[2]));

                                // read normal if it exists
                                if (normBuffer) {
                                        const float* nPtr = reinterpret_cast<const float*>(normBuffer + idx * normStride);
                                        faceNormals.push_back(Vec3f(nPtr[0], nPtr[1], nPtr[2]));
                                }
                                }

                                if (faceNormals.empty()) {
                                Vec3f v1 = normalize(faceVertices[1] - faceVertices[0]);
                                Vec3f v2 = normalize(faceVertices[2] - faceVertices[0]);
                                Vec3f n = normalize(cross(v1, v2));
                                faceNormals.push_back(n);
                                faceNormals.push_back(n);
                                faceNormals.push_back(n);
                                }

                                for (int i = 0; i < 3; ++i) {
                                this->vertices.push_back(faceVertices[i][0]);
                                this->vertices.push_back(faceVertices[i][1]);
                                this->vertices.push_back(faceVertices[i][2]);

                                this->normals.push_back(faceNormals[i][0]);
                                this->normals.push_back(faceNormals[i][1]);
                                this->normals.push_back(faceNormals[i][2]);

                                this->indices.push_back(this->indices.size());
                                }

                                // map primitive material
                                std::optional<tinyobj::material_t> material = std::nullopt;
                                if (primitive.material >= 0 && primitive.material < convertedMaterials.size()) {
                                material = convertedMaterials[primitive.material];
                                }
                                this->materials.push_back(material);
                        }
                        }
                }

                // populate  triangles
                for (size_t faceID = 0; faceID < nFaces(); ++faceID) {
                        this->triangles.emplace_back(vertices.data(), indices.data(), normals.data(), faceID);
                }

                // populate bxdfs
                for (size_t faceID = 0; faceID < nFaces(); ++faceID) {
                        const auto material = this->materials[faceID];
                        if (material) {
                        tinyobj::material_t m = material.value();
                        if (m.dissolve != 1.0f && m.dissolve != 0.0f) // tinygltf sets alpha to 1.0 by default
                                this->bxdfs.push_back(createBxDF(m, m.dissolve, 1.0f - m.dissolve, 0.5f));
                        else
                                this->bxdfs.push_back(createBxDF(m));
                        } else {
                        this->bxdfs.push_back(createDefaultBxDF());
                        }
                }

                // populate lights, primitives
                for (size_t faceID = 0; faceID < nFaces(); ++faceID) {
                        std::shared_ptr<Light> light = nullptr;
                        const auto material = this->materials[faceID];
                        std::string sh_name = "";
                        
                        if (material) {
                        tinyobj::material_t m = material.value();
                        sh_name = m.name;
                        light = createAreaLight(m, &this->triangles[faceID]);
                        if (light != nullptr) {
                                lights.push_back(light);
                        }
                        }

                        primitives.emplace_back(&this->triangles[faceID], this->bxdfs[faceID], sh_name, light);
                }
        }

        /**
         * @fn uint32_t nVertices() const
         * @brief Get the number of vertices in the scene
         * @return the number of vertices in the scene
         */
        uint32_t nVertices() const { return vertices.size() / 3; }

        /**
         * @fn uint32_t nFaces() const
         * @brief Get the number of faces in the scene
         * @return the number of faces in the scene
         */
        uint32_t nFaces() const { return indices.size() / 3; }

        std::vector<Triangle> getTriangles() const { return triangles; }

        /**
         * @fn void build()
         * @brief Creates the embree scene with all the previous info.
         */
        void build(bool back_face_culling = true)
        {
#ifdef __OUTPUT__
                std::cout << "[Scene] building scene..." << std::endl;
#endif

                // setup embree
                device = rtcNewDevice("hugepages=1");
                scene = rtcNewScene(device);

                rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_MEDIUM);
                // rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_HIGH);
                rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);

                RTCGeometry geom =
                  rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

                // set vertices
                float* vb =
                  (float*)rtcSetNewGeometryBuffer(geom,
                                                  RTC_BUFFER_TYPE_VERTEX,
                                                  0,
                                                  RTC_FORMAT_FLOAT3,
                                                  3 * sizeof(float),
                                                  nVertices());
                for (size_t i = 0; i < vertices.size(); ++i) {
                        vb[i] = vertices[i];
                }

                // set indices
                uint32_t* ib =
                  static_cast<uint32_t *>(rtcSetNewGeometryBuffer(geom,
                                                                  RTC_BUFFER_TYPE_INDEX,
                                                                  0,
                                                                  RTC_FORMAT_UINT3,
                                                                  3 * sizeof(uint32_t),
                                                                  nFaces()));
                for (size_t i = 0; i < indices.size(); ++i) {
                        ib[i] = indices[i];
                }

                if (back_face_culling) {
                        rtcSetGeometryIntersectFilterFunction(
                          geom, &intersectionFilter);
                        std::cout << "Backface Culling ON" << std::endl;
                } else {
                        std::cout << "Backface Culling OFF" << std::endl;
                }

                rtcCommitGeometry(geom);
                rtcAttachGeometry(scene, geom);
                rtcReleaseGeometry(geom);

                rtcCommitScene(scene);
        }

        // void build_hiprt(hiprtContext context)
        // {
        //     std::cout << "[Scene] Building AMD HIP RT scene..." << std::endl;

        //     float* d_vertices;
        //     uint32_t* d_indices;
        //     hipMalloc(&d_vertices, vertices.size() * sizeof(float));
        //     hipMalloc(&d_indices, indices.size() * sizeof(uint32_t));
        //     hipMemcpy(d_vertices, vertices.data(), vertices.size() * sizeof(float), hipMemcpyHostToDevice);
        //     hipMemcpy(d_indices, indices.data(), indices.size() * sizeof(uint32_t), hipMemcpyHostToDevice);

        //     hiprtTriangleMeshPrimitive meshGeom = {};
        //     meshGeom.vertices = d_vertices;
        //     meshGeom.vertexCount = nVertices();
        //     meshGeom.vertexStride = 3 * sizeof(float);
        //     meshGeom.triangleIndices = d_indices;
        //     meshGeom.triangleCount = nFaces();
        //     meshGeom.triangleStride = 3 * sizeof(uint32_t);

        //     hiprtGeometryBuildInput input = {};
        //     input.type = hiprtGeometryTypeTriangleMesh;
        //     input.primitive.triangleMesh = meshGeom;

        //     hiprtBuildOptions options = {};
        //     options.buildFlags = hiprtBuildFlagHintFastTrace;

        //     hiprtGeometry geometry;
        //     hiprtCreateGeometry(context, &input, &options, &geometry);

        //     size_t scratchSize;
        //     hiprtGetGeometryBuildTemporaryBufferSize(context, &input, &options, &scratchSize);
            
        //     void* d_scratch;
        //     hipMalloc(&d_scratch, scratchSize);

        //     hiprtBuildGeometry(context, hiprtBuildOperationBuild, &input, &options, d_scratch, nullptr, geometry);

        //     hipFree(d_scratch);
            
        // }

        /**
         * @fn bool intersect(const Ray &ray, IntersectInfo &info) const
         * @brief Gets the intersection information of a ray in the scene.
         * @param ray The ray in the scene
         * @param info The intersection information to retrieve
         * @return True if the ray hit the scene.
         */
        bool intersect(const Ray& ray, IntersectInfo& info) const
        {
                RTCRayHit rayhit{};
                rayhit.ray.org_x = ray.origin[0];
                rayhit.ray.org_y = ray.origin[1];
                rayhit.ray.org_z = ray.origin[2];
                rayhit.ray.dir_x = ray.direction[0];
                rayhit.ray.dir_y = ray.direction[1];
                rayhit.ray.dir_z = ray.direction[2];
                rayhit.ray.tnear = this->tnear; // err
                rayhit.ray.tfar = std::numeric_limits<float>::infinity();
                rayhit.ray.mask = -1;

                rayhit.ray.flags = 0;
                rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

                RTCRayQueryContext context;
                rtcInitRayQueryContext(&context);
                RTCIntersectArguments args;
                rtcInitIntersectArguments(&args);
                args.context = &context;

                rtcIntersect1(scene, &rayhit, &args);

                if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                        info.t = rayhit.ray.tfar;

                        int triId = rayhit.hit.primID;

                        // get triangle shape
                        const Triangle& tri = this->triangles[triId];

                        // set surface info
                        info.surfaceInfo.position = ray(info.t);
                        info.surfaceInfo.barycentric =
                          Vec2f(rayhit.hit.u, rayhit.hit.v);
                        info.surfaceInfo.geometricNormal =
                          tri.getGeometricNormal();

                        info.surfaceInfo.shadingNormal =
                          tri.computeShadingNormal(
                            info.surfaceInfo.barycentric);

                        orthonormalBasis(info.surfaceInfo.shadingNormal,
                                         info.surfaceInfo.dpdu,
                                         info.surfaceInfo.dpdv);

                        // set primitive
                        info.hitPrimitive = &this->primitives[triId];

                        return true;
                } else {
                        return false;
                }
        }

        /**
         * @fn size_t nLights() const
         * @brief Get the number of light sources in the scene.
         * @return the number of light sources in the scene.
         */
        size_t nLights() const { return lights.size(); }

        /**
         * @fn std::shared_ptr<Light> sampleLight(Sampler &sampler, float &pdf)
         * const
         * @brief samples a random light source in the scene.
         * @param sampler
         * @param pdf
         * @return a pointer towards the light source
         */
        std::shared_ptr<Light> sampleLight(Sampler& sampler, float& pdf) const
        {
                uint32_t lightIdx = lights.size() * sampler.getNext1D();
                if (lightIdx == lights.size())
                        lightIdx--;
                pdf = 1.0f / (lights.size());
                return lights[lightIdx];
        }

        /**
         * @fn  std::shared_ptr<Light> sampleLight(float &pdf, unsigned int idx)
         * const
         * @brief samples a specific light source in the scene.
         * @param pdf
         * @param idx the index of the light source
         * @return a pointer towards the light source
         */
        std::shared_ptr<Light> sampleLight(float& pdf, unsigned int idx) const
        {
                if (idx == lights.size())
                        idx--;
                pdf = 1.0f / (lights.size());
                return lights[idx];
        }
};

#endif
