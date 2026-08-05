#include "photon_map.hpp"
#ifdef _MSC_VER
namespace std
{
    #include <execution>
};
#else
    #include <execution>
#endif
#include <algorithm>
#include <numeric>

template<typename T>
void
KdTree<T>::buildTree()
{
        // setup indices of points
        std::vector<int> indices(nPoints);
        std::iota(indices.begin(), indices.end(), 0);

        // build tree recursively
        buildNode(indices.data(), nPoints, 0);
}

template<typename T>
void
KdTree<T>::buildNode(int* indices, int n_points, int depth)
{
        if (n_points <= 0)
                return;

        // choose separation axis
        const int axis = depth % 3;

        // sort indices by coordination in the separation axis.
#ifndef __APPLE__
        std::sort(std::execution::par_unseq,
          indices, indices + n_points, [&](const int idx1, const int idx2) {
                  return points[idx1][axis] < points[idx2][axis];
          });
#else
          std::sort(
          indices, indices + n_points, [&](const int idx1, const int idx2) {
                  return points[idx1][axis] < points[idx2][axis];
          });
#endif

        // index of middle element of indices
        const int mid = (n_points - 1) / 2;

        // add node to node array, remember index of current node(parent node)
        const int parentIdx = nodes.size();
        Node node;
        node.axis = axis;
        node.idx = indices[mid];
        nodes.push_back(node);

        // add left children to node array
        const int leftChildIdx = nodes.size();
        buildNode(indices, mid, depth + 1);

        // set index of left child on parent node
        // if size of nodes doesn't change, it means there is no left children
        if (leftChildIdx == nodes.size()) {
                nodes[parentIdx].leftChildIdx = -1;
        } else {
                nodes[parentIdx].leftChildIdx = leftChildIdx;
        }

        // add right children to node array
        const int rightChildIdx = nodes.size();
        buildNode(indices + mid + 1, n_points - mid - 1, depth + 1);

        // set index of right child on parent node
        // if size of nodes doesn't change, it means there is no right children
        if (rightChildIdx == nodes.size()) {
                nodes[parentIdx].rightChildIdx = -1;
        } else {
                nodes[parentIdx].rightChildIdx = rightChildIdx;
        }
}

const Photon&
PhotonMap::getIthPhoton(int i) const
{
        return this->photons[i];
}

void
PhotonMap::setPhotons(const std::vector<Photon>& p)
{
        this->photons = p;
}

const size_t
PhotonMap::nPhotons() const
{
        return this->photons.size();
}

void
PhotonMap::build()
{
#ifdef __OUTPUT__
        std::cout << "[PhotonMap] photons: " << this->photons.size() << std::endl;
#endif
        kdtree.setPoints(this->photons.data(), this->photons.size());
        kdtree.buildTree();
}

std::vector<int>
PhotonMap::queryKNearestPhotons(Vec3f& p, int k, float& max_dist2) const
{
        return kdtree.searchKNearest(p, k, max_dist2);
}
