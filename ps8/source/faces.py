"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2018 Mar 20
Description : Famous Faces
"""

# python libraries
import collections

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
# libraries specific to project
import util
from cluster import *

######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y) :
    """
    Translate images to (labeled) points.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets

    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """

    n,d = X.shape

    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in xrange(n) :
        images[y[i]].append(X[i,:])

    points = []
    for face in images :
        count = 0
        for im in images[face] :
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def plot_clusters(clusters, title, average) :
    """
    Plot clusters along with average points of each cluster.

    Parameters
    --------------------
        clusters -- ClusterSet, clusters to plot
        title    -- string, plot title
        average  -- method of ClusterSet
                    determines how to calculate average of points in cluster
                    allowable: ClusterSet.centroids, ClusterSet.medoids
    """

    plt.figure()
    np.random.seed(20)
    label = 0
    colors = {}
    centroids = average(clusters)
    for c in centroids :
        coord = c.attrs
        plt.plot(coord[0],coord[1], 'ok', markersize=12)
    for cluster in clusters.members :
        label += 1
        colors[label] = np.random.rand(3,)
        for point in cluster.points :
            coord = point.attrs
            plt.plot(coord[0], coord[1], 'o', color=colors[label])
    plt.title(title)
    plt.show()


def generate_points_2d(N, seed=1234) :
    """
    Generate toy dataset of 3 clusters each with N points.

    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed

    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)

    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]

    label = 0
    points = []
    for m,s in zip(mu, sigma) :
        label += 1
        for i in xrange(N) :
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))

    return points


######################################################################
# k-means and k-medoids
######################################################################

def random_init(points, k) :
    """
    Randomly select k unique elements from points to be initial cluster centers.

    Parameters
    --------------------
        points         -- list of Points, dataset
        k              -- int, number of clusters

    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2c: implement (hint: use np.random.choice)
    return np.random.choice(points, k, replace=False)
    ### ========== TODO : END ========== ###


def cheat_init(points) :
    """
    Initialize clusters by cheating!

    Details
    - Let k be number of unique labels in dataset.
    - Group points into k clusters based on label (i.e. class) information.
    - Return medoid of each cluster as initial centers.

    Parameters
    --------------------
        points         -- list of Points, dataset

    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2d: implement

    #modification -- return the centroids
    initial_points = []
    return initial_points
    ### ========== TODO : END ========== ###


def kMeans(points, k, init='random', plot=False) :
    """
    Cluster points into k clusters using variations of k-means algorithm.

    Parameters
    --------------------
        points  -- list of Points, dataset
        k       -- int, number of clusters
        init    -- string, method of initialization
                   allowable:
                       'cheat'  -- use cheat_init to initialize clusters
                       'random' -- use random_init to initialize clusters
        plot    -- bool, True to plot clusters with corresponding averages
                         for each iteration of algorithm

    Returns
    --------------------
        k_clusters -- ClusterSet, k clusters
    """

    ### ========== TODO : START ========== ###
    # part 2c: implement
    # part 2d: allow for different initialization
    #
    # Hints:
    #   (1) On each iteration, keep track of the new cluster assignments
    #       in a separate data structure. Then use these assignments to
    #       create new Cluster objects and a new ClusterSet object. Then
    #       update the centroids.
    #   (2) Repeat until the clustering no longer changes.
    #   (3) To plot, use plot_clusters(...).
    k_clusters = ClusterSet()
    k_clusters.members = [None for i in range(k)]


    if (init =='random'):
        centroids = random_init(points, k)
    else:
        centroids = cheat_init(points, k)

    iteration = 0;

    while True:
        cluster_assignments = [[] for i in range(k)]
        for p in points:
            distances = map(lambda u : p.distance(u), centroids)
            assigned_cluster = np.argmin(distances)
            cluster_assignments[assigned_cluster].append(p)

        for j in range(len(cluster_assignments)):
            k_clusters.members[j] = Cluster(cluster_assignments[j])

        if (plot):
            title = "Iteration " + str(iteration)
            plot_clusters(k_clusters, title, ClusterSet.centroids)

        new_centroids = map(lambda u : u.centroid(), k_clusters.members)

        old_centroid_attrs = map(lambda u : u.attrs, centroids)
        new_centroid_attrs = map(lambda u : u.attrs, new_centroids)

        iteration += 1

        if np.array_equal(old_centroid_attrs,new_centroid_attrs):
            break
        else:
            centroids=new_centroids



    return k_clusters



    ### ========== TODO : END ========== ###


def kMedoids(points, k, init='random', plot=False) :
    """
    Cluster points in k clusters using k-medoids clustering.
    See kMeans(...).
    """
    ### ========== TODO : START ========== ###
    # part 4a: implement
    k_clusters = ClusterSet()
    return k_clusters
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################

def main() :
    ### ========== TODO : START ========== ###
    # part 1: explore LFW data set
    X, y = util.get_lfw_data()
    #util.show_image(X[0])
    #util.show_image(X[1])
    #util.show_image(X[2])

    #util.show_image(np.mean(X, axis=1))

    U, mu = util.PCA(X)
    #util.plot_gallery([util.vec_to_image(U[:,i]) for i in xrange(12)])


    # for i in [1,10,50, 100, 500, 1288]:

    #     Z, ul = util.apply_PCA_from_Eig(X, U, i, mu)

    #     new_X = util.reconstruct_from_PCA(Z, ul, mu)

    #     util.plot_gallery([new_X[j] for j in xrange(12)])



    ### ========== TODO : END ========== ###



    #========================================
    # part 2

    # part b: test Cluster implementation
    # centroid: [ 1.04022358  0.62914619]

    np.random.seed(1234)
    sim_points = generate_points_2d(20)
    cluster = Cluster(sim_points)
    print 'centroid:', cluster.centroid().attrs

    # parts c-d: test kMeans implementation using toy dataset
    np.random.seed(1234)
    sim_points = generate_points_2d(20)
    k = 3

    # test cluster using random initialization
    kmeans_clusters = kMeans(sim_points, k, init='random', plot=True)

    # test cluster using cheat initialization
    # kmeans_clusters = kMeans(sim_points, k, init='cheat', plot=True)



    ### ========== TODO : START ========== ###
    # part 3

    # part a: explore effect of lower-dimensional representations on clustering performance
    np.random.seed(1234)

    # part b: determine ``most discriminative'' and ``least discriminative'' pairs of images
    np.random.seed(1234)

    ### ========== TODO : END ========== ###



    #========================================
    # part 4a

    # test Cluster implementation
    # medoid:   [ 1.05674064  0.71183522]

    np.random.seed(1234)
    sim_points = generate_points_2d(20)
    cluster = Cluster(sim_points)
    print 'medoid:', cluster.medoid().attrs

    # test kMedoids implementation using toy dataset
    np.random.seed(1234)
    sim_points = generate_points_2d(20)
    k = 3

    # test cluster using random initialization
    kmedoids_clusters = kMedoids(sim_points, k, init='random', plot=True)

    ### ========== TODO : START ========== ###
    # part 4

    # part b: compare k-means and k-medoids
    np.random.seed(1234)

    # part c: explore effect of lower-dimensional representations on clustering performance
    np.random.seed(1234)

    ### ========== TODO : END ========== ###


if __name__ == "__main__" :
    main()
