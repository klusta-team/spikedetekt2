


if __name__ == '__main__':
    exp = Experiment()
    # All spikes in the first shank.
    spikes = exp.shanks[0].spikes
    # For all spikes, the cluster.
    spike_clusters = spikes.clusters
    # The list of all clusters.
    clusters = exp.shanks[0].clusters
    # Cluster of spike #10: two ways of accessing the same bit of data.
    assert spikes[10].cluster == spike_clusters[10]
    # Get all spikes in cluster 3.
    indices = spike_clusters == 3
    spikes_in_3 = spikes[indices]
    # For performance reasons, "clusters" should be stored independently.
    