def overlappingLanelets(scenario):
    """determine all lanelets that overlap"""

    # create dictionary that maps a lanelet ID to the corresponding lanelet
    ids = [l.lanelet_id for l in scenario.lanelet_network.lanelets]
    lanelets = dict(zip(ids, scenario.lanelet_network.lanelets))

    # initialize overlaps dictionary
    overlaps = dict(zip(ids, [set() for i in range(len(ids))]))

    # loop over all lanelet combinations
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):

            l = lanelets[ids[i]]

            if ids[j] != l.adj_right and ids[j] != l.adj_left and not ids[j] in l.successor and not ids[j] in l.predecessor \
                    and l.polygon.shapely_object.intersection(lanelets[ids[j]].polygon.shapely_object).area > 0.1:
                overlaps[ids[i]].add(ids[j])
                overlaps[ids[j]].add(ids[i])

    return overlaps