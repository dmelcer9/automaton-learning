import numpy as np
import graphviz


def get_feat_sequence(feats, feat_num):
    if isinstance(feats, np.ndarray):
        ep_len = feats.shape[1]
        feat_slice = feats[:, :, feat_num, :]
        return np.nonzero(feat_slice)[2].reshape(-1, ep_len).tolist()
    else:
        return [np.nonzero(feat[:, feat_num, :])[1].tolist() for feat in feats]


def nodes_to_graph(dfa, comment=None):
    nodes = set()
    graph = graphviz.Digraph(comment=comment)

    for source, edge, target in dfa:
        nodes.add(str(source))
        nodes.add(str(target))
        graph.edge(str(source), str(target), label=str(edge))

    for node in nodes:
        graph.node(str(node))

    return graph