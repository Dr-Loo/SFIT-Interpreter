# ---- BWSN-1 loader + DEC-ready graph ----
from water_benchmark_hub.networks.bwsn_networks import BWSN1  # docs confirm class+API
import wntr, networkx as nx
import numpy as np

def load_bwsn1_as_graph(download_dir="C:/PythonProjects/Data/BWSN"):
    # returns local path to BWSN_Network_1.inp (downloads if missing)
    inp_path = BWSN1.load(download_dir=download_dir, verbose=True, return_scenario=False)
    wn = wntr.network.io.read_inpfile(inp_path)
    # undirected for DEC on spatial edges; keep attributes if you need weights
    G = wn.get_graph().to_undirected()
    return inp_path, wn, G

def graph_to_DEC_operators(G: nx.Graph):
    # Minimal DEC-on-graph: d0 = incidence, M0/M1 = identity (replace with weights later)
    B = nx.incidence_matrix(G, oriented=True).astype(float)  # edges x nodes (scipy sparse)
    d0 = B.toarray()
    N0 = d0.shape[1]; N1 = d0.shape[0]
    M0 = np.eye(N0)   # you can replace with degree/volume weights
    M1 = np.eye(N1)   # you can replace with pipe-length/flow weights
    # No faces on a general graph → d1 = 0, so Δ1 = d0 M0^{-1} d0^T M1
    return d0, M0, M1

if __name__ == "__main__":
    inp_path, wn, G = load_bwsn1_as_graph()
    print("BWSN-1 INP:", inp_path)
    print(f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    d0, M0, M1 = graph_to_DEC_operators(G)
    # plug d0, M0, M1 into your SFIT/DEC routines
    print("DEC shapes:", d0.shape, M0.shape, M1.shape)
