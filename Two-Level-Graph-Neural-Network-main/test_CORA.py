from CountMotif_and_node_CORA import gen_hyper_graph
    
if __name__ == "__main__":
    ghg = gen_hyper_graph()
    ghg.datasets = ['MUTAG']
    ghg.run()