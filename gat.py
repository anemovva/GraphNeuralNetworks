import torch.nn as nn
import torch as torch

class GATHead(nn.Module):
    def __init__(self, in_dim):
        super(GATHead, self).__init__()
        self.attnn = nn.Linear(2 * in_dim, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
    
    def processnodes(self, i, j, h):
        # i and j are the indices of the nodes connected by the edge
        # h is the input node features
        # return exp(eij)
        
        # concat h[i] and h[j]
        hc = torch.cat((h[i], h[j]))
        out = self.leakyrelu(self.attnn.forward(hc))
        return out
    
    def forward(self, g, h, i, j):
        # g is the graph in Adjacency list format (list of pairs of nodes)
        # h is the input node features
        # i,j is the pair that we need to calc attention for
        # return the output node features
        # print("Edge: {}, {}".format(i, j))
        sumneighbors = 0;
        for pair in g:
            if pair[0]==i:
                sumneighbors += self.processnodes(i, pair[1], h)
        return self.processnodes(i, j, h) / sumneighbors
        
        
        

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, final):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList()
        self.out_dim = out_dim
        self.final = final
        # Change based on whatever is needed
        self.nonlinear = nn.ReLU()
        
        for i in range(num_heads):
            self.heads.append(GATHead(in_dim))
        if final:
            self.fc = nn.Linear(in_dim, out_dim)
        else:
            self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, data):
        # g is the graph in Adjacency list format (list of pairs of nodes)
        # h is the input node features
        # return the output node features


        h = data.x
        g = data.edge_index


        if self.final:
            out = torch.zeros((h.size()[0], self.out_dim))
            for i in range(h.size(0)):
                for head in self.heads:
                    for index in range(g.shape[1]):
                        
                        j = g[0][index]
                        k = g[1][index]
                        
                        if j == i:
                            out[i] += head.forward(g, h, j, k)*self.fc(h[k])
                            # print(head.forward(g, h, j, k))
                            
            
            
            out = out/self.num_heads
            out = self.nonlinear(out)
            # Print the max value in out
            
            return out
        else:
            outcollection = []
            
            for head in self.heads:
                out = torch.zeros((h.size()[0], self.out_dim))
                for i in range(h.size(0)):
                    for pair in g:
                        j = pair[0]
                        k = pair[1]
                        if j == i:
                            out[i] = torch.add(head.forward(g, h, j, k)*self.fc(h[k]), out[i])
                out = self.nonlinear(out)

                outcollection.append(out)
            # Attach all of the outs together in 0th dimension
            outreal = torch.cat(outcollection, 1)
            return outreal
        
class GAT(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GAT, self).__init__()
        self.initiallayer = GATLayer(in_dim, out_dim, 8, False)
        self.predictionlayer = GATLayer(8*out_dim, 1, 1, True)
    
    def forward(self, data):
        hprime = self.initiallayer.forward(data)
        dataprimed = data
        dataprimed.x = hprime
        return self.predictionlayer.forward(dataprimed)
