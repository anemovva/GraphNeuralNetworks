import pytorch.nn as nn
import pytorch as torch

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
        hc = torch.cat((h[i], h[j]), dim=1)
        out = self.leakyrelu(self.attnn.forward(hc))
        return out
    
    def forward(self, g, h, i, j):
        # g is the graph in Adjacency list format (list of pairs of nodes)
        # h is the input node features
        # i,j is the pair that we need to calc attention for
        # return the output node features
        
        sumneighbors = 0;
        # This loop is in matrix form, maybe we'll keep it for now unless the end thing proves to be slow
        for k in g[i]:
            sumneighbors += self.processnodes(i, k, h)
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
            self.heads.append(GATHead(in_dim, out_dim))
        if final:
            self.fc = nn.Linear(in_dim, out_dim)
        else:
            self.fc = nn.Linear(in_dim, out_dim * num_heads)
    
    def forward(self, g, h):
        # g is the graph in Adjacency list format (list of pairs of nodes)
        # h is the input node features
        # return the output node features

        if self.final:
            out = torch.zeros(self.out_dim)
            for i in range(self.num_heads):
                for j in g[i]:
                    for head in self.heads:
                        out[i] += head.forward(g, h, i, j)*self.fc(h[j])
                out [i] = self.nonlinear(out[i])
            return out
        else:
            outreal = torch.empty(0)
            for head in self.heads:
                out = torch.zeros(self.out_dim)
                for i in range(self.num_heads):
                    for j in g[i]:
                        out[i] += head.forward(g, h, i, j)*self.fc(h[j])
                    out [i] = self.nonlinear(out[i])
                outreal.append(out)
            return outreal
        
class GAT(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GAT, self).__init__()
        self.initiallayer = GATLayer(in_dim, out_dim, 8, False)
        self.predictionlayer = GATLayer(out_dim, 1, 1, True)
    
    def forward(self, g, h):
        hprime = self.initiallayer.forward(g, h)
        return self.predictionlayer.forward(g, hprime)
