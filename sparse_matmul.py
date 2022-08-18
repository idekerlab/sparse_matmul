import math
import torch

def tensor_unwrap(T):
    flatT_indices = [[],[]]
    Tsize = list(T.size())
    Tvalues = T.values().tolist()

    for ival in range(T.values().size()[0]):
        ival_indices = [index[ival] for index in T.indices().tolist()]
        row_add = 0
        col_add = 0
        for idim in range(T.dim()-2):
            row_add += ival_indices[idim]*math.prod(Tsize[idim+1:-2])*Tsize[-2]
            col_add += ival_indices[idim]*math.prod(Tsize[idim+1:-2])*Tsize[-1]
        flatT_indices[0].append(T.indices()[-2,ival].item()+row_add)
        flatT_indices[1].append(T.indices()[-1,ival].item()+col_add)
          
    flatT = torch.sparse_coo_tensor(flatT_indices, Tvalues, (Tsize[-2]*math.prod(Tsize[:-2]), Tsize[-1]*math.prod(Tsize[:-2])))
    return(flatT)

def tensor_rewrap(flatT):
    T = flatT
    return(T)

def sparse_matmul(A, B):

    if A.is_sparse == False:
        raise Exception("Tensor A is not sparse.")

    if B.is_sparse == False:
        raise Exception("Tensor B is not sparse.")

    A = A.coalesce()
    B = B.coalesce()

    flatA = tensor_unwrap(A)
    flatB = tensor_unwrap(B)
    
    print(flatA)
    print(flatB)

    C = torch.sparse.mm(flatA, flatB)
    C = tensor_rewrap(C)

    print(C)

if __name__ == "__main__":
    print("sparse matmul implementation")
