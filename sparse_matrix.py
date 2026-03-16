#!/usr/bin/env python3
"""CSR sparse matrix — compressed sparse row format with multiply."""

class CSR:
    def __init__(self, rows, cols, data=None, indices=None, indptr=None):
        self.rows, self.cols = rows, cols
        self.data = data or []; self.indices = indices or []
        self.indptr = indptr or [0]*(rows+1)
    @classmethod
    def from_dense(cls, M):
        m, n = len(M), len(M[0]); data = []; indices = []; indptr = [0]
        for i in range(m):
            for j in range(n):
                if M[i][j] != 0: data.append(M[i][j]); indices.append(j)
            indptr.append(len(data))
        return cls(m, n, data, indices, indptr)
    def to_dense(self):
        M = [[0]*self.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for k in range(self.indptr[i], self.indptr[i+1]):
                M[i][self.indices[k]] = self.data[k]
        return M
    def matvec(self, x):
        y = [0]*self.rows
        for i in range(self.rows):
            for k in range(self.indptr[i], self.indptr[i+1]):
                y[i] += self.data[k] * x[self.indices[k]]
        return y
    def nnz(self): return len(self.data)

def main():
    M = [[1,0,2],[0,3,0],[4,0,5]]
    csr = CSR.from_dense(M)
    print(f"NNZ: {csr.nnz()}")
    print(f"Ax = {csr.matvec([1,2,3])}")

if __name__ == "__main__": main()
