import numpy as np

def main():
    M = 500
    nd = 1000
    K = 25
    V = 96
    mean_alpha = 0.1

    alpha = np.zeros([K])
    for k in range(K):
        alpha[k] = mean_alpha * np.random.normal(1.0 , 0.1)
    
    theta = np.zeros([M, K])
    for d in range(M):
        theta[d] = np.random.dirichlet(alpha)

    phi = np.zeros([K, V])
    signatures = open('data/signature_probability.txt', 'r').readlines()
    count = 0
    for line in signatures:
        if(count != 0):
            words = line.split()
            for signature in range(K + 3):
                if(signature >= 3):
                    phi[signature - 3, count - 1] = float(words[signature])
        count += 1

    Document = np.zeros([M, V])
    for d in range(M):
        for i in range(nd):
            temp_k = np.random.multinomial(1, theta[d])
            z = int(np.argmax(temp_k))
            temp_v = np.random.multinomial(1, phi[z])
            Document[d, int(np.argmax(temp_v))] += 1
    
    output = open('data/sample' + '_M' + str(M) + '_nd' + str(nd) + '_K' + str(K) + '.txt', 'w')
    output.write(str(M) + ' ' + str(V) + '\n')
    for d in range(M):
        for v in range(V):
            output.write(str(int(Document[d, v])) + ' ')
        output.write('\n')

if __name__ == '__main__':
    main()
