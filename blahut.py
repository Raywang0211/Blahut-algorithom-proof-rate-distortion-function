import numpy as np
import matplotlib.pyplot as plt
# In the BlahutArimotoExample() we change different kind of probabilities of X  (al) line=75
#
def BlahutArimato(dist_mat, p_x, beta, max_it=500, eps=1e-4):
    """Compute the rate-distortion function of an i.i.d distribution
    Inputs :
        'dist_mat' -- (numpy matrix) representing the distoriton matrix between the input
            alphabet and the reconstruction alphabet. dist_mat[i,j] = dist(x[i],x_hat[j])
        'p_x' -- (1D numpy array) representing the probability mass function of the source
        'beta' -- (scalar) the slope of the rate-distoriton function at the point where evaluation is
                    required
        'max_it' -- (int) maximal number of iterations
        'eps' -- (float) accuracy required by the algorithm: the algorithm stops if there
                is no change in distoriton value of more than 'eps' between consequtive iterations
    Returns :
        'Iu' -- rate (in bits)
        'Du' -- distortion
    """
    import numpy as np

    l, l_hat = dist_mat.shape
    p_cond = np.tile(p_x, (l_hat, 1)).T  # start with iid conditional distribution

    p_x = p_x / np.sum(p_x)  # normalize
    p_cond /= np.sum(p_cond, 1, keepdims=True)

    it = 0
    Du_prev = 0
    Du = 2 * eps
    stack_du = []
    stack_iu = []
    while it < max_it and np.abs(Du - Du_prev) > eps:
        it += 1
        Du_prev = Du
        p_hat = np.matmul(p_x, p_cond)

        p_cond = np.exp(-beta * dist_mat) * p_hat
        p_cond /= np.expand_dims(np.sum(p_cond, 1), 1)

        Iu = np.matmul(p_x, p_cond * np.log(p_cond / np.expand_dims(p_hat, 0))).sum()
        Du = np.matmul(p_x, (p_cond * dist_mat)).sum()

        # stack_du.append(Du)
        # stack_iu.append(Iu)
        # plt.plot(Du,Iu/np.log(2),'ro')
        # print('@@@@@@@@@@@@',it)
    # print(stack_du)
    # print(stack_iu)
    return Iu / np.log(2), Du
    #return Iu,Du

def BlahutArimotoExample():
    def hamming_dist(x, y):
        return (x != y) + 0.0

    def quad_dist(x, y):
        return (x - y) ** 2

    def bin_ent(x):
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

    def Gauss_pdf(x):
        return 1 / (2 * np.pi) * np.exp(-x ** 2 / 2)
    stack_iu = []
    stack_du = []
    for beta in np.arange(0, 1000, 0.5):
        print("current beta {:<5}".format(beta))

        # Example 1: Bernuolli input with Hamming distortion
        xx = np.array([0, 1])  # binary input
        xx_hat = np.array([0, 1])  # binary reconstruction


        al = 0.5  # P(X=1) = al ~~~~~~~~~~不同的機率分配
        p_x = np.array([(1-al), al])

        X, X_hat = np.meshgrid(xx, xx_hat)  # creat distortion matrix
        dist_mat = hamming_dist(X, X_hat)
        R, D = BlahutArimato(dist_mat, p_x, beta)  # evaluate at beta = 0.3
        stack_iu.append(R)
        stack_du.append(D)
        # check against true R(D) :
        print("Hamming Binary:")

        print("at beta = {}: D = {}, R = {}".format(beta, D, R))

        print("Difference between true R(D) (binary):")


    plt.plot(stack_du,stack_iu,'-ro')
    # print,(np.abs(bin_ent(al) - bin_ent(D)) )
     # difference between true and estimated

    # Example 2: (truncated) Gaussian input with quadratic distortion
    # xx = np.linspace(-5, 5, 1000)  # source alphabet
    # xx_hat = np.linspace(-5, 5, 1000)  # reconstruction alphabet
    # p_x = Gauss_pdf(xx)  # source pdf
    #
    # X, X_hat = np.meshgrid(xx, xx_hat)  # creat distortion matrix
    # dist_mat = quad_dist(X, X_hat)
    # R, D = BlahutArimato(dist_mat, p_x, beta)  # evaluate at beta = 0.3
    #
    # print("Quadratic Gaussian:")
    #
    # print("at beta = {}: D = {}, R = {}".format(beta, D, R))
    #
    # print("Difference between true R(D) (quadratic Gaussian):")
    #
    # print(np.abs(D - 2 ** (-2 * R)) ) # difference between true and estimated)



# if _name_ == "_main_":

print("Starting Blahut-Arimoto example...")

BlahutArimotoExample()
plt.show()