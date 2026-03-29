import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import qmc, norm

def correct_divid_zero(x: np.ndarray, y: np.ndarray, value: float) -> np.ndarray:
    return np.divide(x, y, out=np.ones(y.shape)*value, where=(y != 0))

def erreur_multiplicative(x: float, y: float) -> float:
    return np.log(x)-np.log(y)

def halton_sequence(x: tuple[int, int]) -> np.ndarray:
    dim, size = x
    return qmc.Halton(dim).random(size).transpose()

def box_muller(U: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = 2 * np.pi * U
    first=np.sqrt(-2 * np.log(V))
    return first*np.cos(theta), first*np.sin(theta)

def create_normal_sample(dim: int, size: int, rdm_function=np.random.random) -> np.ndarray:
    dimo = (dim + (dim % 2))//2
    unif=rdm_function((dimo*2,size))
    truc=box_muller(unif[:dimo], unif[dimo:])
    return np.concatenate(truc)[:dim]

class convergence:
    def __init__(self,x: np.ndarray):
        self.x=x
        self.n=np.arange(1,x.shape[-2]+1)
        self.mean()
        self.std()
        self.ic()
    def _convergence(self, x: np.ndarray) -> np.ndarray:
        return x.transpose()[-1].cumsum(axis=0).transpose()/self.n
    def mean(self) -> None:
        self.r_mean=self._convergence(self.x)
    def std(self) -> None:
        self.r_std=np.sqrt((self._convergence(self.x**2)-self.r_mean**2)*correct_divid_zero(self.n, self.n-1,0))
        # car variance avec 1 element inexistante
    def ic(self) -> None:
        self.r_ic=1.96*self.r_std/np.sqrt(self.n)
    def get(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.r_mean, self.r_std, self.r_ic


def result(X, theorical=None, y_title=None, global_title=None, log=True):
    color = ["blue", "red", "green", "purple"]
    print('-' * 30)
    for i in range(len(X)):
        itera = X[i].shape[-2]
        C = convergence(X[i])
        print('x'*30)
        print(f"esperance: {C.r_mean[-1]}")
        print(f"std: {C.r_std[-1]}")
        if log:
            plt.yscale('log')
            plt.xscale('log')

        plt.plot(C.r_mean, color=color[i])
        plt.fill_between(range(itera), C.r_mean - C.r_ic, C.r_mean + C.r_ic, color=color[i], alpha=0.2)
        if theorical is not None:
            plt.plot(np.ones(itera) * theorical, color="black",alpha=0.2)
            print(f"erreur multiplicative: {erreur_multiplicative(C.r_mean[-1], theorical)}")
    print('-' * 30)
    plt.xlabel("itérations")
    plt.ylabel(y_title)
    plt.title(global_title)
    plt.show()

def bsm_d1_d2(S: float, K: float, r: float, sigma: float, T: float, t: float) -> tuple[float, float]:
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * (T-t)) / (sigma * np.sqrt(T-t))
    d2 = d1 - sigma * np.sqrt(T-t)
    return d1,d2

def bsm_call(S: float, K: float, r: float, sigma: float, T: float, t: float = 0.0) -> float:
    d1,d2=bsm_d1_d2(S,K,r,sigma,T,t)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * (T-t)) * norm.cdf(d2)
    return call_price

def bsm_put(S: float, K: float, r: float, sigma: float, T: float, t: float = 0.0) -> float:
    d1,d2=bsm_d1_d2(S,K,r,sigma,T,t)
    put_price = K * np.exp(-r * (T-t)) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def bsm_St(
    S: float,
    r: float,
    sigma: float,
    N: int,
    m: int,
    T: float,
    t: float = 0.0,
    rdm_function=np.random.random,
    Z: np.ndarray | None = None
) -> np.ndarray:
    dt=(T-t)/N
    if Z is None:
        Z_t = create_normal_sample(m, N,rdm_function=rdm_function)
    else:
        Z_t=Z
    return S * np.exp((r - sigma ** 2 / 2) * dt + sigma * Z_t* np.sqrt(dt)).cumprod(axis=-1)

def compare(x,y1,y2,x_title=None,y_title=None,global_title=None):
    #faire test stat
    plt.plot(x,y1)
    plt.plot(x,y2,linestyle='--')
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(global_title)
    plt.show()

def van_der_corput(n: int, base: int) -> float:
    vdc, denom = 0, 1
    while n > 0:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder / denom
    return vdc

def get_primes(n: int) -> list[int]:
    primes = []
    num = 2
    while len(primes) < n:
        for i in range(2, int(num ** 0.5) + 1):
            if (num % i) == 0: break
        else:
            primes.append(num)
        num += 1
    return primes

def get_beta(Y: np.ndarray, C: np.ndarray) -> np.ndarray:
    yc=convergence(Y*C)
    y=convergence(Y)
    c=convergence(C)
    beta=correct_divid_zero(yc.r_mean-y.r_mean*c.r_mean, c.r_std**2,0)
    return beta.reshape(-1,1)