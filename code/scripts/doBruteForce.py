import sys
import numpy as np

import utils


def main(argv):
    domain_a = np.arange(4)
    domain_b = np.arange(2)
    domain_c = np.arange(2)
    domain_d = np.arange(3)
    domain_e = np.arange(2)

    def pabcde(a, b, c, d, e):
        answer = utils.paGb[tuple(a, b)] * \
                 utils.pbGcd[tuple(b, c, d)] * \
                 utils.pc[c] * \
                 utils.pd[d] * \
                 utils.peGd[tuple(e, d)]
        return answer

    pa = [None] * len(domain_a)
    for i, a in enumerate(domain_a):
        total = 0.0
        for b in domain_b:
            for c in domain_c:
                for d in domain_d:
                    for e in domain_e:
                        total += pabcde(a=a, b=b, c=c, d=e, e=e)
        pa[i] = total
    print(f"pa={pa}")


if __name__ == "__main__":
    main(sys.argv)
