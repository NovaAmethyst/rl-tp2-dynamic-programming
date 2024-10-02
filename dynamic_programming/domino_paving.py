# Exercice 3 : pavage d'un rectangle avec des dominos
# ---------------------------------------------------
# On considère un rectangle de dimensions 3xN, et des dominos de
# dimensions 2x1. On souhaite calculer le nombre de façons de paver le
# rectangle avec des dominos.

# Ecrire une fonction qui calcule le nombre de façons de paver le
# rectangle de dimensions 3xN avec des dominos.
# Indice: trouver une relation de récurrence entre le nombre de façons
# de paver un rectangle de dimensions 3xN et le nombre de façons de
# paver un rectangle de dimensions 3x(N-1), 3x(N-2) et 3x(N-3).


def domino_paving(n: int) -> int:
    """
    Calcule le nombre de façons de paver un rectangle de dimensions 3xN
    avec des dominos.
    """
    a = 0
    # BEGIN SOLUTION
    def domino_paving_rec(n: int, mem: list[int]) -> int:
        if mem[n] is not None:
            return mem[n]
        if n < 3:
            mem[n] = {0: 1, 1: 1, 2: 3}[n]
            return mem[n]
        domino_paving_rec(n - 1, mem)
        domino_paving_rec(n - 2, mem)
        if n % 2 == 1:
            mem[n] = mem[n - 1] + mem[n - 2]
        else:
            mem[n] = 2 * mem[n - 1] + mem[n - 2]
        return mem[n]
    if n % 2 == 1:
        return 0
    mem = [None] * (n + 1)
    return domino_paving_rec(n, mem)
    # END SOLUTION
