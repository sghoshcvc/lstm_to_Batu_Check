class initialD:
    "Initialise d and pi"

    def __init__(self, allV, U):
        self.d_v = {}
        self.pi_v = {}
        self.Q = allV

        for y in self.Q:

            if y!= U:
                self.d_v[y] = float("inf")
                # print "d_v[v]", d_v[v]
                self.pi_v[y] = 'NIL'
            elif y == U:
                self.d_v[y] = 0
                self.pi_v[y] = 'NIL'