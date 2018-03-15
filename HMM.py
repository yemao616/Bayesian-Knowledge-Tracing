
import math
import numpy
from numpy import random as rand
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error


class hmm:
    def __init__(self, n_state, obs_symbols, **args):
        """
        Keywords
        :param n_states (int): number of hidden states
        :param output (list, for example["0","1"]): the output symbol notations
        :param mode (string. For example, performance, time, performance+time): the output mode
        :param args: 'Pi' - matrix of initial state probability distribution
                     'T' - matrix of transmission probability
                     'E' - matrix of emission probability
                     'F' - fixed emission probability for the given state {'state1': [0.2, 0.8]}
        :return:
        """

        # Number of hidden states
        self.N = n_state
        # Observation symbols for each type of observation
        # For example, {correct, incorrect} and {fast, slow}
        self.V = obs_symbols
        # Number of observation symbols for each type
        # For example, [2, 5]
        self.M = map(len, obs_symbols)

        # Number of observation types
        self.n_elements = len(self.V)
        # The mapping of symbols to numbers
        self.symbol_map = []
        # Number of observation types
        for i in range(self.n_elements):
            self.symbol_map.append(dict(zip(self.V[i], range(len(self.V[i])))))

        # Initialize transmission probability matrix
        if 'T' in args:
            self.T = args['T']
            if numpy.shape(self.T) != (self.N, self.N):
                raise ValueError("The transmission probability matrix dimension mismatches the given states number.")

            if not numpy.array_equal(self.T.sum(1), numpy.array([1.0] * len(self.T.sum(1)))):
                raise ValueError("The sum of each row in the transmission matrix should equal to 1")
        else:

            raw_T = rand.uniform(0, 1, self.N * self.N).reshape(self.N, self.N)
            raw_T_sum = raw_T.sum(axis=1, keepdims=True)
            self.T = raw_T.astype(float) / raw_T_sum

        self.E = []

        # Initialize emission probability matrix
        if 'E' in args:
            self.E = args['E']

            if len(self.E) != self.n_elements:
                raise ValueError("There are " + str(self.n_elements) + " in the observations.")

            for i in range(self.n_elements):
                if numpy.shape(self.E[i]) != (self.N, self.M[i]):
                    raise ValueError("The emission probability matrix dimension mismatches the given states number and "
                                     "output symbols number")

                if not numpy.allclose(self.E[i].sum(1), numpy.array([1] * len(self.E[i].sum(1)))):
                    raise ValueError("The sum of each row in the emission probability matrix should equal to 1")
        else:
            for i in range(self.n_elements):
                raw_E = rand.uniform(0, 1, self.N * self.M[i]).reshape(self.N, self.M[i])
                raw_E_sum = raw_E.sum(axis=1, keepdims=True)
                self.E.append(raw_E.astype(float) / raw_E_sum)

        # Initialize the initial probability
        if 'Pi' in args:
            self.Pi = args['Pi']

            if len(self.Pi) != self.N:
                raise ValueError("The initial state probability dimension mismatches the given states number.")

            if self.Pi.sum() != 1:
                raise ValueError("The initial state probability does not add up to 1.")

        else:
            raw_Pi = numpy.array([1] * self.N)
            self.Pi = raw_Pi.astype(float) / raw_Pi.sum()

        self._print_HMM("HMM Initialization")

    def _print_HMM(self, label, write_to_file=False):
        """
        Keywords
        :param label (String "The initialized HMM parameters")
        :param write_to_file (boolean): whether to print out to file or not
        :return:
        """
        results = "\n" * 2 + "*" * 24 + "\n" + label + "\n" + "*" * 24 + "\n" \
                  + "\n1) Numerber of hidden states:" + str(self.N) \
                  + "\n2) Number of observable symbols:" + str(self.V) \
                  + "\n3) The symbol mapping in HMM:" + str(self.symbol_map) \
                  + "\n4) The transmission proability matrix T:\n" + str(self.T) \
                  + "\n5) The emission probability matrix E:\n" + str(self.E) \
                  + "\n6) The initial state probability Pi: \n" + str(self.Pi) + "\n"

        print results

    def obs_index(self, Obs, Obs_type):
        """
        Convert the observation sequences into sequence using symbols "0", "1" or "2"
        :param Obs:
        :return:
        """
        obs_index_seq = []

        for o in Obs:
            if o not in self.symbol_map[Obs_type]:
                raise ValueError("The observation symbol \"" + o + "\" is not defined in HMM")
            obs_index_seq.append(self.symbol_map[Obs_type][o])

        return obs_index_seq

    def forward(self, Obs, scaling=True, debug=False):
        """
        Calculate the probability of an observation sequence given the model parameters
        P(Obs|hmm)

        Alpha is defined as P(O_1:T,S_T|hmm)

        :param Obs: List. Observation sequence
        :param scaling: boolean. Scale the Alpha matrix to let the column sums to 1
        :param debug: boolean. Whether to print output of each step

        :return:
        """
        if debug:
            print "\n" * 2 + "*" * 23 + "\n" + "*" * 2 + " FORWARD ALGORITHM " + "*" * 2 + "\n" + "*" * 23 + "\n"

        observation = []
        # The observation sequence using observation symbol notations. It is a list ["1","1","0","1"]

        for i in range(self.n_elements):
            observation.append(self.obs_index(Obs, i))

        T = len(observation[0])
        # create scaling vector
        if scaling:
            c = numpy.zeros([T], float)

        # Initialization
        Alpha = numpy.zeros([self.N, T], float)
        Alpha[:, 0] = self.Pi
        for i in range(self.n_elements):
            Alpha[:, 0] *= self.E[i][:, int(observation[i][0])]

        if scaling:
            c[0] = 1 / Alpha[:, 0].sum()
            Alpha[:, 0] = Alpha[:, 0] * c[0]

        if debug:
            print "t=0"
            print Alpha[:, 0]

        # Induction
        for t in xrange(1, T):
            Alpha[:, t] = numpy.dot(Alpha[:, t - 1], self.T)
            for i in range(self.n_elements):
                Alpha[:, t] *= self.E[i][:, int(observation[i][t])]

            if scaling:
                c[t] = 1 / Alpha[:, t].sum()
                Alpha[:, t] = Alpha[:, t] * c[t]

            if debug:
                print "t=" + str(t)
                print Alpha[:, t]

        # Termination
        if scaling:
            log_prob = - reduce((lambda x, y: x + y), numpy.log(c[:T]))

            if debug:
                print "\nAlpha:"
                print Alpha
                print "\nc:"
                print c
                print "\nP(Obs|hmm)=" + str(log_prob)
                # print "c[T-1]: " + str(c[T-1])
            return (log_prob, Alpha, c)

        else:

            log_prob = numpy.log(numpy.sum(Alpha[:, T - 1]))

            if debug:
                print "\nAlpha:"
                print Alpha
                c = 1.0 / Alpha.sum(0)
                print c
                print "\nP(Obs|hmm)=" + str(log_prob)

            return (log_prob, Alpha)

    def backward(self, Obs, scaling, debug=False, **args):
        """
        Calculate the probability of a partial observation sequence from t+1 to T given the model params.

        Beta is defined as P(O_1:T|S_T, hmm)

        :param Obs: Observation sequence
        :return: Beta
        """
        if debug:
            print "\n" * 2 + "*" * 24 + "\n" + "*" * 2 + " BACKWARD ALGORITHM " + "*" * 2 + "\n" + "*" * 24 + "\n"

        observation = []
        # The observation sequence using observation symbol notations. It is a list ["1","1","0","1"]
        for i in range(self.n_elements):
            observation.append(self.obs_index([each[i] for each in Obs], i))

        T = len(observation[0])

        if scaling:
            c = numpy.zeros([T], float)
        # Initialization
        Beta = numpy.zeros([self.N, T], float)
        Beta[:, T - 1] = 1
        if scaling:
            c[T - 1] = 1 / Beta[:, T - 1].sum()
            Beta[:, T - 1] = Beta[:, T - 1] * c[T - 1]
        if debug:
            print "t=" + str(T - 1)
            print Beta[:, T - 1]

        # Induction

        for t in reversed(xrange(T - 1)):
            temp = self.T.copy()
            for i in range(self.n_elements):
                temp *= self.E[i][:, int(observation[i][t + 1])]

            Beta[:, t] = numpy.dot(temp, Beta[:, t + 1])
            if scaling:
                c[t] = 1 / Beta[:, t].sum()
                Beta[:, t] = Beta[:, t] * c[t]

            if debug:
                print "t=" + str(t)
                print Beta[:, t]

        # if 'c' in args:
        #    Beta = Beta * args['c']

        if scaling:

            if debug:
                print "\nBeta:"
                print Beta

            return Beta
        else:
            if debug:
                print "\nBeta:"
                print Beta
            return Beta

    def baum_welch(self, Obs_seq, **args):
        """
        Adjust the model parameters to maximize the probability of the observation sequence given the model

        Define:

        Gamma_t(i) = P(O_1:T, q_t = S_i | hmm) as the probability of in state i at time t and having the
        observation sequence.

        Xi_t(i,j) = P(O_1:T, q_t-1 = S_i, q_t = S_j | hmm) as the probability of transiting from state i
        to state j and having the observation sequence.


        :param Obs_seq: A set of observation sequence
        :param args:
            epochs: number of iterations to perform EM, default is 20
        :return:
        """
        # print "\n"*2+ "*"*24 + "\n" +"*"*1+" Bawn Welch ALGORITHM "+"*"*1 + "\n" + "*"*24 + "\n"
        epochs = args['epochs'] if 'epochs' in args else 100

        updatePi = args['updatePi'] if 'updatePi' in args else True
        updateT = args['updateT'] if 'updateT' in args else True
        updateE = args['updateE'] if 'updateE' in args else True
        debug = args['debug'] if 'debug' in args else False
        epsilon = args['epsilon'] if 'epsilon' in args else 0.001

        LLS = []

        for epoch in xrange(epochs):
            print "Epoch " + str(epoch)

            # Expected number of probability of starting from Si
            exp_si_t0 = numpy.zeros([self.N], float)
            # Expected number of transition from Si
            exp_num_from_Si = numpy.zeros([self.N], float)
            # Expected number of being in Si
            exp_num_in_Si = numpy.zeros([self.N], float)
            # Expected number of transition from Si to Sj
            exp_num_Si_Sj = numpy.zeros([self.N * self.N], float).reshape(self.N, self.N)
            # Expected number of in Si observing symbol Vk
            exp_num_in_Si_Vk = []
            for i in xrange(self.n_elements):
                exp_num_in_Si_Vk.append(numpy.zeros([self.N, self.M[i]], float))

            LogLikelihood = 0

            for Obs in Obs_seq:
                if debug:
                    print "\nThe observation sequence is " + str(Obs)
                # log_prob1, Alpha1, c1 = self.forward(Obs, scaling=True, debug=False)

                log_prob, Alpha, c = self.forward(Obs, scaling=True, debug=False)
                LogLikelihood += log_prob
                # print "Log Likelihood is " + str(LogLikelihood)

                Beta = self.backward(Obs, scaling=True, debug=False)

                if debug:
                    print "\nAlpha:"
                    print Alpha
                    print "\nBeta:"
                    print Beta
                    print len(Beta[0])

                T = len(Obs)

                observation = []
                # The observation sequence using observation symbol notations. It is a list ["1","1","0","1"]
                for i in range(self.n_elements):
                    observation.append(self.obs_index([each[i] for each in Obs], i))

                #### Calculate Gamma ####
                # Gamma is defined as the probability of
                # being in State Si at time t, given the observation sequence
                # O1, O2, O3, O4 ,....,Ot, and the model parameters

                raw_Gamma = Alpha * Beta
                Gamma = raw_Gamma / raw_Gamma.sum(0)
                if debug:
                    print "\nGamma"
                    print Gamma

                exp_si_t0 += Gamma[:, 0]
                exp_num_from_Si += Gamma[:, :T - 1].sum(1)
                exp_num_in_Si += Gamma.sum(1)

                # The probability in state Si having Observation Oj
                for i in range(self.n_elements):

                    temp = numpy.zeros([self.N, self.M[i]], float)
                    for each in self.symbol_map[i].iterkeys():
                        which = numpy.array([self.symbol_map[i][each] == int(x) for x in observation[i]])
                        temp[:, self.symbol_map[i][each]] = Gamma.T[which, :].sum(0)

                    exp_num_in_Si_Vk[i] += temp

                if debug:
                    print "\nExpected frequency in state S_i at time 0:\n" + str(exp_si_t0)
                    print "Expected number of transition from state S_i:\n" + str(exp_num_from_Si)
                    print "Expected number of time in state S_i:\n" + str(exp_num_in_Si)
                    print "Expected number of time in state S_i observing V_k:\n" + str(exp_num_in_Si_Vk)

                # Xi is defined as given the model and sequence, the probability of being in state Si at time t,
                # and in state Sj at time t+1

                Xi = numpy.zeros([T - 1, self.N, self.N], float)

                for t in xrange(T - 1):
                    for i in xrange(self.N):
                        Xi[t, i, :] = Alpha[i, t] * self.T[i, :]
                        for j in xrange(self.n_elements):
                            Xi[t, i, :] *= self.E[j][:, int(observation[j][t + 1])]
                        Xi[t, i, :] *= Beta[:, t + 1]
                    Xi[t, :, :] /= Xi[t, :, :].sum()

                for t in xrange(T - 2):
                    exp_num_Si_Sj += Xi[t, :, :]

                if debug:
                    print "\nExpected number of transitions from state Si to state Sj: \n" + str(exp_num_Si_Sj)

            # reestimate initial state probabilities
            if updatePi:
                self.Pi = exp_si_t0 / exp_si_t0.sum()
                if debug:
                    print "\nUpdated Pi:"
                    print exp_si_t0

            if updateT:
                T_hat = numpy.zeros([self.N, self.N], float).reshape(self.N, self.N)
                for i in xrange(self.N):
                    T_hat[i, :] = exp_num_Si_Sj[i, :] / exp_num_from_Si[i]
                    T_hat[i, :] /= T_hat[i, :].sum()
                self.T = T_hat

                if debug:
                    print "\nUpdated T"
                    print self.T

            if updateE:
                for j in xrange(self.n_elements):
                    E_hat = numpy.zeros([self.N, self.M[j]], float).reshape(self.N, self.M[j])
                    for i in xrange(self.N):
                        E_hat[i, :] = exp_num_in_Si_Vk[j][i, :] / exp_num_in_Si[i]
                        E_hat[i, :] /= E_hat[i, :].sum()
                    self.E[j] = E_hat
                    if debug:
                        print "\nUpdated E"
                        print self.E
            print LogLikelihood
            LLS.append(LogLikelihood)
            if epoch > 1:
                if abs(LLS[epoch] - LLS[epoch - 1]) < epsilon:
                    print "The loglikelihood improvement falls below threshold, training terminates at epoch " + str(
                        epoch) + "! "
                    break

        self._print_HMM("After training")
        return

    def predict_nlg(self, Obs_seq, debug=False):
        nlg_scores = []
        for Obs in Obs_seq:
            if debug:
                print "\nThe observation sequence is " + str(Obs)
            if Obs:
                log_prob, Alpha, c = self.forward(Obs, scaling=True, debug=False)
                # print "Posttest score:"
                # print round(Alpha[:, -1][1], 2)
                # nlg = round(Alpha[:, -1][1], 2)
                nlg = Alpha[:, -1][1]
            else:
                nlg = 0
            nlg_scores.append(nlg)
            # print "NLG Score:" + str(round(Alpha[:, -1][1], 2) - hmm.Pi[0])
        return nlg_scores


