from tqdm import tqdm
import pandas as pd
from .utils import pinf, jmin_jmax, td
from .parameters import Parameters


class Simulation:
    "Simulation of a model run under given parameters"
    def __init__(
            self, 
            params: Parameters, 
            MOI: float, 
            timestep: int, 
            endtime: int, 
            No_init: float, 
            S_init: float
        ):
        """Initialize Simulation object.
        Args:
            params: a `Parameters` object
            MOI: Initial MOI
            timestep: time for each step (h)
            endtime: time when simulation ends (h)
            No_init: Initial uninfected cell concentration (cells/ml)
            S_init: Initial glutamine concentration in the medium (mM)
        """
        self.params = params
        assert MOI >= 0    # allow 0 only to check uninfected simulation
        self.MOI = MOI
        assert timestep > 0
        self.timestep = timestep
        assert endtime > 0
        self.endtime = endtime
        assert No_init > 0
        self.No_init = No_init
        assert S_init > 0
        self.S_init = S_init
        
        self.cur_idx = 0    # current idx for steps
        self.n_steps = self.endtime // self.timestep
        self.times = list(range(0, endtime+timestep, timestep))

        # results, following the updating order
        # a name starting with "d" means it is calculated for each step 
        # and will be applied to the next step
        self.results = {
            "S": [self.S_init] + [None]*self.n_steps,    # glutamine concentration in the medium (mM)
            "No": [self.No_init] + [None]*self.n_steps,    # uninfected cell concentration (cells/ml)
            "dV1absorp_init": [None]*(self.n_steps+1),     # virus concentration that is absorbed by non-infected cells in initial absorption (pfu/ml)
            "dNinf_add": [None]*(self.n_steps+1),    # newly infected cell
            "Ninf": [0] + [None]*self.n_steps,    # infected cell concentration (cells/ml)
            "dNd": [0] + [None]*self.n_steps,    # dead cell concentration (cells/ml)
            "dV1absorp_reinf": [None]*(self.n_steps+1),     # virus concentration that is absorbed by infected cells in re-infection (pfu/ml)
            "dV1prod": [None]*(self.n_steps+1),    # virus concentration that is produced and released into the medium (pfu/ml)
            "V1": [self.V1_init] + [None]*self.n_steps,    # extracellular virus concentration (pfu/ml)
        }

        # stats: calculating statistics along the run
        self.stats = {
            "j_distr_newinf": {}
        }

    @property
    def V1_init(self):
        return self.No_init * self.MOI

    @property
    def td(self):
        return td(self.V1_init, self.No_init)
    
    def __repr__(self):
        return "\n".join([
            f"Current step: {self.cur_idx}",
            f"Current time: {self.get_time()}",
            f"Results:"
        ]+[
            f"{res}: {self.results[res][:self.cur_idx+1]}" 
            for res in self.results
        ])

    def get_time(self, idx=None):
        if idx is None:
            idx = self.cur_idx
        return self.times[idx]

    def get_result(self, name, idx=None):
        "get the result value in self"
        if idx is None:
            idx = self.cur_idx
        return self.results[name][idx]
    
    def set_result(
            self, name, value, 
            idx=None, 
            use_lowbound=True, lowbound=0, 
            # use_round=True
        ):
        """
        Set value to `self.results[name][idx]`, apply modification as specified.
        Args:
            name: result name, must be a key in self.results
            value: value to be set
            idx: index of self.results[name], default to self.cur_idx
            use_lowbound: boolean, whether to apply a lower bound to the value.
                default to True.
            lowbound: the lower bound value to be used.
            # use_round: boolean, whether to round the value into integer, 
            #     default to True.
        """
        if idx is None:
            idx = self.cur_idx
        if use_lowbound:
            value = max(value, lowbound)
        # if use_round:
        #     value = round(value)
        self.results[name][idx] = value

    def _step_S(self):
        "update S by one step"
        prev_S = self.get_result("S", self.cur_idx-1)
        qs = (
            self.params.mumax *  prev_S
            / (self.params.Ys * (self.params.Ks + prev_S))
        )
        dS = -qs * (
            self.get_result("No", self.cur_idx-1) 
            + self.get_result("Ninf", self.cur_idx-1)
        )
        self.set_result("S", prev_S + dS)
    
    def _stats_j_distr(self, j, Nnewinf):
        if j not in self.stats["j_distr_newinf"]:
            self.stats["j_distr_newinf"][j] = Nnewinf
        else:
            self.stats["j_distr_newinf"][j] += Nnewinf

    def _step_newinf(self):
        "one step of newinfection, updates cell and virus conc"
        # calculate the non-infected cell number change
        dV_absorp_init = 0    # absorbed virus by initial infection at this step
        prev_V1 = self.get_result("V1", self.cur_idx-1)
        prev_No = self.get_result("No", self.cur_idx-1)
        prev_Ninf = self.get_result("Ninf", self.cur_idx-1)
        prev_S = self.get_result("S", self.cur_idx-1)
        if prev_No > 0:    # there are still non-infected cells
            jmin, jmax = jmin_jmax(self.params.alpha, prev_V1, prev_No)
            # calculate non-inf cell number change and virus number change
            dNinf_add = 0   # newly infected 
            for j in range(jmin, jmax+1):
                p = pinf(j, self.params.alpha, prev_V1, prev_No)
                Nnewinf_j = p * prev_No
                dNinf_add += Nnewinf_j
                self._stats_j_distr(j, Nnewinf_j)
                dV_absorp_init += (j * p * prev_No)
            dNo_growth = (self.params.mumax * prev_S / (self.params.Ks + prev_S)) * prev_No
            dNo_death = self.params.kd * prev_No
            dNo = dNo_growth - dNo_death - dNinf_add
        else:    # all cells are infected
            dNo = 0
            dNo_death = 0
            dNinf_add = 0
        self.set_result("No", prev_No + dNo)
        self.set_result("dV1absorp_init", dV_absorp_init, self.cur_idx-1)
        self.set_result("dNinf_add", dNinf_add, self.cur_idx-1)
        
        # calculate the infected cell number change
        if self.get_time() < self.td:
            dNinf_death = self.params.kd * prev_Ninf
        else:
            dNinf_death = self.params.kdinf * prev_Ninf
        self.set_result("Ninf", prev_Ninf + dNinf_add - dNinf_death)
        
        # calculate the dead cell at this step
        self.set_result("dNd", dNo_death + dNinf_death, self.cur_idx-1)

    def _step_reinf(self):
        "reinfection step, updates reinfection related virus concentrations"
        dV_absorp_reinf = 0    # absorbed virus by re-infection at this step
        prev_Ninf = self.get_result("Ninf", self.cur_idx-1)
        prev_V1 = self.get_result("V1", self.cur_idx-1)
        # the j range for reinfection, dynMOI calculated with Ninf
        jmin, jmax = jmin_jmax(
            self.params.alphaprime,
            prev_V1, prev_Ninf
        )

        # define time window for calculating reinfection
        inf_earliest_t = max(
            self.get_time() - self.params.theta,
            self.timestep    # 1 * timestep, the earliest infection time
        )
        inf_earliest_idx = self.times.index(inf_earliest_t)
        
        # for cells infected at idx step: dNinf_add,
        # calculate their reinfection at cur_idx
        for idx in range(inf_earliest_idx, self.cur_idx):
            dNinf_add_then = self.get_result("dNinf_add", idx)
            # time adjustment means the longer the cell was infected, 
            # the less the re-infection rate is. Divide time by time, not by index
            time_adj = (
                (self.params.theta - (self.get_time(self.cur_idx) - self.get_time(idx)))
                / self.params.theta
            )
            for j in range(jmin, jmax+1):
                preinf = pinf(
                    j,
                    self.params.alphaprime,
                    prev_V1, 
                    prev_Ninf
                )
                dV_absorp_reinf += (
                    time_adj *
                    preinf *
                    j *
                    dNinf_add_then
                )
        self.set_result("dV1absorp_reinf", dV_absorp_reinf, self.cur_idx-1)

    def _step_vir(self):
        "calculate virus production and V1. Equation 10 in the article"
        dV1_prod = 0
        # virun prod is dependent on substrate
        prev_S = self.get_result("S", self.cur_idx-1)
        substrate_adj = prev_S / (prev_S + self.params.Ks)
        # define time window for calculating virus production
        inf_earliest_t = max(
            self.get_time() - self.params.deltatau,
            self.timestep    # 1 * timestep, the earliest infection time
        )
        inf_earliest_idx = self.times.index(inf_earliest_t)
        # for a cell infected at idx by j virions, 
        # calculate its production at current time
        for idx in range(inf_earliest_idx, self.cur_idx):
            time_post_inf = self.get_time() - self.get_time(idx)
            jmin, jmax = jmin_jmax(
                self.params.alpha,
                self.get_result("V1", idx),
                self.get_result("No", idx)
            )
            for j in range(jmin, jmax+1):
                tauj = self.params.tau(j)
                fj = self.params.fj(j)
                # determine if the cell starts to produce virus
                if self.get_time(idx) + tauj < self.get_time():
                    # intrinsic decay of infected cells
                    decay_adj = (
                        (self.params.deltatau - time_post_inf + tauj) /
                        (self.params.deltatau)
                    )
                    # n(t, xi, j)
                    p = pinf(
                        j, self.params.alpha, 
                        self.get_result("V1", idx),
                        self.get_result("No", idx)
                    )
                    dV1_prod += (
                        self.params.kv *
                        fj *
                        decay_adj *
                        substrate_adj *
                        p * self.get_result("No", idx)
                    )
        self.set_result("dV1prod", dV1_prod, self.cur_idx-1)
        # update virus
        prev_V1 = self.get_result("V1", self.cur_idx-1)
        cur_V1 = (
            prev_V1
            - self.get_result("dV1absorp_init", self.cur_idx-1)
            - self.get_result("dV1absorp_reinf", self.cur_idx-1)
            + dV1_prod
        )
        self.set_result("V1", cur_V1)
        
    def step(self):
        """
        Run one step of model simulation with current parameters and model run settings.
        """
        # step idx as the first thing to do
        self.cur_idx += 1
        
        self._step_S()
        self._step_newinf()
        self._step_reinf()
        self._step_vir()
    
    def run(self, endtime=None):
        "run the simulation toward the endtime"
        for _ in tqdm(range(self.cur_idx, self.n_steps)):
            self.step()

    def to_df(self):
        "save results until current step into a pandas DataFrame"
        return pd.DataFrame(
            data={name:lst[:self.cur_idx+1] for name, lst in self.results.items()},
            index=pd.Index(self.times[:self.cur_idx+1], name="time")
        )
