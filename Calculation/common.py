import numpy as np
import HallTransport
import json
import datetime
import dill as pickle
from types import SimpleNamespace
import warnings
warnings.filterwarnings("ignore", message="MUMPS is not available,")


C2z = np.array([[0,0,0,1,0,0],
                [0,0,0,0,1,0],
                [0,0,0,0,0,1],
                [1,0,0,0,0,0],
                [0,1,0,0,0,0],
                [0,0,1,0,0,0]])
Mx   = np.array([[0,0,0,1,0,0],
                 [0,0,1,0,0,0],
                 [0,1,0,0,0,0],
                 [1,0,0,0,0,0],
                 [0,0,0,0,0,1],
                 [0,0,0,0,1,0]])
My   = np.array([[1,0,0,0,0,0],
                 [0,0,0,0,0,1],
                 [0,0,0,0,1,0],
                 [0,0,0,1,0,0],
                 [0,0,1,0,0,0],
                 [0,1,0,0,0,0]])

def create_Elist(separation_points, n_points):
    intervals = []
    for i in range(len(separation_points) - 1):
        intervals.append(np.linspace(separation_points[i], separation_points[i+1], n_points[i]))
    Elist = np.concatenate(intervals)
    Elist = np.unique(Elist)
    return Elist

def get_site_index(fsys, x, y):
    for i, site in enumerate(fsys.sites):
        if site.pos[0] == x and site.pos[1] == y:
            return i
    return None
def site_index(x,y,Lx,Ly):
    return (x*Ly) + y
def Add_Gaussian_potential(eU, position, amplitude, width):
    from scipy.stats import norm
    L, W = eU.shape
    x,y = position
    for ix in range(L):
        for iy in range(W):
            eU[ix,iy] +=  amplitude*norm.pdf(np.linalg.norm([x-ix,y-iy]), loc=0, scale=width)
    pass
def class_to_dict(Object):
        # Convert instance to dictionary, but only include JSON serializable attributes
        result = {}
        for key, value in Object.__dict__.items():
            try:
                # Try to serialize the value
                json.dumps(value)
                result[key] = value
            except (TypeError, ValueError):
                # If not serializable, skip it
                pass
        return result
def Write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)  # indent=4 makes it human-readable
    pass
def Read_json(filename):
    with open(filename, "r") as f:
        loaded_data = json.load(f)
    return loaded_data
def save_object_without_unpicklables(obj, filename):
    state = {}
    for k, v in obj.__dict__.items():
        try:
            pickle.dumps(v)
            state[k] = v
        except Exception:
            print(f"Skipping unpicklable attribute: {k}")
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    print(f"Saved to: {filename}")
def load_object_state(filename):
    with open(filename, 'rb') as f:
        state = pickle.load(f)
    return SimpleNamespace(**state)
def load_object_state_v2(cls, filename):
    """
    Loads pickled state from filename and returns a new instance of cls with that state.
    """
    with open(filename, 'rb') as f:
        state = pickle.load(f)
    obj = cls.__new__(cls)  # Create a new instance without calling __init__
    obj.__dict__.update(state)
    return obj
def save_and_reload(obj, filename=None):
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"my_object_{timestamp}.pkl"
        filename = f"Data/Data_{timestamp}.pkl"
    save_object_without_unpicklables(obj, filename)
    obj = load_object_state(filename)
    return obj

def changing_std(std, filename):
    HallData_2nd = load_object_state_v2(HallTransport.HallResistanceData_2nd, filename)
    HallData_2nd.evaluate_Hallresistance_2nd_avg(std)
    fig, ax = HallTransport.HallResistanceData_2nd.PlotHallConductance_2nd_v1(HallData_2nd.Elist, HallData_2nd.Hall_resistance_2nd_avg, gapindex=HallData_2nd.gapindex, downsample=1)