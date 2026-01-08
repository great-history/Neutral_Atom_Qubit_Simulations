import numpy as np


# ============================================================================
# Gaussian Pulse Function ( for Omega_{01}(t) )
# ============================================================================
def gaussian_pulse(t, sigma, t0, amp_Omega_01):
    """ Returns a Gaussian pulse shape at time t """
    return amp_Omega_01 * np.exp(-((t - t0) ** 2) / (2 * (sigma ** 2)))


# ============================================================================
# Window Pulse Function ( for delta_1(t) )
# ============================================================================
def window_pulse(t, T_detuning, amp_delta_1):
    """ Example detuning function that could depend on time and other parameters """
    if t < T_detuning and amp_delta_1 > 0:
        return amp_delta_1
    else:
        return 0


# ============================================================================
# APR Pulse Functions ( for Omega_{1r}(t) & Delta_{r}(t) )
# ============================================================================
def APR_pulse_Omega_r(t, amp_Omega_r= 2 * np.pi * 17, T_gate= 0.54, tau= 0.175 * 0.54):
    
    if t <= 0 or t > T_gate:
        return 0
    
    if t == T_gate / 2:
        return 0
    
    t0 = T_gate / 4
    param_a = np.exp(-(t0/tau)**4)
    # the first pulse
    if t > 0 and t < T_gate / 2:
        param_b = np.exp(-((t - t0)/tau)**4)     
    else:  # the second pulse
        param_b = np.exp(-((t - 3 * t0)/tau)**4)
    
    return amp_Omega_r * (param_b - param_a) / (1 - param_a)


def APR_pulse_shape_Omega_r(t, amp_Omega_r= 2 * np.pi * 17, T_gate= 0.54, tau= 0.175 * 0.54):
    pulse_shape = [APR_pulse_Omega_r(ti, amp_Omega_r, T_gate, tau) for ti in t]
    return pulse_shape


def APR_pulse_Delta_r(t, amp_Delta_r = 2 * np.pi * 17, T_gate = 0.54, tau = 0.175 * 0.54):
    if t < 0 or t > T_gate:
        return 0
    if t >= 0 and t < T_gate / 2:
        return - amp_Delta_r * np.cos(2 * np.pi / T_gate * t)
    elif t == T_gate / 2:
        return 0
    else:
        return amp_Delta_r * np.cos(2 * np.pi / T_gate * t)


def APR_pulse_shape_Delta_r(t, amp_Delta_r= 2 * np.pi * 17, T_gate= 0.54, tau= 0.175 * 0.54):
    pulse_shape = [APR_pulse_Delta_r(ti, amp_Delta_r, T_gate, tau) for ti in t]
    return pulse_shape


def Generate_APR_Pulse(t, T_gate, tau, amp_Omega_r, Delta_r):
    """ 
        Returns Adiabatic Rapid Passage ( APR ) pulse shape for Omega_{1r} at time t
        pulse_center ( t0 ) is half of the length of each pulse duration T, i.e., t0 = T/2
        The parameter a is chosen so that Omega = 0 at the beginning and end of each pulse
    """
    if t < 0 or t > T_gate:
        return 0
    
    t0 = T_gate / 2
    param_a = np.exp(-(t/tau)**4)
    param_b = np.exp(-((t - t0)/tau)**4)

    return amp_Omega_r * (param_b - param_a) / (1 - param_a), - Delta_r * np.cos(2 * np.pi / T_gate * t)


# ============================================================================
# Time-Optimal Gate Pulse Functions ( for Omega_{1r}(t) & Delta_{r}(t) )
# ============================================================================
def time_optimal_pulse_Omega_r(t, amp_Omega_r, T_gate, tau_e):
    if t < 0 or t > T_gate:
        return 0
    
    param_a = np.exp(-(t - 20 * tau_e)/tau_e)
    param_b = np.exp(-(T_gate - 20 * tau_e - t)/tau_e)
    return amp_Omega_r * (1 / (1 + param_a) + 1 / (1 + param_b) - 1)

def time_optimal_pulse_Delta_r(t, Delta_0, amp, T_gate, freq, tau, ):
    if t < 0 or t > T_gate:
        return 0
    
    t0 = T_gate / 2
    param_c = np.exp(-((t - t0)/tau)**4)
    return Delta_0 * t + amp * np.sin(2 * np.pi * freq * (t - t0)) * param_c