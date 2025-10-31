import math
import random
import time as time_module
import matplotlib.pyplot as plt

G = 6.67430e-11
M_E = 5.972e24
R_E = 6371000.0
g0 = 9.80665
MU = G * M_E
OMEGA_E = 7.292115e-5

R_AIR = 287.05
GAMMA = 1.4
MOLAR_MASS = 0.0289644
R_GAS = 8.31432
ISA_G = 9.80665

ISA_LAYERS = [
    (0,      288.15, 101325.0,   -0.0065),
    (11000,  216.65, 22632.1,    0.0),
    (20000,  216.65, 5474.89,    0.001),
    (32000,  228.65, 868.02,     0.0028),
    (47000,  270.65, 110.91,     0.0),
    (51000,  270.65, 66.939,     -0.0028),
    (71000,  214.65, 3.956,      -0.002)
]

def get_isa_layer(altitude_m):
    if altitude_m >= ISA_LAYERS[-1][0]:
        return ISA_LAYERS[-1]
   
    for i in range(len(ISA_LAYERS) - 1):
        h_b, T_b, P_b, L = ISA_LAYERS[i]
        if altitude_m < ISA_LAYERS[i+1][0]:
            return ISA_LAYERS[i]
   
    return ISA_LAYERS[0]

def atmospheric_temperature(altitude_m):
    if altitude_m > 85000: return 0.0
    h_b, T_b, _, L = get_isa_layer(altitude_m)
    return T_b + L * (altitude_m - h_b)

def atmospheric_pressure(altitude_m):
    if altitude_m > 85000: return 0.0
    h_b, T_b, P_b, L = get_isa_layer(altitude_m)
    T = T_b + L * (altitude_m - h_b)
   
    if L == 0:
        exponent = (-ISA_G * MOLAR_MASS / (R_GAS * T_b)) * (altitude_m - h_b)
        P = P_b * math.exp(exponent)
    else:
        exponent = -ISA_G / (L * R_AIR)
        P = P_b * (T / T_b)**exponent
       
    return P

def atmospheric_density(altitude_m):
    if altitude_m > 85000: return 0.0
    P = atmospheric_pressure(altitude_m)
    T = atmospheric_temperature(altitude_m)
    if T <= 0: return 0.0
    return P / (R_AIR * T)

def speed_of_sound(altitude_m):
    T = atmospheric_temperature(altitude_m)
    if T <= 0: return 1.0
    return math.sqrt(GAMMA * R_AIR * T)

def mach_dependent_Cd(mach):
    if mach < 0.8: return 0.30
    if mach < 1.1: return 0.45 + 0.15 * (mach - 0.8) / 0.3
    if mach < 1.3: return 0.50
    if mach < 5.0: return 0.50 - 0.3 * (mach - 1.3) / 3.7
    return 0.20

def get_float(prompt, min_val=None):
    while True:
        try:
            value = float(input(prompt))
            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a numerical value.")

def get_int(prompt, min_val=1):
    while True:
        try:
            value = int(input(prompt))
            if value < min_val:
                print(f"Value must be at least {min_val}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a whole number.")

def calculate_orbital_parameters(r, v_r, v_theta):
    v_mag_sq = v_r**2 + v_theta**2
    v_mag = math.sqrt(v_mag_sq)
    h = r * v_theta
    E = (v_mag_sq / 2) - (MU / r)
    if E >= 0:
        return float('inf'), float('inf')
    try:
        a = -MU / (2 * E)
    except ZeroDivisionError:
        return float('inf'), float('inf')
    e_sq = 1 + (2 * E * h**2) / MU**2
    e = math.sqrt(max(0, e_sq))
    if e < 1:
        r_p = a * (1 - e)
        r_a = a * (1 + e)
        apo_alt = (r_a - R_E) / 1000
        peri_alt = (r_p - R_E) / 1000
        return apo_alt, peri_alt
    else:
        return float('inf'), float('inf')

def calculate_engine_performance(thrust_vac, isp_vac, isp_sl, altitude_m, A_e):
    P_amb = atmospheric_pressure(altitude_m)
    P_vac = 0
   
    T_vac_N = thrust_vac
   
    if altitude_m <= 0:
        isp_current = isp_sl
        thrust_current = isp_sl * g0 * (T_vac_N / (isp_vac * g0))
    else:
        thrust_current = T_vac_N + A_e * P_amb
        m_dot_vac = T_vac_N / (isp_vac * g0)
        isp_current = thrust_current / (m_dot_vac * g0)

    m_dot = thrust_current / (isp_current * g0)
   
    return thrust_current, m_dot

def get_derivatives(r, v_r, v_theta, time, current_mass, stage_data, pitch_start_time, pitch_duration, min_pitch, A_cross):
    altitude = r - R_E
    if altitude < -1000:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    v_mag = math.sqrt(v_r**2 + v_theta**2)
    thrust_current, m_dot = calculate_engine_performance(
        stage_data["thrust_vac"], stage_data["isp_vac"], stage_data["isp_sl"], altitude, stage_data["A_e"]
    )
   
    a_sound = speed_of_sound(altitude)
    mach = v_mag / a_sound
    Cd = mach_dependent_Cd(mach)

    if time < pitch_start_time:
        pitch_angle_deg = 90.0
    else:
        t_pitch = time - pitch_start_time
        pitch_rate = (90.0 - min_pitch) / pitch_duration
        pitch_angle_deg = max(90.0 - t_pitch * pitch_rate, min_pitch)
    pitch_angle_rad = math.radians(pitch_angle_deg)
   
    mu_r = -MU / r**2
   
    T_mag = thrust_current
    T_r = T_mag * math.sin(pitch_angle_rad)
    T_theta = T_mag * math.cos(pitch_angle_rad)

    rho = atmospheric_density(altitude)
    dynamic_pressure = 0.5 * rho * v_mag**2
    drag_mag = dynamic_pressure * Cd * A_cross

    if v_mag > 0:
        drag_r = -drag_mag * (v_r / v_mag)
        drag_theta = -drag_mag * (v_theta / v_mag)
    else:
        drag_r = 0
        drag_theta = 0
   
    dr_dt = v_r
    a_r = mu_r + (v_theta**2 / r) + (T_r / current_mass) + (drag_r / current_mass)
    a_theta = (-2 * v_r * v_theta / r) + (T_theta / current_mass) + (drag_theta / current_mass)
   
    return [dr_dt, a_r, a_theta, m_dot, dynamic_pressure]

def plot_flight_data(time_data, alt_data, vel_data, q_data):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle('Rocket Flight Performance')

    axs[0].plot(time_data, alt_data, label='Altitude (km)', color='blue')
    axs[0].set_ylabel('Altitude (km)')
    axs[0].grid(True, linestyle='--')
    axs[0].legend()

    axs[1].plot(time_data, vel_data, label='Velocity (m/s)', color='green')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].grid(True, linestyle='--')
    axs[1].legend()

    axs[2].plot(time_data, q_data, label='Dynamic Pressure (Pa)', color='red')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Dynamic Pressure (Pa)')
    axs[2].grid(True, linestyle='--')
    axs[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

print("\n--- HIGH-FIDELITY ROCKET LAUNCH SIMULATOR SETUP ---\n")
rocket_name = input("Rocket Name: ")
mission_name = input("Mission Name: ")
num_stages = get_int("Number of Stages: ")

print("\n--- Launch Site Details ---")
latitude_deg = get_float("Launch Site Latitude (degrees, N positive): ")
initial_v_theta_rot = R_E * OMEGA_E * math.cos(math.radians(latitude_deg))
print(f"Initial horizontal velocity due to Earth's rotation: {initial_v_theta_rot:.2f} m/s")

print("\n--- Vehicle Aerodynamics ---")
A_cross = get_float("Maximum Cross-Sectional Area (m^2): ", min_val=0.1)

stages = []
for i in range(num_stages):
    print(f"\nStage {i+1} specifications:")
    dry_mass = get_float("Dry mass (kg): ", min_val=1)
    fuel_mass = get_float("Fuel mass (kg): ", min_val=1)
    thrust_vac = get_float("Vacuum Thrust (kN): ", min_val=0.1) * 1000
    isp_vac = get_float("Vacuum Specific Impulse (s): ", min_val=10)
    isp_sl = get_float("Sea-Level Specific Impulse (s): ", min_val=10)
    A_e = get_float("Nozzle Exit Area (m^2): ", min_val=0.1)
    stages.append({
        "dry_mass": dry_mass,
        "fuel_mass": fuel_mass,
        "thrust_vac": thrust_vac,
        "isp_vac": isp_vac,
        "isp_sl": isp_sl,
        "A_e": A_e
    })

payload_mass = get_float("\nPayload Mass (kg): ", min_val=0.1)
target_apoapsis = get_float("Target Apoapsis Altitude (km): ", min_val=100)
target_periapsis = get_float("Target Periapsis Altitude (km): ", min_val=100)
target_inclination = get_float("Target Inclination (degrees): ")

total_initial_mass = payload_mass + sum(stage["dry_mass"] + stage["fuel_mass"] for stage in stages)
print(f"Calculated Total Initial Liftoff Mass: {total_initial_mass:.2f} kg")

print("\n--- Steering Profile (Gravity Turn) ---")
pitch_duration = get_float("Pitch Duration (s, e.g., 180): ", min_val=10)
min_pitch = get_float("Minimum Pitch Angle (degrees, e.g., 0.5): ", min_val=0.1)
pitch_start_time = get_float("Time to start pitch-over (s, e.g., 20): ", min_val=0)

current_mass = total_initial_mass
r = R_E
v_r = 0.0
v_theta = initial_v_theta_rot
time = 0.0
dt = 0.1
report = []
time_data = []
alt_data = []
vel_data = []
q_data = []
max_q = 0.0
max_thrust = 0.0

print("\n\n--- LAUNCH SEQUENCE STARTED (RK4 POLAR MODEL, MACH DRAG) ---\n")

for stage_index, stage in enumerate(stages):
    stage_name = f"Stage {stage_index+1}"
    print(f"IGNITION: {stage_name}")

    fuel_remaining = stage["fuel_mass"]
   
    while fuel_remaining > 0 and r >= R_E:
       
        k1 = get_derivatives(r, v_r, v_theta, time, current_mass, stage, pitch_start_time, pitch_duration, min_pitch, A_cross)
        k1_mdot = k1[3]

        r2 = r + 0.5 * dt * k1[0]
        vr2 = v_r + 0.5 * dt * k1[1]
        vtheta2 = v_theta + 0.5 * dt * k1[2]
        mass2 = current_mass - 0.5 * k1_mdot * dt
        k2 = get_derivatives(r2, vr2, vtheta2, time + 0.5 * dt, mass2, stage, pitch_start_time, pitch_duration, min_pitch, A_cross)
        k2_mdot = k2[3]

        r3 = r + 0.5 * dt * k2[0]
        vr3 = v_r + 0.5 * dt * k2[1]
        vtheta3 = v_theta + 0.5 * dt * k2[2]
        mass3 = current_mass - 0.5 * k2_mdot * dt
        k3 = get_derivatives(r3, vr3, vtheta3, time + 0.5 * dt, mass3, stage, pitch_start_time, pitch_duration, min_pitch, A_cross)
        k3_mdot = k3[3]

        r4 = r + dt * k3[0]
        vr4 = v_r + dt * k3[1]
        vtheta4 = v_theta + dt * k3[2]
        mass4 = current_mass - k3_mdot * dt
        k4 = get_derivatives(r4, vr4, vtheta4, time + dt, mass4, stage, pitch_start_time, pitch_duration, min_pitch, A_cross)
        k4_mdot = k4[3]

        r += (dt / 6) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        v_r += (dt / 6) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        v_theta += (dt / 6) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
       
        m_dot_avg = (k1_mdot + 2*k2_mdot + 2*k3_mdot + k4_mdot) / 6
        mass_burned = m_dot_avg * dt

        current_mass -= mass_burned
        fuel_remaining -= mass_burned
        time += dt
       
        altitude = r - R_E
        v_mag = math.sqrt(v_r**2 + v_theta**2)
       
        thrust_current, _ = calculate_engine_performance(stage["thrust_vac"], stage["isp_vac"], stage["isp_sl"], altitude, stage["A_e"])
        max_thrust = max(max_thrust, thrust_current)
        max_q = max(max_q, k1[4])

        if time % 1.0 < dt:
            altitude_km = altitude / 1000
           
            time_data.append(time)
            alt_data.append(altitude_km)
            vel_data.append(v_mag)
            q_data.append(k1[4])

        if fuel_remaining <= 0:
            print(f"SECO (Stage Engine Cutoff): {stage_name} at T+{time:.1f}s")
            current_mass -= stage["dry_mass"]
            print(f"Stage Jettisoned. Current Mass: {current_mass:.2f} kg")
            break
       
    if r < R_E and fuel_remaining <= 0 and stage_index < num_stages - 1:
        print("\nFATAL ERROR: Vehicle returned to Earth before final orbit achieved.")
        break

final_r = r
final_v_r = v_r
final_v_theta = v_theta
final_v_mag = math.sqrt(final_v_r**2 + final_v_theta**2)

final_apo_alt, final_peri_alt = calculate_orbital_parameters(final_r, final_v_r, final_v_theta)

print("\n\n--- MISSION REPORT: {} ---\n".format(mission_name))
print("\n--- END OF REPORT (Check Matplotlib windows for plots) ---")

plot_flight_data(time_data, alt_data, vel_data, q_data)