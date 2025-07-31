import numpy as np
import matplotlib.pyplot as plt
import yaml





if __name__ == "__main__":
    config_path = "configs/simulator/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    
    angle_rad = np.deg2rad(config["angle_deg"])
    vy = config["ball_speed"] * np.cos(angle_rad) # links - rechts
    vz = config["ball_speed"] * np.sin(angle_rad) # hoich runter

    pos = np.array(config["ball_start"])
    positions = [pos.copy()]
    velocity = np.array([0.0, vy, vz])  # [vx, vy, vz]

    fps = int(config["frames"] / (config["simulation_time"] / 1000000))
    dt = 1000000.0 * (1.0 / fps)  # delta t in us (1000000 us = 1 s)

    timesteps = config["frames"]
    timesteps = 2000
    dt = 0.002

    # neue anpassungen
    timesteps = 500
    total_time = 150_000e-6  # 0.15s 
    dt = total_time / timesteps  # dt in s

    start_y = -0.6        # linker Bildrand
    end_y = 0.6           # rechter Bildrand
    start_z = 0.0         # Bildmitte als Start
    end_z = -0.25         # unterer Rand (wenn wir wollen, dass er unten endet)

    speed = config["ball_speed"]
    speed = 0.2

    v_y = (end_y - start_y) / total_time
    ratio = np.tan(angle_rad) * (speed / 6.0)  # speed skaliert flacher/steiler
    v_z = v_y * ratio

    pos = np.array([0.0, start_y, start_z])
    velocity = np.array([0.0, v_y, v_z])
    positions = [pos.copy()]

    for t in range(timesteps-1):
        dt_s = dt
        pos += velocity * dt_s  # update position
        velocity[2] -= config["g"] * dt_s

        positions.append(pos.copy())

    positions = np.array(positions)

    # Plot im neuen Koordinatensystem: y (links-rechts) vs z (hoch-runter)
    plt.figure(figsize=(8, 4))
    plt.plot(positions[:,1], positions[:,2], 'o-', markersize=4)

    plt.title(f"Tischtennisball Flugkurve\nSpeed={config['ball_speed']} m/s, Winkel={config['angle_deg']}°")
    plt.xlabel("y [m] (links → rechts)")
    plt.ylabel("z [m] (hoch ↔ runter, 0 = Bildmitte)")
    plt.grid(True)

    plt.xlim(-0.6, 0.6)  # Bildbreite von -0.6 bis 0.6 m
    plt.ylim(-0.25, 0.25)  # z in der Bildmitte ±0.3 m als Beispiel
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Bildmitte

    plt.show()

        