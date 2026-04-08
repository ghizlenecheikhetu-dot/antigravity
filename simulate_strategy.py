from __future__ import annotations

import math

from charging_station_env import ChargingStationEnv


def make_profiles(hours: int = 48):
    pv_profile = []
    grid_prices = []
    for t in range(hours):
        daylight = max(0.0, math.sin(((t % 24) - 6) * math.pi / 12))
        pv_profile.append(55.0 * daylight)

        if 18 <= (t % 24) <= 22:
            price = 0.19
        elif 0 <= (t % 24) <= 5:
            price = 0.10
        else:
            price = 0.11
        grid_prices.append(price)

    return pv_profile, grid_prices


def make_scenario():
    return [
        {"arrival_time": 7, "initial_soc": 0.20, "target_soc": 0.90, "capacity": 60, "duration": 8, "departure_time": 15},
        {"arrival_time": 8, "initial_soc": 0.35, "target_soc": 0.85, "capacity": 75, "duration": 7, "departure_time": 15},
        {"arrival_time": 17, "initial_soc": 0.25, "target_soc": 0.90, "capacity": 70, "duration": 6, "departure_time": 23},
        {"arrival_time": 19, "initial_soc": 0.30, "target_soc": 0.80, "capacity": 82, "duration": 8, "departure_time": 27},
    ]


def heuristic_policy(obs):
    pv, batt_soc, grid_price, _hour, ev_needed = obs

    ev_action = 1.0 if ev_needed > 1.0 else 0.0

    if grid_price <= 0.105 and batt_soc < 120:
        batt_action = 0.8
    elif grid_price >= 0.18 and batt_soc > 30:
        batt_action = -0.7
    elif pv > 30 and batt_soc < 140:
        batt_action = 0.2
    else:
        batt_action = 0.0

    return [batt_action, ev_action]


def run_episode():
    pv_profile, grid_prices = make_profiles(hours=48)
    env = ChargingStationEnv(pv_profile=pv_profile, grid_prices=grid_prices, scenario=make_scenario())

    obs, _ = env.reset(seed=42)
    done = False
    total_reward = 0.0

    traces = []
    while not done:
        action = heuristic_policy(obs)
        obs, reward, done, _truncated, info = env.step(action)
        total_reward += reward
        traces.append(
            {
                "hour": int(obs[3]),
                "pv": float(obs[0]),
                "batt_soc": float(info["batt_soc"]),
                "price": float(info["grid_price"]),
                "ev_demand": float(info["total_ev_demand"]),
                "pv_to_ev": float(info["pv_to_ev"]),
                "grid_to_ev": float(info["grid_to_ev"]),
                "grid_to_batt": float(info["grid_to_batt"]),
                "hourly_cost": float(info["hourly_cost"]),
            }
        )

    print("=== Episode summary ===")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Cumulative cost: {info['cumulative_cost']:.2f}")
    print(f"Final battery SOC (kWh): {info['batt_soc']:.2f}")
    print(f"Satisfaction (%): {info.get('satisfaction', 0):.2f}")
    print(f"PV usage (%): {info.get('pv_usage_pct', 0):.2f}")
    print(f"Grid dependency (%): {info.get('grid_dependency_pct', 0):.2f}")

    print("\n=== First 8 timesteps (EV/PV/Grid/Batterie) ===")
    for i, row in enumerate(traces[:8]):
        print(
            f"t={i:02d} | pv={row['pv']:.1f} | price={row['price']:.2f} | "
            f"batt={row['batt_soc']:.1f} | ev_demand={row['ev_demand']:.1f} | "
            f"pv->ev={row['pv_to_ev']:.1f} | grid->ev={row['grid_to_ev']:.1f} | "
            f"grid->batt={row['grid_to_batt']:.1f} | cost={row['hourly_cost']:.2f}"
        )


if __name__ == "__main__":
    run_episode()
