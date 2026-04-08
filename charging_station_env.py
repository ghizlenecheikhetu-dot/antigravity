from __future__ import annotations

import copy
import math
import random


class ChargingStationEnv:
    """EV charging station with PV + battery + grid energy flows (no external deps)."""

    def __init__(self, pv_profile, grid_prices, scenario=None):
        self.pv_profile = [float(x) for x in pv_profile]
        self.grid_prices = [float(x) for x in grid_prices]
        self.scenario = scenario

        self.BATT_CAPACITY = 150.0
        self.BATT_MAX_POWER = 35.0
        self.PV_MAX_POWER = 60.0
        self.EV_MAX_CH_POWER = 11.0
        self.GRID_EXPORT_PRICE = 0.05

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)

        self.current_step = 0
        self.batt_soc = 45.0
        self.cumulative_cost = 0.0
        self.ep_pv_used = 0.0
        self.ep_pv_avail = 0.0
        self.ep_grid_used = 0.0
        self.ep_total_demand = 0.0

        if self.scenario is not None:
            self.active_evs = copy.deepcopy(self.scenario)
        else:
            self.active_evs = []
            total_hours = len(self.pv_profile)
            for h in range(total_hours):
                num_arrivals = _poisson_sample(1.0)
                for _ in range(num_arrivals):
                    cap = random.uniform(50, 100)
                    init_soc = _clip(random.gauss(0.20, 0.15), 0.05, 0.50)
                    target_soc = _clip(random.gauss(0.80, 0.20), 0.60, 1.00)
                    duration = random.randint(4, 12)
                    self.active_evs.append(
                        {
                            "arrival_time": h,
                            "initial_soc": init_soc,
                            "target_soc": target_soc,
                            "capacity": cap,
                            "duration": duration,
                            "departure_time": h + duration,
                        }
                    )

        for ev in self.active_evs:
            ev["current_soc"] = ev["initial_soc"]

        self.total_reward = 0.0
        self.cumulative_cost = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        hour = self.current_step % 24
        pv = self.pv_profile[self.current_step]
        grid_price = self.grid_prices[self.current_step]

        ev_needed = sum(
            max(0.0, ev["target_soc"] - ev["current_soc"]) * ev["capacity"]
            for ev in self.active_evs
            if ev["arrival_time"] <= self.current_step < ev["departure_time"]
        )

        return (pv, self.batt_soc, grid_price, float(hour), ev_needed)

    def step(self, action):
        batt_action, ev_action = float(action[0]), float(action[1])

        pv_avail = self.pv_profile[self.current_step]
        grid_price = self.grid_prices[self.current_step]

        present_evs = [
            ev
            for ev in self.active_evs
            if ev["arrival_time"] <= self.current_step < ev["departure_time"]
        ]
        total_ev_demand = len(present_evs) * self.EV_MAX_CH_POWER * ev_action

        reward = 0.0

        pv_to_ev = min(pv_avail, total_ev_demand)
        pv_remaining = pv_avail - pv_to_ev
        ev_remaining_demand = total_ev_demand - pv_to_ev
        reward += pv_to_ev * 2.0

        batt_room = self.BATT_CAPACITY - self.batt_soc
        pv_to_batt = min(pv_remaining, batt_room, self.BATT_MAX_POWER)
        pv_remaining -= pv_to_batt
        self.batt_soc += pv_to_batt
        reward += pv_to_batt

        grid_export = pv_remaining
        reward += grid_export * self.GRID_EXPORT_PRICE

        batt_avail = max(0.0, self.batt_soc)
        batt_to_ev = min(batt_avail, ev_remaining_demand, self.BATT_MAX_POWER)
        self.batt_soc -= batt_to_ev
        ev_remaining_demand -= batt_to_ev
        reward += batt_to_ev * 0.3

        grid_to_ev = ev_remaining_demand
        reward -= grid_to_ev * grid_price

        grid_to_batt = 0.0
        if batt_action > 0:
            charge_power = min(batt_action * self.BATT_MAX_POWER, self.BATT_CAPACITY - self.batt_soc)
            self.batt_soc += charge_power
            reward -= charge_power * grid_price
            grid_to_batt = charge_power
            if grid_price <= 0.11:
                reward += charge_power * 0.1
        elif batt_action < 0:
            discharge_power = min(abs(batt_action) * self.BATT_MAX_POWER, self.batt_soc)
            self.batt_soc -= discharge_power
            reward += discharge_power * self.GRID_EXPORT_PRICE

        for ev in self.active_evs:
            if ev["departure_time"] == self.current_step and ev["current_soc"] < ev["target_soc"]:
                shortfall_kwh = (ev["target_soc"] - ev["current_soc"]) * ev["capacity"]
                reward -= shortfall_kwh * 10.0
            elif ev["departure_time"] == self.current_step:
                reward += 50.0

        if present_evs:
            total_energy_delivered = pv_to_ev + batt_to_ev + grid_to_ev
            not_full_evs = [ev for ev in present_evs if ev["current_soc"] < ev["target_soc"]]
            if not_full_evs:
                energy_per_ev = total_energy_delivered / len(not_full_evs)
                for ev in not_full_evs:
                    ev["current_soc"] = min(1.0, ev["current_soc"] + energy_per_ev / ev["capacity"])

        self.current_step += 1
        done = self.current_step >= len(self.pv_profile) - 1

        hourly_cost = (grid_to_ev + grid_to_batt) * grid_price - (grid_export * self.GRID_EXPORT_PRICE)
        self.cumulative_cost += hourly_cost

        self.ep_pv_used += pv_to_ev + pv_to_batt
        self.ep_pv_avail += pv_avail
        self.ep_grid_used += grid_to_ev + grid_to_batt
        self.ep_total_demand += pv_to_ev + batt_to_ev + grid_to_ev + grid_to_batt

        info = {
            "pv_to_ev": pv_to_ev,
            "batt_to_ev": batt_to_ev,
            "grid_to_ev": grid_to_ev,
            "grid_to_batt": grid_to_batt,
            "batt_soc": self.batt_soc,
            "grid_export": grid_export,
            "pv_to_batt": pv_to_batt,
            "total_ev_demand": total_ev_demand,
            "grid_price": grid_price,
            "hourly_cost": hourly_cost,
            "cumulative_cost": self.cumulative_cost,
            "ev_socs": [ev["current_soc"] for ev in self.active_evs],
        }

        if done:
            fulfilled_evs = sum(1 for ev in self.active_evs if ev["current_soc"] >= ev["target_soc"] - 0.01)
            info["satisfaction"] = (fulfilled_evs / len(self.active_evs) * 100) if self.active_evs else 100
            info["pv_usage_pct"] = (self.ep_pv_used / self.ep_pv_avail * 100) if self.ep_pv_avail > 0 else 0
            info["grid_dependency_pct"] = (
                self.ep_grid_used / self.ep_total_demand * 100 if self.ep_total_demand > 0 else 0
            )

        return self._get_obs(), reward, done, False, info


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _poisson_sample(lam):
    l = math.exp(-lam)
    k = 0
    p = 1.0
    while p > l:
        k += 1
        p *= random.random()
    return k - 1
