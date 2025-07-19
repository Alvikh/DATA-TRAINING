import time
from threading import Lock, Timer
from typing import Dict, List, Set

from database.db import MySQLDatabase
from database.device import DeviceManager


class DeviceMonitor:
    def __init__(self, device_manager: DeviceManager, check_interval: int = 10):
        self.device_manager = device_manager
        self.check_interval = check_interval
        self.active_devices: Set[str] = set()
        self.all_known_devices: Set[str] = set()
        self.inactive_counts: Dict[str, int] = {}  # New: track consecutive inactivity
        self.lock = Lock()
        self._timer = None
        self._start_monitoring()

    def _start_monitoring(self):
        if self._timer:
            self._timer.cancel()
        self._timer = Timer(self.check_interval, self._update_device_statuses)
        self._timer.daemon = True
        self._timer.start()

    def add_active_device(self, device_id: str):
        with self.lock:
            self.active_devices.add(device_id)
            self.all_known_devices.add(device_id)
            if device_id in self.inactive_counts:
                del self.inactive_counts[device_id]  # Reset inactive count if active again

    def _update_device_statuses(self):
        with self.lock:
            active_list = list(self.active_devices)
            inactive_list = list(self.all_known_devices - self.active_devices)

            print("=== Active Devices ===")
            for device in active_list:
                print(f"- {device}")
            
            print("=== Inactive Devices ===")
            final_inactive_list = []
            for device in inactive_list:
                # Increment inactive count
                self.inactive_counts[device] = self.inactive_counts.get(device, 0) + 1

                if self.inactive_counts[device] >= 2:
                    # Hapus dari known devices & counter karena terlalu lama tidak aktif
                    self.all_known_devices.discard(device)
                    del self.inactive_counts[device]
                    print(f"⚠️ Removing inactive device: {device}")
                else:
                    final_inactive_list.append(device)

            # Update DB
            updated_active = updated_inactive = 0
            if active_list:
                updated_active = self.device_manager.bulk_update_state(active_list, 'active')
            if final_inactive_list:
                updated_inactive = self.device_manager.bulk_update_state(final_inactive_list, 'inactive')

            print(f"\n✅ Updated {updated_active} devices to active, {updated_inactive} to inactive\n")
            self.active_devices = set()

        self._start_monitoring()

    def reset_all_devices(self):
        """
        Reset all non-maintenance devices to inactive status
        - Preserves devices with 'maintenance' status
        - Only affects devices with other statuses
        - Clears all tracking sets
        """
        with self.lock:
            # 1. Get ALL devices from database (not just known ones)
            all_devices = self.device_manager.get_all_devices_with_status()
            
            # 2. Filter out maintenance devices
            devices_to_reset = [
                dev_id for dev_id, status in all_devices.items()
                if status.lower() != 'maintenance'
            ]
            
            # 3. Bulk update non-maintenance devices
            if devices_to_reset:
                updated = self.device_manager.bulk_update_state(
                    devices_to_reset,
                    'inactive'
                )
                print(f"Reset {updated} non-maintenance devices to inactive")
            else:
                print("No non-maintenance devices to reset")
            
            # 4. Clear tracking sets
            self.active_devices = set()
            self.all_known_devices = set()
            if hasattr(self, 'inactive_counts'):
                self.inactive_counts.clear()
