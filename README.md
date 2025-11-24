# Etekcity Fitness Scale BLE Integration for Home Assistant

This custom integration allows you to connect your Etekcity Bluetooth Low Energy (BLE) fitness scale to Home Assistant. It provides real-time weight measurements and body composition metrics directly in your Home Assistant instance, without requiring an internet connection or the VeSync app.

[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/yellow_img.png)](https://www.buymeacoffee.com/ronnnnnnn)

## Features

- Automatic discovery of Etekcity BLE fitness scales
- Intelligent multi-user support:
    - Automatically detects which person is using the scale based on their weight history.
    - Uses an adaptive tolerance system that adjusts to each user's weight fluctuations over time.
    - Supports linking users to Home Assistant Person entities to exclude users who are `not_home`.
- Real-time weight and impedance measurements
- Optional body composition metrics calculation including:
    - Body Mass Index (BMI)
    - Body Fat Percentage
    - Fat Free Weight
    - Subcutaneous Fat Percentage
    - Visceral Fat Value
    - Body Water Percentage
    - Basal Metabolic Rate
    - Skeletal Muscle Percentage
    - Muscle Mass
    - Bone Mass
    - Protein Percentage
    - Metabolic Age
- Customizable display units (kg, lb)
- Direct Bluetooth communication (no internet or VeSync app required)

## Notes

- This integration does not currently support "Athlete Mode". All body composition measurements are based on standard calculations.
- This integration uses the [etekcity_esf551_ble](https://github.com/ronnnnnnnnnnnnn/etekcity_esf551_ble) Python library for communication with the scale.

## Installation

### HACS (Recommended)

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=ronnnnnnnnnnnnn&repository=etekcity_fitness_scale_ble&category=integration)

1. Ensure that [HACS](https://hacs.xyz/) is installed in your Home Assistant instance.
2. In the HACS panel, search for "Etekcity Fitness Scale BLE".
3. Click "Download" on the Etekcity Fitness Scale BLE integration.
4. Restart Home Assistant.

### Manual Installation

1. Copy the `etekcity_fitness_scale_ble` folder to your Home Assistant's `custom_components` directory.
2. Restart Home Assistant.

## Configuration

### Initial Setup

1. In Home Assistant, go to "Configuration" > "Integrations".
2. Click the "+" button to add a new integration.
3. Search for "Etekcity Fitness Scale BLE" and select it.
4. Follow the configuration steps:
    - Choose your preferred unit system (Metric or Imperial)
    - Optionally enable body composition metrics
    - If body composition is enabled:
        - Select your sex
        - Enter your date of birth
        - Enter your height

### User Profile Configuration Options

When adding or editing user profiles (**Settings > Devices & Services > Etekcity Fitness Scale BLE > Configure**), you can configure the following options:

- **User Name:** Display name for the user profile.

- **Person Entity (optional):** Link this user profile to a Home Assistant person entity. When linked, the integration uses the person's location state to improve automatic assignment:
  - If the person is marked as `not_home`, they are excluded from automatic assignment for new measurements
  - This helps avoid incorrectly assigning measurements when household members are away

- **Mobile Devices (optional):** Select one or more mobile devices (via Home Assistant companion app) to receive actionable notifications for ambiguous measurements:
  - When enabled, you'll receive a mobile notification with tap-to-assign buttons directly on your phone
  - Each candidate user gets a personalized notification with "This is me" and "Not me" buttons

- **Enable body composition metrics:** Calculate additional health metrics (BMI, body fat %, etc.) based on impedance measurements. Requires sex, date of birth, and height.

## Multi-User Support

This integration is designed for households with multiple users. You can create a unique profile for each person using the scale.

### Person Detection

When a new measurement is received, the integration attempts to automatically assign it to the correct person based on two factors:

1. **Weight History:** The measurement is compared against each user's weight history.
2. **Location:** If a user profile is linked to a Home Assistant `person` entity, the integration checks if that person is `home`. Users who are `not_home` are excluded from automatic assignment.

If a single user is a clear match, the measurement is assigned automatically.

### Ambiguous Measurements

If the measurement is ambiguous (e.g., two users have similar weights, or a new user has no history), the integration will notify you:

- **Mobile Notifications (if configured):** Each candidate user receives a personalized notification on their mobile device with actionable buttons:
  - "This is me" - Assigns the measurement to you
  - "Not me" - Dismisses your notification (measurement remains available for others)

- **Persistent Notifications:** A notification appears in the Home Assistant notifications panel with instructions to manually assign the measurement using the `assign_measurement` service.

### Managing Users

You can manage user profiles by navigating to your device in **Settings > Devices & Services > Etekcity Fitness Scale BLE**. Click **CONFIGURE** to:
- **Add a new user:** Create a new profile with optional person entity link and mobile notification settings.
- **Edit a user:** Update a user's name, linked person entity, mobile devices, or body metric settings.
- **Remove a user:** Delete a user's profile and all associated sensor entities.

## Legacy Default User (Old Version Migration)

If you used the original single-user version of the integration, migrating to this version keeps your existing sensors by creating a "Default User" whose `user_id` is an empty string (`""`):
- Sensors for the legacy user keep their original entity IDs (no name prefix) so dashboards and automations continue working.
- When calling services, set `user_id: ""` anytime you want to target the legacy profile.

## Services

The integration provides services to manage measurements, especially for handling ambiguous weigh-ins. You can use these in scripts or automations, or call them directly from **Developer Tools > Actions**.

### `etekcity_fitness_scale_ble.assign_measurement`
Assign a pending (ambiguous) measurement to a specific user. The `timestamp` and candidate `user_id`s are provided in the persistent notification.

**Example:**
```yaml
service: etekcity_fitness_scale_ble.assign_measurement
data:
  device_id: <your_scale_device_id>
  timestamp: "2025-11-06T15:30:00.123456"
  user_id: "jane" # or "" for legacy user
```

### `etekcity_fitness_scale_ble.reassign_measurement`
Reassign the most recent measurement from one user to another. This is useful if a measurement was automatically but incorrectly assigned.

**Example:**
```yaml
service: etekcity_fitness_scale_ble.reassign_measurement
data:
  device_id: <your_scale_device_id>
  from_user_id: "john2"
  to_user_id: "jane" # or "" for legacy user
```

### `etekcity_fitness_scale_ble.remove_measurement`
Remove the last measurement for a specific user. This will revert the user's sensors to their previous values.

**Example:**
```yaml
service: etekcity_fitness_scale_ble.remove_measurement
data:
  device_id: <your_scale_device_id>
  user_id: "john2" # or "" for legacy user
```

## Diagnostic Sensors

The integration creates two diagnostic sensors to provide visibility into its state:
- **User Directory:** Shows the number of configured user profiles and lists their details (including `user_id`) in the attributes.
- **Pending Measurements:** Shows the number of ambiguous measurements awaiting manual assignment and lists their timestamps in the attributes.

## Supported Devices

This integration has been tested with the following Etekcity scale models:

- [ESF-551 (Smart Fitness Scale)](https://etekcity.com/products/smart-fitness-scale-esf551)

Other Etekcity BLE fitness scale models may work but have not been tested. If you try it with a different model, please let me know whether it works or not.

## Troubleshooting

- Make sure your scale is within range of your Home Assistant device, or within range of at least one ESPHome device configured as a Bluetooth proxy in Home Assistant.
- If you encounter any issues, please check the Home Assistant logs for more information.

### Raspberry Pi 4 and other Linux machines using BlueZ

If you encounter a `org.bluez.Error.InProgress` error, try the following in `bluetoothctl`:

```
power off
power on
scan on
```

(See [this GitHub issue](https://github.com/home-assistant/core/issues/76186#issuecomment-1204954485) for more information)

## Support the Project

If you find this unofficial project helpful, consider buying me a coffee! Your support helps maintain and improve this integration.

[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/yellow_img.png)](https://www.buymeacoffee.com/ronnnnnnn)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

This integration is not official. It is not endorsed by, directly affiliated with, maintained, authorized, or sponsored by Etekcity, VeSync Co., Ltd., or any of their affiliates or subsidiaries. All product and company names are the registered trademarks of their original owners. The use of any trade name or trademark is for identification and reference purposes only and does not imply any association with the trademark holder of their product brand.

Use this integration at your own risk.
