# Etekcity Fitness Scale BLE Integration for Home Assistant

This custom integration allows you to connect your Etekcity Bluetooth Low Energy (BLE) fitness scale to Home Assistant. It provides real-time weight measurements and, for supported models, body composition metrics directly in your Home Assistant instance, without requiring an internet connection or the VeSync app.

[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/yellow_img.png)](https://www.buymeacoffee.com/ronnnnnnn)

## Features

- **Supported models:** ESF-551 (full features: weight, impedance, body composition), FIT-8S (weight, impedance, body composition; experimental), ESF-24 (weight, impedance, body composition; experimental), EFS-A591S-KUS / Apex HR (weight, impedance, body composition, heart rate; experimental), EFS-C651 (weight, impedance, body composition; experimental) and ESF-37 (weight only; experimental)
- Automatic discovery of Etekcity BLE fitness scales
- Intelligent multi-user support:
    - Automatically detects which person is using the scale based on their weight history.
    - Uses an adaptive tolerance system that adjusts to each user's weight fluctuations over time.
    - Supports linking users to Home Assistant Person entities to exclude users who are `not_home`.
- Real-time weight and impedance measurements
- Optional body composition metrics (all supported models), with optional per-user Athlete Mode and a per-user alternative-algorithm option, including:
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
- Heart rate in beats per minute (EFS-A591S only)
- Customizable display units (kg, lb)
- Direct Bluetooth communication (no internet or VeSync app required)

## Notes

- "Athlete Mode" is available as a per-user option on the body composition profile, applying the same athlete-tuned body fat calculation as the VeSync app's Athlete Mode toggle.
- **FIT-8S** is advertisement-based: the display unit you select affects the Home Assistant display only and is *not* sent to the scale. For ESF-551, ESF-24, EFS-A591S and EFS-C651, the selected unit is pushed to the scale's screen; Home Assistant cannot change what a FIT-8S shows (use the button on the scale for that), so for FIT-8S the two are independent.
- This integration uses the [etekcity_esf551_ble](https://github.com/ronnnnnnnnnnnnn/etekcity_esf551_ble) Python library (v0.8.3+) for scale communication.

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

1. In Home Assistant, go to "Configuration" → "Integrations".
2. Click the "+" button to add a new integration.
3. Search for "Etekcity Fitness Scale BLE" and select it.
4. Follow the configuration steps:
    - Choose your preferred unit system (Metric or Imperial)
    - Optionally enable body composition metrics
    - If body composition is enabled:
        - Select your sex
        - Enter your date of birth
        - Enter your height
        - Optionally enable Athlete Mode

Devices that aren't recognized as a known model are never auto-discovered — they only appear when adding a device manually. In the manual device picker they show up labeled "[unknown device]" or "[Etekcity device — unknown model]". You can still configure them by choosing which supported scale protocol to try. If none of them work, please [open an issue](https://github.com/ronnnnnnnnnnnnn/etekcity_fitness_scale_ble/issues) with debug logs: they contain the device's model identifier when it broadcasts one, which is exactly what's needed to add support.

### User Profile Configuration Options

When adding or editing user profiles (**Settings → Devices & Services → Etekcity Fitness Scale BLE → Configure**), you can configure the following options:

- **User Name:** Display name for the user profile.

- **Person Entity (optional):** Link this user profile to a Home Assistant person entity. When linked, the integration uses the person's location state to improve automatic assignment:
  - If the person is marked as `not_home`, they are excluded from automatic assignment for new measurements
  - This helps avoid incorrectly assigning measurements when household members are away

- **Mobile Devices (optional):** Select one or more mobile devices (via Home Assistant companion app) to receive actionable notifications for ambiguous measurements:
  - When enabled, you'll receive a mobile notification with tap-to-assign buttons directly on your phone
  - Each candidate user gets a personalized notification with "This is me" and "Not me" buttons

- **Enable body composition metrics:** Calculate additional health metrics (BMI, body fat %, etc.) based on impedance measurements. Requires sex, date of birth and height.

- **Athlete mode (optional):** Applies the athlete-tuned body fat calculation, matching the VeSync app's Athlete Mode. Recommended for users who train regularly and have above-average muscle mass. Every metric derived from body fat (fat-free weight, muscle, water, BMR, etc.) follows the adjusted value.

- **Use alternative body composition algorithm (optional):** Each scale model computes body composition with the algorithm matching the VeSync app by default (the EFS-C651 uses a different algorithm than the other models). Enable this to switch a user to the other algorithm — useful if the values don't match what you expect, for example from another scale or app. Toggling causes a one-time step change in that user's body composition sensors; past recorded values are not changed.

### Advanced Settings

Under **Configure → Advanced Settings** you can control how measurement history is stored:

- **Keep history forever:** Never delete weight measurements automatically. While enabled, the cleanup limits below are ignored.

- **Automatic cleanup limits** (collapsed section): How many days of measurements to keep (1–365, default 90) and the maximum number of stored measurements per user (10–1000, default 100). Even when the whole history of a user is older than the retention window, their most recent measurement is always kept so automatic person detection can still recognize them when they return.

- **Enable library logging:** Include the BLE library's logs in Home Assistant's logs, useful for troubleshooting communication issues.

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

You can manage user profiles by navigating to your device in **Settings → Devices & Services → Etekcity Fitness Scale BLE**. Click **CONFIGURE** to:
- **Add a new user:** Create a new profile with optional person entity link and mobile notification settings.
- **Edit a user:** Update a user's name, linked person entity, mobile devices or body metric settings.
- **Remove a user:** Delete a user's profile and all associated sensor entities.

## Legacy Default User (Old Version Migration)

If you used the original single-user version of the integration, migrating to this version keeps your existing sensors by creating a "Default User" whose `user_id` is an empty string (`""`):
- Sensors for the legacy user keep their original entity IDs (no name prefix) so dashboards and automations continue working.
- When calling services, set `user_id: ""` anytime you want to target the legacy profile.

## Services

The integration provides services to manage measurements, especially for handling ambiguous weigh-ins. You can use these in scripts or automations, or call them directly from **Developer Tools → Actions**.

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

This integration supports the following Etekcity scale models:

| Model | Status | Features |
|-------|--------|----------|
| [ESF-551 Smart Fitness Scale](https://etekcity.com/products/smart-fitness-scale-esf551) | Fully supported | Weight, impedance, body composition, display unit |
| [FIT-8S Smart Fitness Scale](https://etekcity.com/products/smart-fitness-scale-fit-8s) | Experimental | Weight, impedance, body composition |
| [ESF-24 Smart Fitness Scale](https://us.vesync.com/product-detail/etekcity-esf24-smart-fitness-scale-335) | Experimental | Weight, impedance, body composition, display unit |
| [EFS-A591S-KUS (Apex HR)](https://etekcity.com/collections/fitness-scales/products/hr-smart-fitness-scale) | Experimental | Weight, impedance, body composition, heart rate, display unit |
| [EFS-C651 Smart Fitness Scale](https://etekcity.com/collections/fitness-scales/products/cobra-dark-blue) | Experimental | Weight, impedance, body composition (own algorithm), display unit |
| ESF-37 Smart Fitness Scale | Experimental | Weight only (scale sends both kg and lb over BLE) |

Other Etekcity BLE fitness scale models may work but have not been tested. If you'd like to help diagnose protocol compatibility for an unsupported model, see [Diagnosing Protocol Compatibility](#diagnosing-protocol-compatibility).

## Troubleshooting

- Make sure your scale is within range of your Home Assistant device, or within range of at least one ESPHome device configured as a Bluetooth proxy in Home Assistant.
- If you encounter any issues, please check the Home Assistant logs for more information.

### Bluetooth issues with a native Linux adapter (BlueZ)

If setup occasionally fails with `org.bluez.Error.InProgress`, or
measurements stop arriving on a machine using its own Bluetooth adapter,
one possible cause seen on some systems (often around Home
Assistant startup) is the integration's scanner conflicting with Home
Assistant's shared Bluetooth scanner.

**The durable fix for that cause — enable BlueZ passive scanning.**
Whenever the integration uses the machine's own adapter, it prefers
*passive* scanning, which coexists cleanly with Home Assistant's scanner.
On Linux this requires BlueZ's experimental features.

Home Assistant OS enables BlueZ experimental features out of the box, so this is
typically relevant mostly to Container/Core installs on Linux (including Raspberry
Pi). To enable BlueZ experimental features on a supported system (requires BlueZ >= 5.56 and kernel >= 5.10), on the host:

1. Edit `/etc/bluetooth/main.conf` and set, under `[General]`:
   `Experimental = true`
2. Restart the Bluetooth service: `sudo systemctl restart bluetooth`
3. Restart Home Assistant.

When passive scanning isn't available, the integration logs a warning and
raises an item under **Settings → System → Repairs**; both clear
automatically once passive scanning becomes available. If your setup works
fine as-is, you can simply ignore the repair — it won't come back unless
the situation changes.

**If the adapter is already stuck** (the `InProgress` error persists across
retries), the adapter may be in a wedged state — which can have several
causes, not all related to this integration. Reset it in `bluetoothctl`:

```
power off
power on
scan on
```

then restart Home Assistant. (See [this GitHub issue](https://github.com/home-assistant/core/issues/76186#issuecomment-1204954485) for more information.)
This recovers a stuck adapter; if the cause was scanner contention,
enabling passive scanning above can help prevent it from recurring.

## Reporting Issues

Before opening a GitHub issue:

1. **Check Settings → Repairs.** If a repair card explains the problem, the description tells you how to fix it without filing anything.
2. **Download diagnostics.** Open **Settings → Devices & Services → Etekcity Fitness Scale BLE → your scale's device card → Download Diagnostics**. This produces a redacted JSON dump of your config, coordinator state and pending measurements. Attach it to your issue.
3. **Include version info:**
   - Home Assistant version (Settings → About)
   - Integration version (visible on the scale's device card under **Configuration**)
   - Scale model (e.g. ESF-551, FIT-8S, ESF-24, EFS-A591S)
4. **If it's a BLE / connection issue,** also enable library logging in the integration's advanced settings, reproduce the problem and include the relevant log lines.

Issues go to the [GitHub issue tracker](https://github.com/ronnnnnnnnnnnnn/etekcity_fitness_scale_ble/issues).

## Diagnosing Protocol Compatibility

If you have an Etekcity BLE scale that isn't in the [Supported Devices](#supported-devices) list — or that's listed but isn't behaving the way the integration expects — you can help me diagnose the protocol-level compatibility by capturing a Bluetooth log of the official VeSync app talking to the scale. From that I can usually tell whether the scale speaks a protocol close to what this integration already understands, something different but parseable or something out of reach — and we can figure out next steps from there.

### Capturing the log on Android

1. Delete any old `btsnoop_hci*` files on your phone first.
2. In Developer Options, enable **Bluetooth HCI Snoop Log**.
3. Toggle Bluetooth off and on. This starts a fresh log file.
4. With the VeSync app, weigh yourself and **note down the exact date/time of the measurement** along with every value the app reports (weight, body fat, water %, bone mass, etc.). Also note your user profile info — sex, body height, activity level, age. All of this is the ground truth needed to verify the byte decoding against the capture.
5. Repeat step 4 at least 3 more times at noticeably different weights (e.g. yourself holding something heavy, like a crate of beer).
6. Disable **Bluetooth HCI Snoop Log**.

Then trigger a bug report from Developer Options (interactive is enough, no need for a full one). Inside the resulting zip, look under `FS\data\misc\bluetooth\logs\` for one or more files whose names begin with `btsnoop_hci` — the exact suffix varies by Android version and manufacturer (`.log`, `.log.last`, `-1.log`, `-2.log`, etc.). If you see several, send all of them.

> **Before sending: confirm the filenames start with `btsnoop_hci`, not `btsnooz_hci`.** (Note the 'z'.) The `btsnooz_hci*` variants are Android's truncated always-on diagnostic buffer — they exist even without HCI snoop logging enabled, and aren't usable for protocol analysis. If those are all you find, the snoop log wasn't actually capturing; double-check the developer option is on, toggle Bluetooth off and on and redo from step 4. Catching this before sending saves an unnecessary round-trip.

### Capturing the log on iOS

Apple provides a signed Bluetooth-logging configuration profile that enables BLE packet capture on iOS without jailbreaking. The Twocanoes Software knowledge base has a well-illustrated walkthrough: [Capture Bluetooth Packet Trace on iOS](https://twocanoes.com/knowledge-base/capture-bluetooth-packet-trace-on-ios/). You'll need a Mac to extract the captures from the resulting sysdiagnose. Android is still the easier path if you have access to both.

### What to include in the issue

Open a [GitHub issue](https://github.com/ronnnnnnnnnnnnn/etekcity_fitness_scale_ble/issues) and include:

- The marketed model name (from the box)
- The model code from the regulatory sticker on the back of the scale
- All `btsnoop_hci*` log files from the bug report, attached to the issue (or a WeTransfer / Drive link if too large for GitHub) — see filename note above before attaching
- For every weigh-in in the capture: the exact timestamp, every value the VeSync app showed (weight, body fat, water %, bone mass, etc.) and the user profile data that was active (sex, height, activity level, age)
- A note on what the scale's display shows during a measurement and after — only the weight or also other metrics, and whether the display changes over the course of the measurement until it stabilizes

## Acknowledgments

- FIT-8S support contributed by [@Flautz](https://github.com/Flautz) — thank you!
- EFS-A591S support contributed by [@r3klawz](https://github.com/r3klawz) — thank you!
- Unencrypted EFS-A591S variant support contributed by [@gthelding](https://github.com/gthelding) — thank you!
- EFS-C651 support contributed by [@tobsen111](https://github.com/tobsen111) — thank you!


## Support the Project

If you find this unofficial project helpful, consider buying me a coffee! Your support helps maintain and improve this integration.

[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/yellow_img.png)](https://www.buymeacoffee.com/ronnnnnnn)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

This integration is not official. It is not endorsed by, directly affiliated with, maintained, authorized or sponsored by Etekcity, VeSync Co., Ltd. or any of their affiliates or subsidiaries. All product and company names are the registered trademarks of their original owners. The use of any trade name or trademark is for identification and reference purposes only and does not imply any association with the trademark holder of their product brand.

Use this integration at your own risk.
