# Robot Teleoperation - ServoP Control

Smooth teleoperation using ServoP with tunable PID parameters.

## Setup

```bash
pip install -r requirements.txt
python test2_teleoperation.py
```

## Usage

1. Enter IP (8.209.98.146)
2. Connect → Enable
3. Adjust speed (30%)
4. Control

## How It Works

Single loop at 20Hz (0.05s):
- Calculate position
- Send ServoP command with t=0.05
- ServoP tells robot "reach target in 0.05s"
- Continuous ServoP = smooth motion

## Reduce Jitter

Robot vibrating/jittery? Adjust PID in `config.py`:

```python
UPDATE_RATE = 0.05

# Default (moderate response):
SERVO_GAIN = 300        # P gain (200-1000)
SERVO_AHEADTIME = 30    # D term (20-100)

# Less jitter (softer):
SERVO_GAIN = 200
SERVO_AHEADTIME = 20

# More responsive (may jitter more):
SERVO_GAIN = 500
SERVO_AHEADTIME = 50
```

**Tuning Guide:**
- **SERVO_GAIN**: Lower = smoother but slower response
- **SERVO_AHEADTIME**: Lower = less overshoot/jitter
- Start low, increase until responsive without jitter

**Control rate:**
- `UPDATE_RATE = 0.05` (20Hz) - standard
- `UPDATE_RATE = 0.04` (25Hz) - faster
- `UPDATE_RATE = 0.1` (10Hz) - slower/safer
