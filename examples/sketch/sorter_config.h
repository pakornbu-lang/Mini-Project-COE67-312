#pragma once

// Set this to the GPIO connected to the OUT pin of the fruit-detection sensor.
// Leave -1 while testing the system with the dashboard's "Capture & Classify" button.
#define FRUIT_SENSOR_PIN -1

// Most IR obstacle sensors pull OUT LOW when a fruit is detected.
// Change LOW to HIGH only if your sensor behaves in the opposite way.
#define FRUIT_SENSOR_ACTIVE_LEVEL LOW

// Wait briefly after detection so the fruit is centered in the camera view.
#define SENSOR_SETTLE_TIME_MS 300
