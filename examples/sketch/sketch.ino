#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "img_converters.h"
#include "esp_heap_caps.h"

#include <math.h>

#include "select_pins.h"
#include "model_data.h"
#include "sorter_config.h"
#if __has_include("wifi_secrets.local.h")
#include "wifi_secrets.local.h"
#else
#include "wifi_secrets.h"
#endif

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// =====================================================
// การตั้งค่าโมเดล
// =====================================================

constexpr int MODEL_WIDTH = 96;
constexpr int MODEL_HEIGHT = 96;
constexpr int MODEL_CHANNELS = 3;

constexpr size_t TENSOR_ARENA_SIZE = 2 * 1024 * 1024;

const char *CLASS_NAMES[] = {
    "unripe",
    "ripe",
    "overripe"
};

const tflite::Model *mangosteen_model = nullptr;
tflite::MicroInterpreter *mangosteen_interpreter = nullptr;

// Implemented in app_httpd.cpp. The web page reads this latest result.
extern void updateMangosteenResult(
    const char *label,
    const float *probabilities,
    unsigned long inference_time_ms,
    unsigned long cycle_time_ms
);
extern void updateLatestCapture(
    const uint8_t *jpeg,
    size_t jpeg_len
);
extern void startCameraServer();

static portMUX_TYPE capture_request_lock = portMUX_INITIALIZER_UNLOCKED;
static bool capture_requested = false;

// Called by the web dashboard; the loop safely runs the capture on the next cycle.
void requestClassification()
{
    portENTER_CRITICAL(&capture_request_lock);
    capture_requested = true;
    portEXIT_CRITICAL(&capture_request_lock);
}

bool takeClassificationRequest()
{
    portENTER_CRITICAL(&capture_request_lock);
    bool requested = capture_requested;
    capture_requested = false;
    portEXIT_CRITICAL(&capture_request_lock);
    return requested;
}

void setupFruitSensor()
{
#if FRUIT_SENSOR_PIN >= 0
    pinMode(FRUIT_SENSOR_PIN, INPUT_PULLUP);
    Serial.printf("Fruit sensor enabled on GPIO %d\n", FRUIT_SENSOR_PIN);
#else
    Serial.println("Fruit sensor disabled; use the dashboard Capture & Classify button");
#endif
}

bool fruitSensorTriggered()
{
#if FRUIT_SENSOR_PIN >= 0
    static bool was_detected = false;
    bool detected =
        digitalRead(FRUIT_SENSOR_PIN) == FRUIT_SENSOR_ACTIVE_LEVEL;
    bool triggered = detected && !was_detected;
    was_detected = detected;
    return triggered;
#else
    return false;
#endif
}

TfLiteTensor *model_input = nullptr;
TfLiteTensor *model_output = nullptr;

uint8_t *tensor_arena = nullptr;

// =====================================================
// เปิดกล้อง
// =====================================================

bool setupCamera()
{
#if defined(PWR_ON_PIN)
    pinMode(PWR_ON_PIN, OUTPUT);
    digitalWrite(PWR_ON_PIN, HIGH);
    delay(200);
#endif

    camera_config_t config = {};

    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;

    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;

    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;

    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;

    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;

    config.xclk_freq_hz = 20000000;

    // ถ่ายเป็น JPEG แล้วแปลงเป็น RGB ก่อนส่งเข้าโมเดล
    config.pixel_format = PIXFORMAT_JPEG;

    // ใช้ภาพ 320x240
    config.frame_size = FRAMESIZE_QVGA;

    config.jpeg_quality = 10;
    // Two frame buffers let the video stream and the classifier share the camera.
    config.fb_count = 2;

    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_LATEST;

    esp_err_t result = esp_camera_init(&config);

    if (result != ESP_OK) {
        Serial.printf(
            "Camera initialization failed: 0x%x\n",
            result
        );

        return false;
    }

    sensor_t *sensor = esp_camera_sensor_get();

    if (sensor != nullptr) {
        sensor->set_framesize(sensor, FRAMESIZE_QVGA);
        sensor->set_brightness(sensor, 0);
        sensor->set_contrast(sensor, 0);
        sensor->set_saturation(sensor, 0);

        // เปิด Auto White Balance
        sensor->set_whitebal(sensor, 1);
        sensor->set_awb_gain(sensor, 1);

        // เปิด Auto Exposure และ Auto Gain
        sensor->set_exposure_ctrl(sensor, 1);
        sensor->set_gain_ctrl(sensor, 1);
    }

    Serial.println("Camera initialized successfully");

    return true;
}

// =====================================================
// เปิดโมเดล TensorFlow Lite
// =====================================================

bool setupMangosteenModel()
{
    Serial.println("Loading mangosteen model...");

    mangosteen_model =
        tflite::GetModel(g_mangosteen_model);

    if (mangosteen_model == nullptr) {
        Serial.println("Cannot read model");
        return false;
    }

    if (mangosteen_model->version() !=
        TFLITE_SCHEMA_VERSION) {

        Serial.printf(
            "Model schema %d, expected %d\n",
            mangosteen_model->version(),
            TFLITE_SCHEMA_VERSION
        );

        return false;
    }

    tensor_arena = (uint8_t *)heap_caps_malloc(
        TENSOR_ARENA_SIZE,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );

    if (tensor_arena == nullptr) {
        Serial.println("Cannot allocate Tensor Arena");
        return false;
    }

    static tflite::AllOpsResolver resolver;

    static tflite::MicroInterpreter interpreter(
        mangosteen_model,
        resolver,
        tensor_arena,
        TENSOR_ARENA_SIZE
    );

    mangosteen_interpreter = &interpreter;

    if (mangosteen_interpreter->AllocateTensors()
        != kTfLiteOk) {

        Serial.println("AllocateTensors failed");
        return false;
    }

    model_input = mangosteen_interpreter->input(0);
    model_output = mangosteen_interpreter->output(0);

    if (model_input == nullptr ||
        model_output == nullptr) {

        Serial.println("Cannot access model tensors");
        return false;
    }

    Serial.println("Model loaded successfully");

    Serial.printf(
        "Input dimensions: %d x %d x %d\n",
        model_input->dims->data[1],
        model_input->dims->data[2],
        model_input->dims->data[3]
    );

    Serial.printf(
        "Input type: %d\n",
        model_input->type
    );

    Serial.printf(
        "Input scale: %.6f\n",
        model_input->params.scale
    );

    Serial.printf(
        "Input zero point: %d\n",
        model_input->params.zero_point
    );

    if (model_input->type != kTfLiteInt8) {
        Serial.println("Model input is not INT8");
        return false;
    }

    if (model_output->type != kTfLiteInt8) {
        Serial.println("Model output is not INT8");
        return false;
    }

    if (model_input->dims->data[1] != MODEL_HEIGHT ||
        model_input->dims->data[2] != MODEL_WIDTH ||
        model_input->dims->data[3] != MODEL_CHANNELS) {

        Serial.println("Model input size is incorrect");
        return false;
    }

    Serial.printf(
        "Tensor Arena remaining PSRAM: %u bytes\n",
        ESP.getFreePsram()
    );

    return true;
}

// =====================================================
// แปลงพิกเซล 0-255 เป็น INT8
// =====================================================

int8_t quantizePixel(uint8_t pixel)
{
    float scale = model_input->params.scale;
    int zero_point = model_input->params.zero_point;

    int32_t quantized = (int32_t)roundf(
        ((float)pixel / scale) + zero_point
    );

    if (quantized < -128) {
        quantized = -128;
    }

    if (quantized > 127) {
        quantized = 127;
    }

    return (int8_t)quantized;
}

// =====================================================
// ถ่ายภาพและจำแนกมังคุด
// =====================================================

bool connectWiFi()
{
    constexpr unsigned long WIFI_TIMEOUT_MS = 20000;

    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    Serial.printf("Connecting to Wi-Fi: %s", WIFI_SSID);
    unsigned long started_at = millis();

    while (WiFi.status() != WL_CONNECTED &&
           millis() - started_at < WIFI_TIMEOUT_MS) {
        delay(500);
        Serial.print(".");
    }

    Serial.println();

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("Wi-Fi connection failed; AI will continue without the web page");
        return false;
    }

    Serial.println("Wi-Fi connected");
    Serial.print("Open the dashboard at: http://");
    Serial.println(WiFi.localIP());
    return true;
}

bool classifyCameraFrame()
{
    unsigned long cycle_start = millis();

    if (mangosteen_interpreter == nullptr ||
        model_input == nullptr ||
        model_output == nullptr) {

        Serial.println("Model is not ready");
        return false;
    }

    camera_fb_t *frame = esp_camera_fb_get();

    if (frame == nullptr) {
        Serial.println("Camera capture failed");
        return false;
    }

    Serial.printf(
        "Captured: %d x %d, %u bytes\n",
        frame->width,
        frame->height,
        frame->len
    );

    // The dashboard displays this exact image after the matching AI result is ready.
    if (frame->format == PIXFORMAT_JPEG) {
        updateLatestCapture(frame->buf, frame->len);
    }

    size_t rgb_buffer_size =
        frame->width * frame->height * 3;

    uint8_t *rgb888 =
        (uint8_t *)heap_caps_malloc(
            rgb_buffer_size,
            MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
        );

    if (rgb888 == nullptr) {
        Serial.println("Cannot allocate RGB buffer");
        esp_camera_fb_return(frame);
        return false;
    }

    bool conversion_success = fmt2rgb888(
        frame->buf,
        frame->len,
        frame->format,
        rgb888
    );

    if (!conversion_success) {
        Serial.println("JPEG to RGB conversion failed");

        heap_caps_free(rgb888);
        esp_camera_fb_return(frame);

        return false;
    }

    int source_width = frame->width;
    int source_height = frame->height;

    // ตัดภาพตรงกลางให้เป็นสี่เหลี่ยม
    int crop_size =
        source_width < source_height
            ? source_width
            : source_height;

    int crop_x =
        (source_width - crop_size) / 2;

    int crop_y =
        (source_height - crop_size) / 2;

    // ย่อภาพตรงกลางเป็น 96x96
    for (int output_y = 0;
         output_y < MODEL_HEIGHT;
         output_y++) {

        int source_y =
            crop_y +
            (output_y * crop_size) / MODEL_HEIGHT;

        for (int output_x = 0;
             output_x < MODEL_WIDTH;
             output_x++) {

            int source_x =
                crop_x +
                (output_x * crop_size) / MODEL_WIDTH;

            int source_index =
                (source_y * source_width + source_x) * 3;

            int model_index =
                (output_y * MODEL_WIDTH + output_x) * 3;

            uint8_t red =
                rgb888[source_index + 0];

            uint8_t green =
                rgb888[source_index + 1];

            uint8_t blue =
                rgb888[source_index + 2];

            model_input->data.int8[
                model_index + 0
            ] = quantizePixel(red);

            model_input->data.int8[
                model_index + 1
            ] = quantizePixel(green);

            model_input->data.int8[
                model_index + 2
            ] = quantizePixel(blue);
        }
    }

    heap_caps_free(rgb888);
    esp_camera_fb_return(frame);

    unsigned long inference_start = millis();

    TfLiteStatus inference_status =
        mangosteen_interpreter->Invoke();

    unsigned long inference_time =
        millis() - inference_start;

    if (inference_status != kTfLiteOk) {
        Serial.println("Model inference failed");
        return false;
    }

    int predicted_class = 0;
    int8_t highest_raw_value =
        model_output->data.int8[0];

    float probabilities[3] = {0.0f, 0.0f, 0.0f};

    float output_scale =
        model_output->params.scale;

    int output_zero_point =
        model_output->params.zero_point;

    Serial.println();
    Serial.println("============================");
    Serial.println("Mangosteen classification");
    Serial.println("============================");

    for (int class_index = 0;
         class_index < 3;
         class_index++) {

        int8_t raw_value =
            model_output->data.int8[class_index];

        float probability =
            (raw_value - output_zero_point) *
            output_scale;

        if (probability < 0.0f) {
            probability = 0.0f;
        }

        if (probability > 1.0f) {
            probability = 1.0f;
        }

        probabilities[class_index] = probability * 100.0f;

        Serial.printf(
            "%s: %.2f%%\n",
            CLASS_NAMES[class_index],
            probabilities[class_index]
        );

        if (raw_value > highest_raw_value) {
            highest_raw_value = raw_value;
            predicted_class = class_index;
        }
    }

    Serial.println("----------------------------");

    Serial.print("Prediction: ");
    Serial.println(CLASS_NAMES[predicted_class]);

    Serial.printf(
        "Inference time: %lu ms\n",
        inference_time
    );

    Serial.println("============================");
    Serial.println();

    updateMangosteenResult(
        CLASS_NAMES[predicted_class],
        probabilities,
        inference_time,
        millis() - cycle_start
    );

    return true;
}

// =====================================================
// Setup
// =====================================================

void setup()
{
    Serial.begin(115200);
    delay(3000);

    Serial.println();
    Serial.println("Mangosteen AI starting...");

    if (!psramFound()) {
        Serial.println("PSRAM not found");

        while (true) {
            delay(1000);
        }
    }

    Serial.printf(
        "Free PSRAM: %u bytes\n",
        ESP.getFreePsram()
    );

    if (!setupCamera()) {
        Serial.println("Camera setup failed");

        while (true) {
            delay(1000);
        }
    }

    if (!setupMangosteenModel()) {
        Serial.println("Model setup failed");

        while (true) {
            delay(1000);
        }
    }

    if (connectWiFi()) {
        startCameraServer();
    }

    setupFruitSensor();

    Serial.println();
    Serial.println("System is ready");
    Serial.println("Waiting for a fruit sensor trigger or dashboard capture request");
    Serial.println();
}

// =====================================================
// Loop
// =====================================================

void loop()
{
    if (fruitSensorTriggered()) {
        delay(SENSOR_SETTLE_TIME_MS);
        requestClassification();
    }

    if (takeClassificationRequest()) {
        classifyCameraFrame();
    }

    delay(10);
}
