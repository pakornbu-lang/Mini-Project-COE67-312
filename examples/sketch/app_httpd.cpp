// Copyright 2015-2016 Espressif Systems (Shanghai) PTE LTD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "esp_http_server.h"
#include "esp_timer.h"
#include "esp_camera.h"
#include "img_converters.h"
#include "camera_index.h"
#include "Arduino.h"
#include <WiFi.h>
#include "esp_heap_caps.h"
#include "sorter_config.h"


static portMUX_TYPE classification_lock = portMUX_INITIALIZER_UNLOCKED;
static char latest_label[16] = "waiting";
static float latest_probabilities[3] = {0.0f, 0.0f, 0.0f};
static unsigned long latest_inference_time_ms = 0;
static unsigned long latest_cycle_time_ms = 0;
static unsigned long latest_updated_at_ms = 0;
static unsigned long total_inspections = 0;
static unsigned long total_inference_time_ms = 0;
static unsigned long class_counts[3] = {0, 0, 0};
static unsigned long dashboard_started_at_ms = 0;
static uint8_t *latest_capture_jpeg = nullptr;
static size_t latest_capture_jpeg_len = 0;


extern void requestClassification();

// Keeps the exact JPEG frame sent to the AI so the dashboard can freeze that image.
void updateLatestCapture(const uint8_t *jpeg, size_t jpeg_len)
{
    uint8_t *copy = (uint8_t *)heap_caps_malloc(
        jpeg_len,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );

    if (copy == nullptr) {
        Serial.println("Cannot store captured image for dashboard");
        return;
    }

    memcpy(copy, jpeg, jpeg_len);

    uint8_t *previous_copy = nullptr;
    portENTER_CRITICAL(&classification_lock);
    previous_copy = latest_capture_jpeg;
    latest_capture_jpeg = copy;
    latest_capture_jpeg_len = jpeg_len;
    portEXIT_CRITICAL(&classification_lock);

    if (previous_copy != nullptr) {
        heap_caps_free(previous_copy);
    }
}


void updateMangosteenResult(
    const char *label,
    const float *probabilities,
    unsigned long inference_time_ms,
    unsigned long cycle_time_ms
)
{
    portENTER_CRITICAL(&classification_lock);
    snprintf(latest_label, sizeof(latest_label), "%s", label);
    for (int index = 0; index < 3; index++) {
        latest_probabilities[index] = probabilities[index];
    }
    latest_inference_time_ms = inference_time_ms;
    latest_cycle_time_ms = cycle_time_ms;
    latest_updated_at_ms = millis();
    total_inspections++;
    total_inference_time_ms += inference_time_ms;
    if (!strcmp(label, "unripe")) {
        class_counts[0]++;
    } else if (!strcmp(label, "ripe")) {
        class_counts[1]++;
    } else if (!strcmp(label, "overripe")) {
        class_counts[2]++;
    }
    portEXIT_CRITICAL(&classification_lock);
}


typedef struct {
    httpd_req_t *req;
    size_t len;
} jpg_chunking_t;

#define PART_BOUNDARY "123456789000000000000987654321"
static const char *_STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *_STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *_STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

httpd_handle_t stream_httpd = NULL;
httpd_handle_t camera_httpd = NULL;


static size_t jpg_encode_stream(void *arg, size_t index, const void *data, size_t len)
{
    jpg_chunking_t *j = (jpg_chunking_t *)arg;
    if (!index) {
        j->len = 0;
    }
    if (httpd_resp_send_chunk(j->req, (const char *)data, len) != ESP_OK) {
        return 0;
    }
    j->len += len;
    return len;
}

static esp_err_t capture_handler(httpd_req_t *req)
{
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;
    int64_t fr_start = esp_timer_get_time();

    fb = esp_camera_fb_get();
    if (!fb) {
        Serial.printf("Camera capture failed");
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");

    size_t fb_len = 0;
    if (fb->format == PIXFORMAT_JPEG) {
        fb_len = fb->len;
        res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
    } else {
        jpg_chunking_t jchunk = {req, 0};
        res = frame2jpg_cb(fb, 80, jpg_encode_stream, &jchunk) ? ESP_OK : ESP_FAIL;
        httpd_resp_send_chunk(req, NULL, 0);
        fb_len = jchunk.len;
    }
    esp_camera_fb_return(fb);
    int64_t fr_end = esp_timer_get_time();
    Serial.printf("JPG: %uB %ums", (uint32_t)(fb_len), (uint32_t)((fr_end - fr_start) / 1000));
    return res;
}

static esp_err_t stream_handler(httpd_req_t *req)
{
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;
    size_t _jpg_buf_len = 0;
    uint8_t *_jpg_buf = NULL;
    char *part_buf[64];

    static int64_t last_frame = 0;
    if (!last_frame) {
        last_frame = esp_timer_get_time();
    }

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) {
        return res;
    }

    while (true) {

        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.printf("Camera capture failed");
            res = ESP_FAIL;
        } else {
            if (fb->format != PIXFORMAT_JPEG) {
                bool jpeg_converted = frame2jpg(fb, 80, &_jpg_buf, &_jpg_buf_len);
                esp_camera_fb_return(fb);
                fb = NULL;
                if (!jpeg_converted) {
                    Serial.printf("JPEG compression failed");
                    res = ESP_FAIL;
                }
            } else {
                _jpg_buf_len = fb->len;
                _jpg_buf = fb->buf;
            }
        }
        if (res == ESP_OK) {
            size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, _jpg_buf_len);
            res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
        }
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
        }
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        }
        if (fb) {
            esp_camera_fb_return(fb);
            fb = NULL;
            _jpg_buf = NULL;
        } else if (_jpg_buf) {
            free(_jpg_buf);
            _jpg_buf = NULL;
        }
        if (res != ESP_OK) {
            break;
        }
        last_frame = esp_timer_get_time();
    }

    last_frame = 0;
    return res;
}

static esp_err_t stream_hmi_handler(httpd_req_t *req)
{
    Serial.println("stream_hmi_handler");
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;
    size_t _jpg_buf_len = 0;
    uint8_t *_jpg_buf = NULL;

    static int64_t last_frame = 0;
    if (!last_frame) {
        last_frame = esp_timer_get_time();
    }
    const char *request = "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    res = httpd_resp_send(req, request, strlen(request));
    if (res != ESP_OK) {
        return res;
    }
    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.printf("Camera capture failed");
            res = ESP_FAIL;
        } else {
            if (fb->format != PIXFORMAT_JPEG) {
                bool jpeg_converted = frame2jpg(fb, 80, &_jpg_buf, &_jpg_buf_len);
                esp_camera_fb_return(fb);
                fb = NULL;
                if (!jpeg_converted) {
                    Serial.printf("JPEG compression failed");
                    res = ESP_FAIL;
                }
            } else {
                _jpg_buf_len = fb->len;
                _jpg_buf = fb->buf;
            }
        }
        if (res == ESP_OK) {
            uint8_t buff[4];
            memcpy(buff, &_jpg_buf_len, sizeof(_jpg_buf_len));
            res = httpd_send(req, (const char *)buff, sizeof(buff));
        }
        if (fb) {
            esp_camera_fb_return(fb);
            fb = NULL;
            _jpg_buf = NULL;
        } else if (_jpg_buf) {
            free(_jpg_buf);
            _jpg_buf = NULL;
        }
        if (res != ESP_OK) {
            break;
        }
    }
    last_frame = 0;
    return res;
}

static esp_err_t cmd_handler(httpd_req_t *req)
{
    char  *buf;
    size_t buf_len;
    char variable[32] = {0,};
    char value[32] = {0,};

    buf_len = httpd_req_get_url_query_len(req) + 1;
    if (buf_len > 1) {
        buf = (char *)malloc(buf_len);
        if (!buf) {
            httpd_resp_send_500(req);
            return ESP_FAIL;
        }
        if (httpd_req_get_url_query_str(req, buf, buf_len) == ESP_OK) {
            if (httpd_query_key_value(buf, "var", variable, sizeof(variable)) == ESP_OK &&
                    httpd_query_key_value(buf, "val", value, sizeof(value)) == ESP_OK) {
            } else {
                free(buf);
                httpd_resp_send_404(req);
                return ESP_FAIL;
            }
        } else {
            free(buf);
            httpd_resp_send_404(req);
            return ESP_FAIL;
        }
        free(buf);
    } else {
        httpd_resp_send_404(req);
        return ESP_FAIL;
    }

    int val = atoi(value);
    sensor_t *s = esp_camera_sensor_get();
    int res = 0;

    if (!strcmp(variable, "framesize")) {
        if (s->pixformat == PIXFORMAT_JPEG) res = s->set_framesize(s, (framesize_t)val);
    } else if (!strcmp(variable, "quality")) res = s->set_quality(s, val);
    else if (!strcmp(variable, "contrast")) res = s->set_contrast(s, val);
    else if (!strcmp(variable, "brightness")) res = s->set_brightness(s, val);
    else if (!strcmp(variable, "saturation")) res = s->set_saturation(s, val);
    else if (!strcmp(variable, "gainceiling")) res = s->set_gainceiling(s, (gainceiling_t)val);
    else if (!strcmp(variable, "colorbar")) res = s->set_colorbar(s, val);
    else if (!strcmp(variable, "awb")) res = s->set_whitebal(s, val);
    else if (!strcmp(variable, "agc")) res = s->set_gain_ctrl(s, val);
    else if (!strcmp(variable, "aec")) res = s->set_exposure_ctrl(s, val);
    else if (!strcmp(variable, "hmirror")) res = s->set_hmirror(s, val);
    else if (!strcmp(variable, "vflip")) res = s->set_vflip(s, val);
    else if (!strcmp(variable, "awb_gain")) res = s->set_awb_gain(s, val);
    else if (!strcmp(variable, "agc_gain")) res = s->set_agc_gain(s, val);
    else if (!strcmp(variable, "aec_value")) res = s->set_aec_value(s, val);
    else if (!strcmp(variable, "aec2")) res = s->set_aec2(s, val);
    else if (!strcmp(variable, "dcw")) res = s->set_dcw(s, val);
    else if (!strcmp(variable, "bpc")) res = s->set_bpc(s, val);
    else if (!strcmp(variable, "wpc")) res = s->set_wpc(s, val);
    else if (!strcmp(variable, "raw_gma")) res = s->set_raw_gma(s, val);
    else if (!strcmp(variable, "lenc")) res = s->set_lenc(s, val);
    else if (!strcmp(variable, "special_effect")) res = s->set_special_effect(s, val);
    else if (!strcmp(variable, "wb_mode")) res = s->set_wb_mode(s, val);
    else if (!strcmp(variable, "ae_level")) res = s->set_ae_level(s, val);
    else {
        res = -1;
    }

    if (res) {
        return httpd_resp_send_500(req);
    }

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, NULL, 0);
}

static esp_err_t status_handler(httpd_req_t *req)
{
    static char json_response[1024];

    sensor_t *s = esp_camera_sensor_get();
    char *p = json_response;
    *p++ = '{';

    p += sprintf(p, "\"framesize\":%u,", s->status.framesize);
    p += sprintf(p, "\"quality\":%u,", s->status.quality);
    p += sprintf(p, "\"brightness\":%d,", s->status.brightness);
    p += sprintf(p, "\"contrast\":%d,", s->status.contrast);
    p += sprintf(p, "\"saturation\":%d,", s->status.saturation);
    p += sprintf(p, "\"special_effect\":%u,", s->status.special_effect);
    p += sprintf(p, "\"wb_mode\":%u,", s->status.wb_mode);
    p += sprintf(p, "\"awb\":%u,", s->status.awb);
    p += sprintf(p, "\"awb_gain\":%u,", s->status.awb_gain);
    p += sprintf(p, "\"aec\":%u,", s->status.aec);
    p += sprintf(p, "\"aec2\":%u,", s->status.aec2);
    p += sprintf(p, "\"ae_level\":%d,", s->status.ae_level);
    p += sprintf(p, "\"aec_value\":%u,", s->status.aec_value);
    p += sprintf(p, "\"agc\":%u,", s->status.agc);
    p += sprintf(p, "\"agc_gain\":%u,", s->status.agc_gain);
    p += sprintf(p, "\"gainceiling\":%u,", s->status.gainceiling);
    p += sprintf(p, "\"bpc\":%u,", s->status.bpc);
    p += sprintf(p, "\"wpc\":%u,", s->status.wpc);
    p += sprintf(p, "\"raw_gma\":%u,", s->status.raw_gma);
    p += sprintf(p, "\"lenc\":%u,", s->status.lenc);
    p += sprintf(p, "\"hmirror\":%u,", s->status.hmirror);
    p += sprintf(p, "\"dcw\":%u,", s->status.dcw);
    p += sprintf(p, "\"colorbar\":%u", s->status.colorbar);
    *p++ = '}';
    *p++ = 0;
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json_response, strlen(json_response));
}

static esp_err_t mangosteen_result_handler(httpd_req_t *req)
{
    char label[sizeof(latest_label)];
    float probabilities[3];
    unsigned long inference_time_ms;
    unsigned long cycle_time_ms;
    unsigned long updated_at_ms;
    unsigned long inspection_count;
    unsigned long average_inference_time_ms;
    unsigned long unripe_count;
    unsigned long ripe_count;
    unsigned long overripe_count;

    portENTER_CRITICAL(&classification_lock);
    snprintf(label, sizeof(label), "%s", latest_label);
    for (int index = 0; index < 3; index++) {
        probabilities[index] = latest_probabilities[index];
    }
    inference_time_ms = latest_inference_time_ms;
    cycle_time_ms = latest_cycle_time_ms;
    updated_at_ms = latest_updated_at_ms;
    inspection_count = total_inspections;
    average_inference_time_ms =
        inspection_count == 0 ? 0 :
        total_inference_time_ms / inspection_count;
    unripe_count = class_counts[0];
    ripe_count = class_counts[1];
    overripe_count = class_counts[2];
    portEXIT_CRITICAL(&classification_lock);

    unsigned long uptime_ms =
        dashboard_started_at_ms == 0 ? 0 :
        millis() - dashboard_started_at_ms;

    char json[512];
    snprintf(
        json,
        sizeof(json),
        "{\"label\":\"%s\",\"probabilities\":[%.2f,%.2f,%.2f],\"inferenceMs\":%lu,\"cycleMs\":%lu,\"updatedAtMs\":%lu,\"inspectionCount\":%lu,\"averageInferenceMs\":%lu,\"classCounts\":[%lu,%lu,%lu],\"uptimeMs\":%lu,\"freeHeap\":%u,\"freePsram\":%u,\"wifiRssi\":%d,\"sensorEnabled\":%s}",
        label,
        probabilities[0],
        probabilities[1],
        probabilities[2],
        inference_time_ms,
        cycle_time_ms,
        updated_at_ms,
        inspection_count,
        average_inference_time_ms,
        unripe_count,
        ripe_count,
        overripe_count,
        uptime_ms,
        ESP.getFreeHeap(),
        ESP.getFreePsram(),
        WiFi.RSSI(),
        FRUIT_SENSOR_PIN >= 0 ? "true" : "false"
    );

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    return httpd_resp_send(req, json, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t capture_request_handler(httpd_req_t *req)
{
    requestClassification();
    httpd_resp_set_status(req, "202 Accepted");
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(
        req,
        "{\"message\":\"Capture request queued\"}",
        HTTPD_RESP_USE_STRLEN
    );
}

static esp_err_t latest_capture_handler(httpd_req_t *req)
{
    uint8_t *image_copy = nullptr;
    size_t image_len = 0;

    portENTER_CRITICAL(&classification_lock);
    if (latest_capture_jpeg != nullptr && latest_capture_jpeg_len > 0) {
        image_len = latest_capture_jpeg_len;
        image_copy = (uint8_t *)heap_caps_malloc(
            image_len,
            MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
        );
        if (image_copy != nullptr) {
            memcpy(image_copy, latest_capture_jpeg, image_len);
        }
    }
    portEXIT_CRITICAL(&classification_lock);

    if (image_copy == nullptr) {
        httpd_resp_send_404(req);
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    esp_err_t result = httpd_resp_send(
        req,
        (const char *)image_copy,
        image_len
    );
    heap_caps_free(image_copy);
    return result;
}

static esp_err_t index_handler(httpd_req_t *req)
{
    static const char MANGOSTEEN_DASHBOARD[] PROGMEM = R"HTML(
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Mangosteen Sorting Station</title>
  <style>
    :root{--ink:#172520;--muted:#6a7773;--line:#e7e2d9;--paper:#fffdf9;--ground:#f5f2eb;--leaf:#159567;--leaf-soft:#e9f7ef;--amber:#e9a039;--berry:#742044;--berry-soft:#f8edf2;--night:#152820}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 85% -10%,#e2f1e8 0,transparent 31%),linear-gradient(135deg,#f8f6f0,#f1f6f3);color:var(--ink);font-family:Arial,"Noto Sans Thai",sans-serif}main{max-width:1180px;margin:auto;padding:clamp(18px,3vw,36px)}
    header{display:flex;justify-content:space-between;gap:20px;align-items:center;margin:0 0 22px}.eyebrow{display:flex;align-items:center;gap:7px;color:var(--leaf);font-size:11px;font-weight:800;letter-spacing:1.8px}.eyebrow:before{content:"";width:24px;height:3px;border-radius:9px;background:var(--leaf)}.title{margin:8px 0 5px;font-size:clamp(25px,3.2vw,40px);line-height:1.05;letter-spacing:-1.5px}.title span{color:var(--berry)}.subtitle{margin:0;color:var(--muted);font-size:15px}.badge{border:1px solid #a4d9c0;border-radius:99px;background:var(--leaf-soft);color:#08704a;padding:9px 13px;font-size:11px;font-weight:800;letter-spacing:.3px;text-align:center;white-space:nowrap}
    .top{display:grid;grid-template-columns:1.08fr .92fr;gap:18px;align-items:stretch}.panel{background:var(--paper);border:1px solid var(--line);border-radius:20px;padding:18px;box-shadow:0 16px 40px #2a3f3310}.camera-panel{padding:14px}.result-panel{display:flex;flex-direction:column}.panel-title{font-size:11px;color:var(--muted);font-weight:800;letter-spacing:1.15px;text-transform:uppercase}
    .camera{position:relative;margin-top:10px;aspect-ratio:4/3;background:var(--night);border-radius:14px;overflow:hidden}.camera:after{content:"";position:absolute;inset:0;border:1px solid #ffffff25;border-radius:14px;pointer-events:none}.camera img{display:block;width:100%;height:100%;object-fit:cover}.empty{position:absolute;inset:0;z-index:2;display:flex;align-items:center;justify-content:center;text-align:center;padding:20px;color:#e6f2ec;font-size:16px;background:#152820cc;backdrop-filter:blur(2px)}.empty[hidden]{display:none}.empty small{display:block;margin-top:7px;color:#a2b8ad;font-size:13px}.snapshot{position:absolute;z-index:3;top:12px;left:12px;border-radius:99px;background:#10251de8;color:#e6f7ed;padding:7px 10px;font-size:11px;font-weight:800;letter-spacing:.25px}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#45e3a0;margin-right:5px;box-shadow:0 0 0 3px #45e3a033}
    .result{font-size:clamp(32px,4.1vw,51px);font-weight:900;letter-spacing:-1.4px;line-height:1;margin:11px 0 9px}.waiting{color:var(--muted)}.unripe{color:var(--leaf)}.ripe{color:var(--amber)}.overripe{color:var(--berry)}.status{min-height:42px;border-left:3px solid #c9d5cf;padding:3px 0 3px 10px;color:var(--muted);font-size:13px;line-height:1.5}.action{margin:17px 0 7px;width:100%;border:0;border-radius:12px;padding:15px 14px;background:linear-gradient(135deg,#1b332a,#0c2019);box-shadow:0 8px 16px #1235252b;color:#fff;font-weight:800;font-size:15px;cursor:pointer;transition:transform .15s,box-shadow .15s}.action:active{transform:translateY(1px);box-shadow:none}.action:disabled{opacity:.55;cursor:wait}
    .row{margin:17px 0 0}.row-head{display:flex;justify-content:space-between;gap:12px;font-size:13px;font-weight:700}.row-head span{color:#40514b}.bar{height:9px;background:#edf0ed;border-radius:99px;overflow:hidden;margin-top:7px}.fill{height:100%;width:0;border-radius:99px;transition:width .3s}.unripe-fill{background:linear-gradient(90deg,#2db784,var(--leaf))}.ripe-fill{background:linear-gradient(90deg,#f5bf62,var(--amber))}.overripe-fill{background:linear-gradient(90deg,#a74170,var(--berry))}
    .bench{margin-top:18px;padding:20px}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:14px}.metric{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:13px;background:#fffefa;padding:13px;min-height:83px}.metric:before{content:"";position:absolute;top:0;left:0;width:34px;height:3px;background:#cfdcd6}.metric span{display:block;color:var(--muted);font-size:11px;line-height:1.3}.metric strong{display:block;font-size:20px;line-height:1.1;margin-top:9px;letter-spacing:-.4px}.wide{grid-column:span 2}.counts{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}.count{border-left:4px solid var(--leaf);border-radius:9px;background:#f0faf4;padding:11px 12px;font-size:11px;color:#4e6259}.count strong{display:block;color:var(--ink);font-size:22px;line-height:1;margin-top:6px}.count.ripe-count{border-color:var(--amber);background:#fff8ec}.count.overripe-count{border-color:var(--berry);background:var(--berry-soft)}
    @media(max-width:760px){main{padding:16px}header{align-items:flex-start;flex-direction:column;gap:12px}.title{font-size:29px}.badge{white-space:normal}.top{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.wide{grid-column:span 1}.counts{grid-template-columns:1fr}.camera{aspect-ratio:4/3}}@media(min-width:761px) and (max-width:1000px){.metrics{grid-template-columns:repeat(3,minmax(0,1fr))}}
  </style>
</head>
<body>
  <main>
    <header>
      <div><div class="eyebrow">ESP32 CAMERA + TENSORFLOW LITE</div><h1 class="title">Mangosteen <span>Sorting Station</span></h1><p class="subtitle">AI fruit grading dashboard · ระบบคัดเกรดมังคุดด้วย AI</p></div>
      <div id="mode" class="badge">WEB TRIGGER MODE (โหมดกดจากเว็บ)</div>
    </header>
    <section class="top">
      <div class="panel camera-panel">
        <div class="panel-title">Camera View (มุมมองกล้อง)</div>
        <div class="camera"><img id="photo" alt="Mangosteen camera view"><div id="empty" class="empty">Starting camera video<br><small>กำลังเปิดวิดีโอจากกล้อง</small></div><div id="view-badge" class="snapshot"><span class="dot"></span>LIVE VIDEO (วิดีโอสด)</div></div>
      </div>
      <div class="panel result-panel">
        <div class="panel-title">Latest Inspection (ผลการตรวจล่าสุด)</div>
        <div id="result" class="result waiting">WAITING</div>
        <div id="status" class="status">Live video ready. Position the fruit, then capture. (วิดีโอพร้อม จัดวางมังคุดแล้วกดถ่าย)</div>
        <button id="capture" class="action">Capture &amp; Classify (ถ่ายภาพและตรวจ)</button>
        <div class="row"><div class="row-head"><span>UNRIPE (ดิบ)</span><strong id="unripe-value">0.00%</strong></div><div class="bar"><div id="unripe-bar" class="fill unripe-fill"></div></div></div>
        <div class="row"><div class="row-head"><span>RIPE (สุกพอดี)</span><strong id="ripe-value">0.00%</strong></div><div class="bar"><div id="ripe-bar" class="fill ripe-fill"></div></div></div>
        <div class="row"><div class="row-head"><span>OVERRIPE (สุกเกิน)</span><strong id="overripe-value">0.00%</strong></div><div class="bar"><div id="overripe-bar" class="fill overripe-fill"></div></div></div>
      </div>
    </section>
    <section class="panel bench">
      <div class="panel-title">Live Benchmark (ข้อมูลประสิทธิภาพสด)</div>
      <div class="metrics">
        <div class="metric"><span>AI inference (เวลา AI)</span><strong id="inference">—</strong></div>
        <div class="metric"><span>Capture-to-result (ถ่ายถึงผล)</span><strong id="cycle">—</strong></div>
        <div class="metric"><span>Average inference (เวลาเฉลี่ย)</span><strong id="average">—</strong></div>
        <div class="metric"><span>Confidence (ความมั่นใจ)</span><strong id="confidence">—</strong></div>
        <div class="metric"><span>Inspections (จำนวนตรวจ)</span><strong id="inspections">0</strong></div>
        <div class="metric"><span>Wi-Fi signal (สัญญาณ)</span><strong id="wifi">—</strong></div>
        <div class="metric"><span>Stream speed (ความเร็ววิดีโอ)</span><strong id="stream-fps">—</strong></div>
        <div class="metric wide"><span>Free PSRAM (หน่วยความจำว่าง)</span><strong id="psram">—</strong></div>
        <div class="metric wide"><span>System uptime (เวลาที่เปิดระบบ)</span><strong id="uptime">—</strong></div>
      </div>
      <div class="counts">
        <div class="count"><span>UNRIPE total (ดิบทั้งหมด)</span><strong id="unripe-count">0</strong></div>
        <div class="count ripe-count"><span>RIPE total (สุกพอดีทั้งหมด)</span><strong id="ripe-count">0</strong></div>
        <div class="count overripe-count"><span>OVERRIPE total (สุกเกินทั้งหมด)</span><strong id="overripe-count">0</strong></div>
      </div>
    </section>
  </main>
  <script>
    const keys=['unripe','ripe','overripe'];
    const photo=document.getElementById('photo'),empty=document.getElementById('empty'),capture=document.getElementById('capture'),viewBadge=document.getElementById('view-badge');
    let captureQueued=false,lastResultAt=0,captureBaseline=0,lastPhotoAt=0,viewMode='live',sensorEnabled=false,frames=0,lastFrameAt=Date.now();
    const text=(id,value)=>document.getElementById(id).textContent=value;
    const millis=(value)=>value ? value+' ms':'—';
    const uptime=(value)=>{const seconds=Math.floor(value/1000);return Math.floor(seconds/60)+'m '+(seconds%60)+'s'};
    function setModeLabel(){
      const trigger=sensorEnabled?'SENSOR TRIGGER (เซ็นเซอร์)':'WEB TRIGGER (กดจากเว็บ)';
      const view=viewMode==='live'?'LIVE VIDEO (วิดีโอสด)':'FROZEN IMAGE (ภาพนิ่ง)';
      text('mode',trigger+' · '+view);
    }
    function showLiveVideo(){
      viewMode='live';frames=0;lastFrameAt=Date.now();empty.hidden=false;
      empty.innerHTML='Starting live video<br><small>กำลังเปิดวิดีโอจากกล้อง</small>';
      viewBadge.innerHTML='<span class="dot"></span>LIVE VIDEO (วิดีโอสด)';
      photo.src=location.protocol+'//'+location.hostname+':81/stream?t='+Date.now();
      capture.disabled=false;capture.textContent='Capture & Classify (ถ่ายภาพและตรวจ)';
      text('status','Live video ready. Position the fruit, then capture. (วิดีโอพร้อม จัดวางมังคุดแล้วกดถ่าย)');
      setModeLabel();
    }
    function freezeCapturedImage(timestamp){
      viewMode='frozen';frames=0;text('stream-fps','Paused (ภาพค้าง)');
      viewBadge.innerHTML='<span class="dot"></span>FROZEN IMAGE (ภาพนิ่ง)';
      photo.src='/api/last-capture?t='+timestamp;empty.hidden=true;
      capture.disabled=false;capture.textContent='New Inspection (ตรวจลูกใหม่)';
      setModeLabel();
    }
    photo.onload=()=>{empty.hidden=true;if(viewMode==='live')frames++;};
    setInterval(()=>{if(viewMode==='live'){const now=Date.now();text('stream-fps',(frames*1000/(now-lastFrameAt)).toFixed(1)+' FPS');frames=0;lastFrameAt=now;}},1000);
    function render(data){
      const result=document.getElementById('result'),label=(data.label||'waiting');
      result.textContent=label.toUpperCase();result.className='result '+label;
      keys.forEach((key,index)=>{const value=Math.max(0,Math.min(100,(data.probabilities||[])[index]||0));text(key+'-value',value.toFixed(2)+'%');document.getElementById(key+'-bar').style.width=value+'%';});
      const confidence=Math.max(...(data.probabilities||[0]));
      text('inference',millis(data.inferenceMs));text('cycle',millis(data.cycleMs));text('average',millis(data.averageInferenceMs));text('confidence',confidence.toFixed(2)+'%');
      text('inspections',data.inspectionCount||0);text('wifi',data.wifiRssi+' dBm');text('psram',((data.freePsram||0)/1048576).toFixed(2)+' MB');text('uptime',uptime(data.uptimeMs||0));
      const counts=data.classCounts||[0,0,0];text('unripe-count',counts[0]);text('ripe-count',counts[1]);text('overripe-count',counts[2]);
      sensorEnabled=Boolean(data.sensorEnabled);setModeLabel();
      if(data.updatedAtMs>lastResultAt){
        const wasInitialResult=lastResultAt===0;
        lastResultAt=data.updatedAtMs;lastPhotoAt=data.updatedAtMs;
        if(captureQueued||!wasInitialResult)freezeCapturedImage(data.updatedAtMs);
      }
      if(captureQueued&&data.updatedAtMs>captureBaseline){captureQueued=false;text('status','Inspection complete. Image frozen for review. (ตรวจเสร็จแล้ว ภาพถูกค้างไว้ให้ตรวจสอบ)');}
    }
    async function refresh(){try{const response=await fetch('/api/result',{cache:'no-store'});render(await response.json());}catch(error){text('status','Dashboard connection lost.');}}
    capture.onclick=async()=>{if(viewMode==='frozen'){showLiveVideo();return;}capture.disabled=true;captureQueued=true;captureBaseline=lastResultAt;text('status','Capturing image and running AI... (กำลังถ่ายภาพและให้ AI ตรวจ)');try{const response=await fetch('/api/capture',{method:'POST'});if(!response.ok)throw new Error('Request failed');}catch(error){captureQueued=false;capture.disabled=false;text('status','Could not queue capture. Please try again. (สั่งถ่ายภาพไม่สำเร็จ)');}};
    showLiveVideo();
    refresh();setInterval(refresh,700);
  </script>
</body>
</html>
)HTML";
    httpd_resp_set_type(req, "text/html");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    return httpd_resp_send(req, MANGOSTEEN_DASHBOARD, HTTPD_RESP_USE_STRLEN);
}

void startCameraServer()
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    dashboard_started_at_ms = millis();

    httpd_uri_t index_uri = {
        .uri       = "/",
        .method    = HTTP_GET,
        .handler   = index_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t status_uri = {
        .uri       = "/status",
        .method    = HTTP_GET,
        .handler   = status_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t cmd_uri = {
        .uri       = "/control",
        .method    = HTTP_GET,
        .handler   = cmd_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t capture_uri = {
        .uri       = "/capture",
        .method    = HTTP_GET,
        .handler   = capture_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t mangosteen_result_uri = {
        .uri       = "/api/result",
        .method    = HTTP_GET,
        .handler   = mangosteen_result_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t capture_request_uri = {
        .uri       = "/api/capture",
        .method    = HTTP_POST,
        .handler   = capture_request_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t latest_capture_uri = {
        .uri       = "/api/last-capture",
        .method    = HTTP_GET,
        .handler   = latest_capture_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t stream_uri = {
        .uri       = "/stream",
        .method    = HTTP_GET,
        .handler   = stream_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t hmi_uri = {
        .uri       = "/hmi",
        .method    = HTTP_GET,
        .handler   = stream_hmi_handler,
        .user_ctx  = NULL
    };

    Serial.printf("Starting web server on port: '%d'\n", config.server_port);
    if (httpd_start(&camera_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(camera_httpd, &index_uri);
        httpd_register_uri_handler(camera_httpd, &cmd_uri);
        httpd_register_uri_handler(camera_httpd, &status_uri);
        httpd_register_uri_handler(camera_httpd, &capture_uri);
        httpd_register_uri_handler(camera_httpd, &mangosteen_result_uri);
        httpd_register_uri_handler(camera_httpd, &capture_request_uri);
        httpd_register_uri_handler(camera_httpd, &latest_capture_uri);
    }

    config.server_port += 1;
    config.ctrl_port += 1;
    Serial.printf("Starting stream server on port: '%d'\n", config.server_port);
    if (httpd_start(&stream_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(stream_httpd, &stream_uri);
        httpd_register_uri_handler(stream_httpd, &hmi_uri);
    }
}
