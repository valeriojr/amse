#include <esp_camera.h>
#include <esp_log.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <tensorflow/lite/micro/micro_common.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>


#define TAG "main"

#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM (-1)
#define XCLK_GPIO_NUM  0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM  5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

#define GAS_UP_PIN GPIO_NUM_1
#define GAS_DOWN_PIN GPIO_NUM_2
#define STEER_LEFT_PIN GPIO_NUM_3
#define STEER_RIGHT_PIN GPIO_NUM_4


typedef tflite::MicroMutableOpResolver<7> CarlosOpResolver;

typedef enum {
    STEER_NONE = 0,
    STEER_LEFT = -1,
    STEER_RIGHT = 1,
} steer_action_t;

typedef enum {
    GAS_NONE = 0,
    GAS_DOWN = -1,
    GAS_UP = 1,
} gas_action_t;


extern const unsigned char actor_tflite[];
static camera_config_t camera_config = {
        .pin_pwdn = PWDN_GPIO_NUM,
        .pin_reset = RESET_GPIO_NUM,
        .pin_xclk = XCLK_GPIO_NUM,
        .pin_sccb_sda = SIOD_GPIO_NUM,
        .pin_sccb_scl = SIOC_GPIO_NUM,
        .pin_d7 = Y9_GPIO_NUM,
        .pin_d6 = Y8_GPIO_NUM,
        .pin_d5 = Y7_GPIO_NUM,
        .pin_d4 = Y6_GPIO_NUM,
        .pin_d3 = Y5_GPIO_NUM,
        .pin_d2 = Y4_GPIO_NUM,
        .pin_d1 = Y3_GPIO_NUM,
        .pin_d0 = Y2_GPIO_NUM,
        .pin_vsync = VSYNC_GPIO_NUM,
        .pin_href = HREF_GPIO_NUM,
        .pin_pclk = PCLK_GPIO_NUM,
        .xclk_freq_hz = 20000000,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_JPEG,
        .frame_size = FRAMESIZE_QQVGA,
        .jpeg_quality = 10,
        .fb_count = 2,
};
static uint8_t* frame_rgb_buffer = nullptr;


static struct {
    struct {
        TfLiteTensor *camera;
        TfLiteTensor *speed;
        TfLiteTensor *steer;
    } inputs;

    struct {
        TfLiteTensor *gas;
        TfLiteTensor *steer;
    } outputs;
} actor;


static esp_err_t add_operations(CarlosOpResolver &op_resolver) {
    TfLiteStatus status;

    status = op_resolver.AddFullyConnected();
    if (status == kTfLiteError) {
        ESP_LOGE(TAG, "op_resolver.AddFullyConnected() failed");
        return ESP_ERR_NOT_SUPPORTED;
    }

    status = op_resolver.AddRelu();
    if (status == kTfLiteError) {
        ESP_LOGE(TAG, "op_resolver.AddRelu() failed");
        return ESP_ERR_NOT_SUPPORTED;
    }

    status = op_resolver.AddConv2D();
    if (status == kTfLiteError) {
        ESP_LOGE(TAG, "op_resolver.AddConv2D() failed");
        return ESP_ERR_NOT_SUPPORTED;
    }

    status = op_resolver.AddMaxPool2D();
    if (status == kTfLiteError) {
        ESP_LOGE(TAG, "op_resolver.AddMaxPool2D() failed");
        return ESP_ERR_NOT_SUPPORTED;
    }

    status = op_resolver.AddReshape();
    if (status == kTfLiteError) {
        ESP_LOGE(TAG, "op_resolver.AddReshape() failed");
        return ESP_ERR_NOT_SUPPORTED;
    }

    status = op_resolver.AddConcatenation();
    if (status == kTfLiteError) {
        ESP_LOGE(TAG, "op_resolver.AddConcatenation() failed");
        return ESP_ERR_NOT_SUPPORTED;
    }

    status = op_resolver.AddSoftmax();
    if (status == kTfLiteError) {
        ESP_LOGE(TAG, "op_resolver.AddSoftmax() failed");
        return ESP_ERR_NOT_SUPPORTED;
    }

    return ESP_OK;
}


static esp_err_t init_gpio() {
    gpio_config_t led_gpio_config = {
        .pin_bit_mask = (1ULL << GAS_UP_PIN) | (1ULL << GAS_DOWN_PIN) | (1ULL << STEER_LEFT_PIN) | (1ULL << STEER_RIGHT_PIN),
        .mode = GPIO_MODE_OUTPUT,
    };

    gpio_config(&led_gpio_config);
}


static void preprocess_input(camera_fb_t *frame, float speed, float steer) {
    memcpy(actor.inputs.camera->data.data, &speed, sizeof(float));
    memcpy(actor.inputs.steer->data.data, &steer, sizeof(float));

    bool success = fmt2rgb888(frame->buf, frame->len, frame->format, frame_rgb_buffer);
    if (success) {
        for(int i = 0;i < 96 * 96 * 3;i++) {
            actor.inputs.camera->data.f[i] = (float) frame_rgb_buffer[i] / 255.0f;
        }
    }
    else {
        ESP_LOGE("preprocess_input", "fmt2rgb888() failed");
    }
}


static int sample(float *distribution, int n) {
    float r;
    float cumulative_distribution[n];

    memcpy(cumulative_distribution, distribution, n * sizeof(float));

    for (int i = 1; i < n; i++) {
        cumulative_distribution[i] += cumulative_distribution[i - 1];
    }

    r = (float) rand() / (float) INT_MAX;

    for (int i = 0; i < n; i++) {
        if (r < cumulative_distribution[i]) {
            return i;
        }
    }

    return n - 1;
}


static void postprocess_output(gas_action_t* gas_action, steer_action_t* steer_action) {
    *gas_action = (gas_action_t) (sample(actor.outputs.gas->data.f, 3) - 1);
    *steer_action = (steer_action_t) (sample(actor.outputs.steer->data.f, 3) - 1);

    char steer_left = ' ';
    char steer_right = ' ';
    char speed_up = ' ';
    char speed_down = ' ';

    switch (*gas_action) {
        case GAS_DOWN:
            speed_up = 'V';
            gpio_set_level(GAS_UP_PIN, 1);
            gpio_set_level(GAS_DOWN_PIN, 0);
            break;
        case GAS_NONE:
            gpio_set_level(GAS_UP_PIN, 0);
            gpio_set_level(GAS_DOWN_PIN, 0);
            break;
        case GAS_UP:
            speed_down = 'A';
            gpio_set_level(GAS_UP_PIN, 0);
            gpio_set_level(GAS_DOWN_PIN, 1);
            break;
        default:
            break;
    }

    switch (*steer_action) {
        case STEER_LEFT:
            steer_left = '<';
            gpio_set_level(STEER_LEFT_PIN, 1);
            gpio_set_level(STEER_RIGHT_PIN, 0);
            break;
        case STEER_NONE:
            gpio_set_level(STEER_LEFT_PIN, 0);
            gpio_set_level(STEER_RIGHT_PIN, 0);
            break;
        case STEER_RIGHT:
            steer_right = '>';
            gpio_set_level(STEER_LEFT_PIN, 0);
            gpio_set_level(STEER_RIGHT_PIN, 1);
            break;
        default:
            break;
    }

    ESP_LOGI(TAG, "%c %c\t%c %c", steer_left, steer_right, speed_up, speed_down);
}


#ifdef __cplusplus
extern "C" {
#endif
void app_main(void) {
    /* Inputs */
    float current_speed = 0.0f;
    float current_steer = 0.0f;

    TfLiteStatus status;
    camera_fb_t *frame = nullptr;
    const size_t tensor_arena_size = 1000000;
    uint8_t *tensor_arena = (uint8_t *) malloc(tensor_arena_size);
    const tflite::Model *model = tflite::GetModel(actor_tflite);
    CarlosOpResolver op_resolver;
    tflite::MicroInterpreter interpreter = tflite::MicroInterpreter(model, op_resolver, tensor_arena,
                                                                    tensor_arena_size);

    esp_camera_init(&camera_config);

    frame_rgb_buffer = (uint8_t*) malloc(2 * 96 * 96 * 3);

    ESP_LOGI(TAG, "Initializing tflite");

    esp_err_t err = add_operations(op_resolver);
    if (err) {
        ESP_LOGE(TAG, "add_operations() failed");
        return;
    }

    status = interpreter.AllocateTensors();
    if (status == kTfLiteError) {
        ESP_LOGE(TAG, "interpreter.AllocateTensors() failed");
        return;
    }

    actor.inputs.camera = interpreter.input(0);
    actor.inputs.speed = interpreter.input(1);
    actor.inputs.steer = interpreter.input(2);

    actor.outputs.gas = interpreter.output(0);
    actor.outputs.steer = interpreter.output(1);

    ESP_LOGI(TAG, "tflite ok");

    while (true) {
        frame = esp_camera_fb_get();

        if (frame != nullptr) {
            preprocess_input(frame, current_speed, current_steer);

            uint32_t start = esp_log_timestamp();
            status = interpreter.Invoke();
            if (status == kTfLiteError) {
                ESP_LOGE(TAG, "interpreter.Invoke() failed");
                break;
            }
            uint32_t finish = esp_log_timestamp();
            ESP_LOGI("interpreter.Invoke()", "%ldms", finish - start);

            gas_action_t gas_action;
            steer_action_t steer_action;
            postprocess_output(&gas_action, &steer_action);

            // controller_update(gas_action, steer_action, &current_speed, &current_steer);

            esp_camera_fb_return(frame);
        }
        else {
            ESP_LOGE(TAG, "esp_camera_fb_get() failed");
        }

        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
#ifdef __cplusplus
};
#endif