#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "esp_bit_defs.h"
#include "esp_event.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>

#include "config.h"
#include "net.h"
#include "train.h"
#include "weights.h"

#define BS 4

static EventGroupHandle_t s_wifi_event_group;

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data) {
  if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
    esp_wifi_connect();
  } else if (event_base == WIFI_EVENT &&
             event_id == WIFI_EVENT_STA_DISCONNECTED) {
    esp_wifi_connect();
    xEventGroupClearBits(s_wifi_event_group, BIT0);
  } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
    ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
    printf("got ip %d.%d.%d.%d\n", IP2STR(&event->ip_info.ip));
    xEventGroupSetBits(s_wifi_event_group, BIT0);
  }
}

static void fetch_batch(const int sock, const int index, int bs, uint8 *batch,
                        uint8 *labels) {
  // send index and bs
  char tx_buffer[8];
  memcpy(tx_buffer, &index, sizeof(index));
  memcpy(tx_buffer + sizeof(index), &bs, sizeof(bs));
  send(sock, tx_buffer, sizeof(index) + sizeof(bs), 0);

  static uint8 rx_buffer[4096];
  int len = recv(sock, rx_buffer, sizeof(rx_buffer) - 1, 0);
  if (len < 0) {
    printf("recv failed\n");
    exit(1);
  } else {
    rx_buffer[len] = 0;
  }

  bs = bs == 0 ? 1 : bs;
  memcpy(batch, rx_buffer, sizeof(uint8) * bs * 28 * 28);
  memcpy(labels, rx_buffer + sizeof(uint8) * bs * 28 * 28,
         sizeof(uint8) * bs * 1);
}

void app_main(void) {
  srand(time(NULL));

  // init nvs
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ret = nvs_flash_init();
  }
  ESP_ERROR_CHECK(ret);

  // init event loop
  s_wifi_event_group = xEventGroupCreate();
  ESP_ERROR_CHECK(esp_event_loop_create_default());

  ESP_ERROR_CHECK(esp_netif_init());
  esp_netif_create_default_wifi_sta();

  // setup wifi
  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_init(&cfg));
  wifi_config_t wifi_config = {
      .sta =
          {
              .ssid = WIFI_SSID,
              .password = WIFI_PWD,
          },
  };
  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
  ESP_ERROR_CHECK(esp_wifi_set_config(ESP_IF_WIFI_STA, &wifi_config));
  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));
  ESP_ERROR_CHECK(esp_wifi_start());

  // wait for wifi to connect
  xEventGroupWaitBits(s_wifi_event_group, BIT0, false, false, portMAX_DELAY);

  // tcp connect
  struct sockaddr_in dest_addr;
  inet_pton(AF_INET, SERVER_IP, &dest_addr.sin_addr);
  dest_addr.sin_family = AF_INET;
  dest_addr.sin_port = htons(SERVER_PORT);
  int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
  if (sock < 0) {
    printf("Failed to create socket\n");
    return;
  }
  while (1) {
    int err = connect(sock, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
    if (err != 0) {
      printf("Socket connect failed\n");
    } else {
      break;
    }
    vTaskDelay(1000 / portTICK_PERIOD_MS);
  }
  printf("Socket connected\n");

  // load initial weights
  void *weights = malloc(sizeof(weights_data));
  memcpy(weights, weights_data, sizeof(weights_data));

  // init network
  train_t *train = train_init(weights, sizeof(weights_data));

  // train loop
  fprintf(stderr, "Training...\n");
  uint8 *batch = malloc(sizeof(uint8) * BS * 28 * 28);
  uint8 *labels = malloc(sizeof(uint8) * BS * 1);
  for (int i = 0; i < 1000; i++) {
    // fetch batch
    int sample = rand() % (60000 - BS);
    fetch_batch(sock, sample, BS, batch, labels);

    float accuracy, loss;
    train_fn(train, batch, labels, &loss, &accuracy);
    fprintf(stderr, "loss: %f, accuracy: %f\r", loss, accuracy);
  }

  // cleanup from training
  train_free(train);

  // init network
  net_t *net = net_init(weights, sizeof(weights_data));

  // eval loop
  fprintf(stderr, "\nEvaluating...\n");
  float avg_accuracy = 0.0;
  for (int i = 0; i < 10000; i++) {
    // fetch batch
    fetch_batch(sock, i, 0, batch, labels);

    float accuracy;
    net_fn(net, batch, labels, &accuracy);
    avg_accuracy += accuracy;
    fprintf(stderr, "accuracy: %f\r", avg_accuracy / (i + 1));
  }
  avg_accuracy /= 10000.0;
  fprintf(stderr, "avg_accuracy: %f\n", avg_accuracy);

  // cleanup from eval
  net_free(net);
  free(weights);
}
