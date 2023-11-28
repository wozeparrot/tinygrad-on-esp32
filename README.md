# tinygrad on esp32

A proof-of-concept for training tinygrad models on *the edge*.

Works by leveraging the tinygrad C backend to generate c code for the entire train step.

## Usage

run `CLANG=1 python compile.py` first to export the kernels.

drop a `config.h` file in `esp32/main/` with the following contents:

```c
#define WIFI_SSID "your wifi ssid"
#define WIFI_PWD "your wifi password"
#define SERVER_IP "dataserver address"
#define SERVER_PORT "dataserver port"
```

then build the esp-idf project in `esp32/` and flash.

make sure the data server is running on your computer.

```bash
python dataserver.py
```

## Tidbits

There isn't enough flash on the esp32 to actually store the entire dataset, so we are streaming it from somewhere.
