#ifndef A2LITE_BF592_H
#define A2LITE_BF592_H

#include <stdint.h>

#define A2LITE_CHANNELS       3
#define A2LITE_LAYERS         23
#define A2LITE_HEAD_KERNEL    16
#define A2LITE_Q_FRAC         14
#define A2LITE_Q_SCALE        16384

#define A2LITE_PARAMETER_COUNT 1871
#define A2LITE_PAYLOAD_BYTES   3742

#define A2LITE_HEADER_SIZE     32

/*
 * Sum of:
 *
 *     (K - 1) * D + 1
 *
 * over all 23 layers.
 */
#define A2LITE_HISTORY_SAMPLES 1241

/*
 * Three Q2.14 channels per history entry.
 */
#define A2LITE_HISTORY_WORDS \
    (A2LITE_HISTORY_SAMPLES * A2LITE_CHANNELS)

/*
 * 1241 * 3 * 2 = 7446 bytes.
 */
#define A2LITE_HISTORY_BYTES \
    (A2LITE_HISTORY_WORDS * sizeof(int16_t))

/*
 * Head source is the sum of 23 layer activations.
 */
#define A2LITE_HEAD_HISTORY 16

typedef struct
{
    /* 1871 Q2.14 parameters */
    int16_t q[A2LITE_PARAMETER_COUNT];

    /* Parameter pointers */
    const int16_t *rechannel;

    const int16_t *conv[A2LITE_LAYERS];
    const int16_t *conv_bias[A2LITE_LAYERS];
    const int16_t *mixin[A2LITE_LAYERS];
    const int16_t *l1[A2LITE_LAYERS];
    const int16_t *l1_bias[A2LITE_LAYERS];

    const int16_t *head;
    const int16_t *head_bias;
    const int16_t *head_scale;

    /*
     * One contiguous history arena.
     *
     * Every layer owns a section of this array.
     */
    int16_t history[A2LITE_HISTORY_WORDS];

    uint16_t history_offset[A2LITE_LAYERS];
    uint16_t history_size[A2LITE_LAYERS];
    uint16_t history_pos[A2LITE_LAYERS];

    /*
     * 16-sample head history.
     *
     * head_source is an INT32 sum of 23 activations.
     */
    int32_t head_history[A2LITE_HEAD_HISTORY]
                        [A2LITE_CHANNELS];

    uint16_t head_pos;

} A2LiteBF592;


/* -------------------------------------------------------------
 * Initialization
 * ------------------------------------------------------------- */

int a2lite_bf592_init(
    A2LiteBF592 *model,
    const int16_t *parameters
);


/* -------------------------------------------------------------
 * Binary loader
 * ------------------------------------------------------------- */

int a2lite_bf592_load(
    A2LiteBF592 *model,
    const char *filename
);


/* -------------------------------------------------------------
 * Runtime
 * ------------------------------------------------------------- */

void a2lite_bf592_reset(
    A2LiteBF592 *model
);

int16_t a2lite_bf592_process(
    A2LiteBF592 *model,
    int16_t input
);

#endif