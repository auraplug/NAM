#ifndef A2LITE_BF592_H
#define A2LITE_BF592_H

#include <stdint.h>

/*
 * A2-Lite BF592-compatible fixed-point runtime.
 *
 * Parameters:
 *     Q4.12
 *
 * Runtime samples / activations:
 *     Q2.14
 *
 * Therefore:
 *
 *     Q4.12 * Q2.14 = Q6.26
 *
 * and accumulator normalization divides by 2^12.
 */

#define A2LITE_CHANNELS        3
#define A2LITE_LAYERS          23
#define A2LITE_HEAD_KERNEL     16

#define A2LITE_PARAM_Q_FRAC   12
#define A2LITE_PARAM_Q_SCALE  4096

#define A2LITE_SIGNAL_Q_FRAC  14
#define A2LITE_SIGNAL_Q_SCALE 16384

#define A2LITE_PARAMETER_COUNT 1871
#define A2LITE_PAYLOAD_BYTES   3742

#define A2LITE_HEADER_SIZE     32

#define A2LITE_MAGIC0          'A'
#define A2LITE_MAGIC1          '2'
#define A2LITE_MAGIC2          'L'
#define A2LITE_MAGIC3          'T'

/*
 * Sum of:
 *
 *     (K - 1) * D + 1
 *
 * over all 23 convolution layers.
 */
#define A2LITE_HISTORY_SAMPLES 1241

#define A2LITE_HISTORY_WORDS \
    (A2LITE_HISTORY_SAMPLES * A2LITE_CHANNELS)

#define A2LITE_HISTORY_BYTES \
    (A2LITE_HISTORY_WORDS * sizeof(int16_t))

#define A2LITE_HEAD_HISTORY 16

typedef struct
{
    /*
     * 1871 Q4.12 parameters.
     */
    int16_t q[A2LITE_PARAMETER_COUNT];

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
     * Q2.14 layer history.
     */
    int16_t history[A2LITE_HISTORY_WORDS];

    uint16_t history_offset[A2LITE_LAYERS];
    uint16_t history_size[A2LITE_LAYERS];
    uint16_t history_pos[A2LITE_LAYERS];

    /*
     * Sum of 23 Q2.14 activations.
     */
    int32_t head_history[A2LITE_HEAD_HISTORY]
                        [A2LITE_CHANNELS];

    uint16_t head_pos;

} A2LiteBF592;


/*
 * Initialize from 1871 Q4.12 parameters.
 */
int a2lite_bf592_init(
    A2LiteBF592 *model,
    const int16_t *parameters
);


/*
 * Load the 32-byte A2LT header + 3742-byte payload.
 *
 * The loader accepts only Q4.12 model files.
 */
int a2lite_bf592_load(
    A2LiteBF592 *model,
    const char *filename
);


/*
 * Reset streaming state.
 */
void a2lite_bf592_reset(
    A2LiteBF592 *model
);


/*
 * Process one Q2.14 input sample.
 *
 * Returns one Q2.14 output sample.
 */
int16_t a2lite_bf592_process(
    A2LiteBF592 *model,
    int16_t input
);

#endif