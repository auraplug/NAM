#include "a2lite_bf592.h"

#include <stdio.h>
#include <string.h>


/* =============================================================
 * Model geometry
 * ============================================================= */

static const uint8_t K[A2LITE_LAYERS] =
{
    6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6,
    15, 15,
    6, 6, 6, 6, 6, 6, 6
};


static const uint16_t D[A2LITE_LAYERS] =
{
    1, 3, 7, 17, 41, 101, 239,
    1, 3, 7, 17, 41, 101, 239,
    1, 13,
    1, 3, 7, 17, 41, 101, 239
};


/*
 * Current A2-Lite geometry requires no additional
 * accumulator shift.
 */
static const uint8_t LAYER_SHIFT[A2LITE_LAYERS] =
{
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0,
    0, 0, 0, 0, 0, 0, 0
};


/* =============================================================
 * Integer helpers
 * ============================================================= */

static int32_t sat16(
    int32_t x)
{
    if (x > 32767)
        return 32767;

    if (x < -32768)
        return -32768;

    return x;
}


static int32_t sat32(
    int64_t x)
{
    if (x > 2147483647LL)
        return 2147483647L;

    if (x < -2147483648LL)
        return -2147483648LL;

    return (int32_t)x;
}


/*
 * Round-half-away-from-zero division.
 */
static int32_t round_div(
    int64_t x,
    int32_t divisor)
{
    if (divisor <= 0)
        return 0;

    if (x >= 0)
    {
        return (int32_t)(
            (x + divisor / 2)
            / divisor
        );
    }

    return -(int32_t)(
        ((-x) + divisor / 2)
        / divisor
    );
}


/*
 * Q4.12 parameter × Q2.14 signal
 *
 * product:
 *
 *     Q4.12 * Q2.14 = Q6.26
 *
 * normalize:
 *
 *     Q6.26 -> Q2.14
 *
 * by dividing by 2^12.
 */
static int32_t mul_param_signal(
    int16_t parameter,
    int16_t signal)
{
    return round_div(
        (int64_t)parameter
        * (int64_t)signal,
        A2LITE_PARAM_Q_SCALE
    );
}


/*
 * Q4.12 bias into Q6.26 accumulator domain.
 */
static int64_t bias_to_acc(
    int16_t bias)
{
    return
        (int64_t)bias
        * A2LITE_PARAM_Q_SCALE;
}


/*
 * Q4.12 parameter × Q2.14 activation.
 *
 * Same operation as mul_param_signal(), but
 * kept separate semantically for the layer path.
 */
static int32_t mul_param_activation(
    int16_t parameter,
    int32_t activation)
{
    return round_div(
        (int64_t)parameter
        * (int64_t)activation,
        A2LITE_PARAM_Q_SCALE
    );
}


/*
 * Q6.26 accumulator -> Q2.14.
 *
 * shift is applied before the Q-format conversion.
 *
 * For shift=0:
 *
 *     acc / 4096
 */
static int32_t normalize_accumulator(
    int64_t acc,
    unsigned shift)
{
    if (shift != 0)
    {
        int64_t divisor = 1LL << shift;

        acc = round_div(
            acc,
            (int32_t)divisor
        );
    }

    return sat32(
        round_div(
            acc,
            A2LITE_PARAM_Q_SCALE
        )
    );
}


/*
 * LeakyReLU slope = 0.01.
 *
 * Runtime value remains Q2.14.
 */
static int32_t leaky_relu(
    int32_t x)
{
    if (x >= 0)
        return x;

    return round_div(
        (int64_t)x * 1,
        100
    );
}


/* =============================================================
 * Parameter layout
 * ============================================================= */

static int setup_parameters(
    A2LiteBF592 *m)
{
    int p = 0;
    int i;

    m->rechannel = &m->q[p];
    p += 3;

    for (i = 0; i < A2LITE_LAYERS; ++i)
    {
        int conv_count =
            (int)K[i] * 9;

        m->conv[i] =
            &m->q[p];

        p += conv_count;

        m->conv_bias[i] =
            &m->q[p];

        p += 3;

        m->mixin[i] =
            &m->q[p];

        p += 3;

        m->l1[i] =
            &m->q[p];

        p += 9;

        m->l1_bias[i] =
            &m->q[p];

        p += 3;
    }

    m->head =
        &m->q[p];

    p += 48;

    m->head_bias =
        &m->q[p];

    p++;

    m->head_scale =
        &m->q[p];

    p++;

    return
        p == A2LITE_PARAMETER_COUNT
        ? 0
        : -1;
}


/* =============================================================
 * History layout
 * ============================================================= */

static int setup_history(
    A2LiteBF592 *m)
{
    int i;
    int offset = 0;

    for (i = 0;
         i < A2LITE_LAYERS;
         ++i)
    {
        int size =
            ((int)K[i] - 1)
            * (int)D[i]
            + 1;

        m->history_offset[i] =
            (uint16_t)offset;

        m->history_size[i] =
            (uint16_t)size;

        m->history_pos[i] = 0;

        offset += size;
    }

    return
        offset == A2LITE_HISTORY_SAMPLES
        ? 0
        : -1;
}


/* =============================================================
 * History access
 * ============================================================= */

static inline int16_t *
history_ptr(
    A2LiteBF592 *m,
    int layer,
    int position)
{
    return
        &m->history[
            m->history_offset[layer]
            + position * A2LITE_CHANNELS
        ];
}


static void history_push(
    A2LiteBF592 *m,
    int layer,
    const int16_t x[3])
{
    uint16_t pos =
        m->history_pos[layer];

    int16_t *dst =
        history_ptr(
            m,
            layer,
            pos
        );

    dst[0] = x[0];
    dst[1] = x[1];
    dst[2] = x[2];

    pos++;

    if (pos >= m->history_size[layer])
        pos = 0;

    m->history_pos[layer] = pos;
}


static inline const int16_t *
history_get(
    const A2LiteBF592 *m,
    int layer,
    int delay)
{
    int pos;
    int size;

    size =
        (int)m->history_size[layer];

    pos =
        (int)m->history_pos[layer]
        - 1
        - delay;

    if (pos < 0)
    {
        pos %= size;

        if (pos < 0)
            pos += size;
    }

    return
        &m->history[
            m->history_offset[layer]
            + pos * A2LITE_CHANNELS
        ];
}


/* =============================================================
 * Initialization
 * ============================================================= */

int a2lite_bf592_init(
    A2LiteBF592 *m,
    const int16_t *parameters)
{
    if (!m || !parameters)
        return -1;

    memset(
        m,
        0,
        sizeof(*m)
    );

    memcpy(
        m->q,
        parameters,
        sizeof(m->q)
    );

    if (setup_parameters(m) != 0)
        return -2;

    if (setup_history(m) != 0)
        return -3;

    a2lite_bf592_reset(m);

    return 0;
}


/* =============================================================
 * Reset
 * ============================================================= */

void a2lite_bf592_reset(
    A2LiteBF592 *m)
{
    if (!m)
        return;

    memset(
        m->history,
        0,
        sizeof(m->history)
    );

    memset(
        m->head_history,
        0,
        sizeof(m->head_history)
    );

    memset(
        m->history_pos,
        0,
        sizeof(m->history_pos)
    );

    m->head_pos = 0;
}


/* =============================================================
 * One layer
 * ============================================================= */

static void process_layer(
    A2LiteBF592 *m,
    int layer,
    const int16_t input[3],
    int16_t condition,
    int16_t output[3],
    int32_t activated[3])
{
    int out_ch;
    int tap;

    const int kernel =
        (int)K[layer];

    const int dilation =
        (int)D[layer];

    const unsigned shift =
        LAYER_SHIFT[layer];


    /*
     * Current sample becomes available at delay zero.
     */
    history_push(
        m,
        layer,
        input
    );


    /*
     * ---------------------------------------------------------
     * Convolution + bias + condition mixin
     * ---------------------------------------------------------
     */

    for (out_ch = 0;
         out_ch < 3;
         ++out_ch)
    {
        int64_t acc;

        /*
         * Q4.12 bias -> Q6.26.
         */
        acc =
            bias_to_acc(
                m->conv_bias[layer][out_ch]
            );


        for (tap = 0;
             tap < kernel;
             ++tap)
        {
            int delay =
                (kernel - 1 - tap)
                * dilation;

            const int16_t *x =
                history_get(
                    m,
                    layer,
                    delay
                );

            const int16_t *w =
                &m->conv[layer][tap * 9];


            /*
             * tap * 9
             * + in_channel * 3
             * + out_channel
             */

            acc +=
                (int64_t)w[out_ch]
                * (int64_t)x[0];

            acc +=
                (int64_t)w[3 + out_ch]
                * (int64_t)x[1];

            acc +=
                (int64_t)w[6 + out_ch]
                * (int64_t)x[2];
        }


        /*
         * Condition is Q2.14.
         * Mixin is Q4.12.
         *
         * Result is Q6.26.
         */
        acc +=
            (int64_t)m->mixin[layer][out_ch]
            * (int64_t)condition;


        /*
         * Q6.26 -> Q2.14.
         */
        {
            int32_t z =
                normalize_accumulator(
                    acc,
                    shift
                );

            z =
                leaky_relu(z);

            activated[out_ch] = z;
        }
    }


    /*
     * ---------------------------------------------------------
     * 1x1 residual
     * ---------------------------------------------------------
     */

    for (out_ch = 0;
         out_ch < 3;
         ++out_ch)
    {
        int64_t acc;

        const int16_t *w =
            m->l1[layer];


        /*
         * Q4.12 bias -> Q6.26.
         */
        acc =
            bias_to_acc(
                m->l1_bias[layer][out_ch]
            );


        /*
         * Q4.12 × Q2.14 = Q6.26.
         */
        acc +=
            (int64_t)w[out_ch]
            * (int64_t)activated[0];

        acc +=
            (int64_t)w[3 + out_ch]
            * (int64_t)activated[1];

        acc +=
            (int64_t)w[6 + out_ch]
            * (int64_t)activated[2];


        /*
         * Q6.26 -> Q2.14.
         */
        {
            int32_t residual =
                normalize_accumulator(
                    acc,
                    shift
                );

            output[out_ch] =
                (int16_t)sat16(
                    (int32_t)input[out_ch]
                    + residual
                );
        }
    }
}


/* =============================================================
 * Main streaming sample
 * ============================================================= */

int16_t a2lite_bf592_process(
    A2LiteBF592 *m,
    int16_t input)
{
    int ch;
    int layer;

    int16_t x[3];
    int16_t next[3];

    int32_t activated[3];

    int32_t head_source[3];


    /*
     * ---------------------------------------------------------
     * Rechannel
     *
     * Q4.12 × Q2.14 -> Q2.14
     * ---------------------------------------------------------
     */

    for (ch = 0;
         ch < 3;
         ++ch)
    {
        int32_t v =
            mul_param_signal(
                m->rechannel[ch],
                input
            );

        x[ch] =
            (int16_t)sat16(v);
    }


    /*
     * ---------------------------------------------------------
     * Sum of all 23 layer activations.
     * ---------------------------------------------------------
     */

    head_source[0] = 0;
    head_source[1] = 0;
    head_source[2] = 0;


    for (layer = 0;
         layer < A2LITE_LAYERS;
         ++layer)
    {
        process_layer(
            m,
            layer,
            x,
            input,
            next,
            activated
        );

        head_source[0] +=
            activated[0];

        head_source[1] +=
            activated[1];

        head_source[2] +=
            activated[2];

        x[0] = next[0];
        x[1] = next[1];
        x[2] = next[2];
    }


    /*
     * ---------------------------------------------------------
     * Head history
     * ---------------------------------------------------------
     */

    {
        uint16_t pos =
            m->head_pos;

        int64_t acc;


        m->head_history[pos][0] =
            head_source[0];

        m->head_history[pos][1] =
            head_source[1];

        m->head_history[pos][2] =
            head_source[2];


        /*
         * Q4.12 bias -> Q6.26.
         */
        acc =
            bias_to_acc(
                *m->head_bias
            );


        /*
         * 16 taps × 3 channels.
         *
         * Head source is Q2.14.
         * Head weights are Q4.12.
         */
        for (int tap = 0;
             tap < A2LITE_HEAD_KERNEL;
             ++tap)
        {
            int hp =
                (int)pos
                - (A2LITE_HEAD_KERNEL - 1 - tap);

            if (hp < 0)
                hp += A2LITE_HEAD_KERNEL;

            const int32_t *src =
                m->head_history[hp];

            const int16_t *w =
                &m->head[tap * 3];


            acc +=
                (int64_t)w[0]
                * (int64_t)src[0];

            acc +=
                (int64_t)w[1]
                * (int64_t)src[1];

            acc +=
                (int64_t)w[2]
                * (int64_t)src[2];
        }


        pos++;

        if (pos >= A2LITE_HEAD_KERNEL)
            pos = 0;

        m->head_pos = pos;


        /*
         * Q6.26 -> Q2.14.
         */
        {
            int32_t y =
                normalize_accumulator(
                    acc,
                    0
                );


            /*
             * Q4.12 scale × Q2.14 output
             * -> Q6.26 -> Q2.14.
             */
            y =
                mul_param_activation(
                    *m->head_scale,
                    y
                );

            return
                (int16_t)sat16(y);
        }
    }
}


/* =============================================================
 * Binary loader
 * ============================================================= */

/*
 * Little-endian helpers.
 *
 * We deliberately do not depend on compiler struct packing.
 */
static uint16_t rd16(
    const unsigned char *p)
{
    return
        (uint16_t)p[0]
        | ((uint16_t)p[1] << 8);
}


static uint32_t rd32(
    const unsigned char *p)
{
    return
        (uint32_t)p[0]
        | ((uint32_t)p[1] << 8)
        | ((uint32_t)p[2] << 16)
        | ((uint32_t)p[3] << 24);
}


int a2lite_bf592_load(
    A2LiteBF592 *model,
    const char *filename)
{
    FILE *f;
    unsigned char header[A2LITE_HEADER_SIZE];
    unsigned char payload[A2LITE_PAYLOAD_BYTES];

    uint16_t version;
    uint16_t channels;
    uint16_t layers;
    uint16_t bottleneck;
    uint16_t head_kernel;
    uint16_t q_frac;

    uint32_t parameter_count;
    uint32_t payload_bytes;
    uint32_t header_bytes;
    uint32_t reserved;

    int i;


    if (!model || !filename)
        return -1;


    f = fopen(
        filename,
        "rb"
    );

    if (!f)
        return -2;


    if (fread(
            header,
            1,
            A2LITE_HEADER_SIZE,
            f
        ) != A2LITE_HEADER_SIZE)
    {
        fclose(f);
        return -3;
    }


    if (header[0] != A2LITE_MAGIC0 ||
        header[1] != A2LITE_MAGIC1 ||
        header[2] != A2LITE_MAGIC2 ||
        header[3] != A2LITE_MAGIC3)
    {
        fclose(f);
        return -4;
    }


    version =
        rd16(&header[4]);

    channels =
        rd16(&header[6]);

    layers =
        rd16(&header[8]);

    bottleneck =
        rd16(&header[10]);

    head_kernel =
        rd16(&header[12]);

    q_frac =
        rd16(&header[14]);

    parameter_count =
        rd32(&header[16]);

    payload_bytes =
        rd32(&header[20]);

    header_bytes =
        rd32(&header[24]);

    reserved =
        rd32(&header[28]);


    /*
     * Version is informational.
     *
     * The format characteristics themselves are what
     * determine whether the file can be used.
     */
    (void)version;
    (void)reserved;


    if (channels != A2LITE_CHANNELS ||
        layers != A2LITE_LAYERS ||
        bottleneck != 6 ||
        head_kernel != A2LITE_HEAD_KERNEL ||
        q_frac != A2LITE_PARAM_Q_FRAC ||
        parameter_count != A2LITE_PARAMETER_COUNT ||
        payload_bytes != A2LITE_PAYLOAD_BYTES ||
        header_bytes != A2LITE_HEADER_SIZE)
    {
        fclose(f);
        return -5;
    }


    if (fread(
            payload,
            1,
            A2LITE_PAYLOAD_BYTES,
            f
        ) != A2LITE_PAYLOAD_BYTES)
    {
        fclose(f);
        return -6;
    }


    fclose(f);


    /*
     * Explicit little-endian int16 conversion.
     */
    for (i = 0;
         i < A2LITE_PARAMETER_COUNT;
         ++i)
    {
        uint16_t u =
            (uint16_t)payload[i * 2]
            | ((uint16_t)payload[i * 2 + 1] << 8);

        model->q[i] =
            (int16_t)u;
    }


    if (setup_parameters(model) != 0)
        return -7;

    if (setup_history(model) != 0)
        return -8;


    a2lite_bf592_reset(model);

    return 0;
}