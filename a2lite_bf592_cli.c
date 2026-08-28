#include "a2lite_bf592.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>


typedef struct
{
    FILE *file;

    uint16_t channels;
    uint16_t bits_per_sample;

    uint32_t sample_rate;
    uint32_t data_bytes;

    long data_offset;

} WavReader;


typedef struct
{
    FILE *file;

    uint16_t channels;
    uint16_t bits_per_sample;

    uint32_t sample_rate;

    uint32_t data_bytes;

} WavWriter;


/* =============================================================
 * Little endian
 * ============================================================= */

static uint16_t read_u16(
    FILE *f)
{
    unsigned char b[2];

    if (fread(b, 1, 2, f) != 2)
        return 0;

    return
        (uint16_t)b[0]
        | ((uint16_t)b[1] << 8);
}


static uint32_t read_u32(
    FILE *f)
{
    unsigned char b[4];

    if (fread(b, 1, 4, f) != 4)
        return 0;

    return
        (uint32_t)b[0]
        | ((uint32_t)b[1] << 8)
        | ((uint32_t)b[2] << 16)
        | ((uint32_t)b[3] << 24);
}


static void write_u16(
    FILE *f,
    uint16_t x)
{
    unsigned char b[2];

    b[0] = (unsigned char)(x & 0xff);
    b[1] = (unsigned char)(x >> 8);

    fwrite(b, 1, 2, f);
}


static void write_u32(
    FILE *f,
    uint32_t x)
{
    unsigned char b[4];

    b[0] = (unsigned char)(x & 0xff);
    b[1] = (unsigned char)((x >> 8) & 0xff);
    b[2] = (unsigned char)((x >> 16) & 0xff);
    b[3] = (unsigned char)((x >> 24) & 0xff);

    fwrite(b, 1, 4, f);
}


/* =============================================================
 * WAV input
 * ============================================================= */

static int wav_read_open(
    WavReader *w,
    const char *filename)
{
    FILE *f;

    char riff[4];
    char wave[4];

    int fmt_found = 0;
    int data_found = 0;


    memset(
        w,
        0,
        sizeof(*w)
    );


    f = fopen(
        filename,
        "rb"
    );

    if (!f)
        return -1;


    if (fread(riff, 1, 4, f) != 4 ||
        memcmp(riff, "RIFF", 4) != 0)
    {
        fclose(f);
        return -2;
    }


    (void)read_u32(f);


    if (fread(wave, 1, 4, f) != 4 ||
        memcmp(wave, "WAVE", 4) != 0)
    {
        fclose(f);
        return -3;
    }


    while (!fmt_found || !data_found)
    {
        char id[4];
        uint32_t size;
        long chunk_end;


        if (fread(id, 1, 4, f) != 4)
            break;


        size =
            read_u32(f);

        chunk_end =
            ftell(f) + (long)size;


        if (memcmp(id, "fmt ", 4) == 0)
        {
            uint16_t format =
                read_u16(f);

            uint16_t channels =
                read_u16(f);

            uint32_t rate =
                read_u32(f);

            (void)read_u32(f);
            (void)read_u16(f);

            uint16_t bits =
                read_u16(f);


            if (format != 1 ||
                channels != 1 ||
                bits != 16)
            {
                fclose(f);
                return -4;
            }


            w->channels =
                channels;

            w->sample_rate =
                rate;

            w->bits_per_sample =
                bits;

            fmt_found = 1;
        }
        else if (memcmp(id, "data", 4) == 0)
        {
            w->data_bytes =
                size;

            w->data_offset =
                ftell(f);

            data_found = 1;
        }


        fseek(
            f,
            chunk_end,
            SEEK_SET
        );

        /*
         * RIFF chunks are word aligned.
         */
        if (size & 1)
            fseek(f, 1, SEEK_CUR);
    }


    if (!fmt_found || !data_found)
    {
        fclose(f);
        return -5;
    }


    /*
     * Position at first sample.
     */
    fseek(
        f,
        w->data_offset,
        SEEK_SET
    );


    w->file = f;

    return 0;
}


static int wav_read_sample(
    WavReader *w,
    int16_t *sample)
{
    unsigned char b[2];

    if (fread(b, 1, 2, w->file) != 2)
        return 0;

    *sample =
        (int16_t)(
            (uint16_t)b[0]
            | ((uint16_t)b[1] << 8)
        );

    return 1;
}


static void wav_read_close(
    WavReader *w)
{
    if (w->file)
    {
        fclose(w->file);
        w->file = NULL;
    }
}


/* =============================================================
 * WAV output
 * ============================================================= */

static int wav_write_open(
    WavWriter *w,
    const char *filename,
    uint32_t sample_rate)
{
    FILE *f;


    memset(
        w,
        0,
        sizeof(*w)
    );


    f = fopen(
        filename,
        "wb"
    );

    if (!f)
        return -1;


    /*
     * Write a standard PCM WAV header.
     *
     * Sizes are patched at close.
     */
    fwrite("RIFF", 1, 4, f);
    write_u32(f, 0);

    fwrite("WAVE", 1, 4, f);

    fwrite("fmt ", 1, 4, f);
    write_u32(f, 16);

    write_u16(f, 1);
    write_u16(f, 1);

    write_u32(f, sample_rate);

    write_u32(
        f,
        sample_rate * 2
    );

    write_u16(f, 2);
    write_u16(f, 16);


    fwrite("data", 1, 4, f);
    write_u32(f, 0);


    w->file = f;
    w->channels = 1;
    w->bits_per_sample = 16;
    w->sample_rate = sample_rate;
    w->data_bytes = 0;

    return 0;
}


static void wav_write_sample(
    WavWriter *w,
    int16_t sample)
{
    unsigned char b[2];

    b[0] =
        (unsigned char)(
            (uint16_t)sample & 0xff
        );

    b[1] =
        (unsigned char)(
            ((uint16_t)sample >> 8)
        );

    fwrite(
        b,
        1,
        2,
        w->file
    );

    w->data_bytes += 2;
}


static void wav_write_close(
    WavWriter *w)
{
    long end;
    uint32_t riff_size;


    if (!w->file)
        return;


    end =
        ftell(w->file);

    riff_size =
        36 + w->data_bytes;


    /*
     * RIFF size at byte 4.
     */
    fseek(
        w->file,
        4,
        SEEK_SET
    );

    write_u32(
        w->file,
        riff_size
    );


    /*
     * data size at byte 40.
     */
    fseek(
        w->file,
        40,
        SEEK_SET
    );

    write_u32(
        w->file,
        w->data_bytes
    );


    fseek(
        w->file,
        end,
        SEEK_SET
    );


    fclose(
        w->file
    );

    w->file = NULL;
}


/* =============================================================
 * Main
 * ============================================================= */

int main(
    int argc,
    char **argv)
{
    const char *model_file;
    const char *input_file;
    const char *output_file;

    A2LiteBF592 model;

    WavReader input;
    WavWriter output;

    int16_t sample;
    int16_t result;

    uint32_t sample_count = 0;

    int rc;


    printf(
        "A2-Lite BF592 Q4.12 CLI\n"
    );

    printf(
        "========================\n"
    );


    if (argc != 4)
    {
        fprintf(
            stderr,
            "\nUsage:\n"
            "  %s model.bin input.wav output.wav\n\n",
            argv[0]
        );

        return 2;
    }


    model_file = argv[1];
    input_file = argv[2];
    output_file = argv[3];


    printf(
        "Model      : %s\n",
        model_file
    );

    printf(
        "Input      : %s\n",
        input_file
    );

    printf(
        "Output     : %s\n",
        output_file
    );

    printf(
        "Parameters : %d\n",
        A2LITE_PARAMETER_COUNT
    );

    printf(
        "Format     : Q4.12\n"
    );

    printf(
        "Payload    : %d bytes\n",
        A2LITE_PAYLOAD_BYTES
    );

    printf(
        "History    : %d bytes\n\n",
        A2LITE_HISTORY_BYTES
    );


    /*
     * ---------------------------------------------------------
     * Model
     * ---------------------------------------------------------
     */

    rc =
        a2lite_bf592_load(
            &model,
            model_file
        );

    if (rc != 0)
    {
        fprintf(
            stderr,
            "ERROR: model load failed (%d)\n",
            rc
        );

        return 3;
    }


    printf(
        "Model loaded.\n"
    );


    /*
     * ---------------------------------------------------------
     * WAV input
     * ---------------------------------------------------------
     */

    rc =
        wav_read_open(
            &input,
            input_file
        );

    if (rc != 0)
    {
        fprintf(
            stderr,
            "ERROR: WAV input failed (%d)\n",
            rc
        );

        return 4;
    }


    printf(
        "WAV        : %u Hz, mono, 16-bit\n",
        (unsigned)input.sample_rate
    );


    /*
     * ---------------------------------------------------------
     * WAV output
     * ---------------------------------------------------------
     */

    rc =
        wav_write_open(
            &output,
            output_file,
            input.sample_rate
        );

    if (rc != 0)
    {
        fprintf(
            stderr,
            "ERROR: WAV output failed (%d)\n",
            rc
        );

        wav_read_close(&input);

        return 5;
    }


    /*
     * ---------------------------------------------------------
     * Streaming DSP
     * ---------------------------------------------------------
     */

    a2lite_bf592_reset(
        &model
    );


    printf(
        "Processing...\n"
    );


    while (wav_read_sample(
               &input,
               &sample))
    {
        result =
            a2lite_bf592_process(
                &model,
                sample
            );

        wav_write_sample(
            &output,
            result
        );

        sample_count++;
    }


    wav_read_close(
        &input
    );

    wav_write_close(
        &output
    );


    printf(
        "Done.\n"
    );

    printf(
        "Samples    : %u\n",
        (unsigned)sample_count
    );

    printf(
        "Output     : %s\n",
        output_file
    );


    return 0;
}