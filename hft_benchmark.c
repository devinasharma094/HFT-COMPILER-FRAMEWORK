#include <stdio.h>
#include <stdint.h>
#include <time.h>

int execute_decision(double realized_vol,
                     double trade_intensity,
                     double avg_trade_size,
                     double position,
                     double order_size);

static uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + t.tv_nsec;
}

int main(void)
{
    const size_t N = 100000;

    uint64_t start = now_ns();

    int accepted = 0;

    for (size_t i = 0; i < N; i++)
    {
        accepted += execute_decision(
            0.001,
            100.0,
            50.0,
            0.0,
            10.0
        );
    }

    uint64_t end = now_ns();

    double total_ns = (double)(end - start);
    double mean_ns = total_ns / N;

    printf("=== FULL HFT DECISION BENCHMARK ===\n");
    printf("iterations: %zu\n", N);
    printf("accepted: %d\n", accepted);
    printf("total time: %.0f ns\n", total_ns);
    printf("mean decision latency: %.2f ns\n", mean_ns);

    return 0;
}
