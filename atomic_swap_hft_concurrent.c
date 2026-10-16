#include <stdio.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <pthread.h>

typedef double (*model_fn)(double, double, double);

double predict_baseline(double, double, double);
double predict_normal(double, double, double);

static inline bool guard_normal(
    double realized_vol,
    double trade_intensity,
    double avg_trade_size)
{
    return
        (realized_vol >= 3.2500718817508836e-05 &&
         realized_vol <= 5.371523950454927e-05) &&
        (trade_intensity >= 15.703333333333333 &&
         trade_intensity <= 44.36) &&
        (avg_trade_size >= 1682.6495128056938 &&
         avg_trade_size <= 4825.72626559685);
}

_Atomic(model_fn) active_model = predict_baseline;

void controlled_swap(model_fn new_model)
{
    atomic_store_explicit(
        &active_model,
        new_model,
        memory_order_release
    );
}

double execute_model(
    double realized_vol,
    double trade_intensity,
    double avg_trade_size)
{
    model_fn model =
        atomic_load_explicit(
            &active_model,
            memory_order_acquire
        );

    if (model == predict_normal &&
        !guard_normal(
            realized_vol,
            trade_intensity,
            avg_trade_size))
    {
        model = predict_baseline;
    }

    return model(
        realized_vol,
        trade_intensity,
        avg_trade_size
    );
}

typedef struct {
    long iterations;
    long errors;
} worker_result;

void *worker(void *arg)
{
    worker_result *result = (worker_result *)arg;

    /*
     * Inside the normal VEC envelope.
     */
    double rv = 4.0e-05;
    double ti = 30.0;
    double size = 2500.0;

    double expected =
        predict_baseline(rv, ti, size);

    for (long i = 0; i < result->iterations; i++)
    {
        double value =
            execute_model(rv, ti, size);

        /*
         * The specialized model must remain
         * semantically equivalent to baseline.
         */
        if (value != expected)
            result->errors++;
    }

    return NULL;
}

int main(void)
{
    const int THREADS = 4;
    const long ITERATIONS = 250000;

    pthread_t threads[THREADS];
    worker_result results[THREADS];

    printf("=== CONCURRENT HFT ATOMIC SWAP TEST ===\n");

    printf("Initial model: baseline\n");

    for (int i = 0; i < THREADS; i++)
    {
        results[i].iterations = ITERATIONS;
        results[i].errors = 0;

        pthread_create(
            &threads[i],
            NULL,
            worker,
            &results[i]
        );
    }

    /*
     * Give the worker threads time to execute,
     * then atomically activate the specialized model.
     */
    for (volatile long i = 0; i < 1000000; i++)
        ;

    printf("Performing atomic swap: baseline -> specialized normal\n");

    controlled_swap(predict_normal);

    for (int i = 0; i < THREADS; i++)
        pthread_join(threads[i], NULL);

    long total_iterations = 0;
    long total_errors = 0;

    for (int i = 0; i < THREADS; i++)
    {
        total_iterations += results[i].iterations;
        total_errors += results[i].errors;
    }

    printf("Threads: %d\n", THREADS);
    printf("Iterations: %ld\n", total_iterations);
    printf("Semantic mismatches: %ld\n", total_errors);

    /*
     * Also verify an input outside the VEC envelope
     * after the swap.
     */
    double outside =
        execute_model(0.001, 100.0, 50.0);

    double baseline_outside =
        predict_baseline(0.001, 100.0, 50.0);

    printf("Outside-envelope baseline: %.9f\n",
           baseline_outside);

    printf("Outside-envelope active: %.9f\n",
           outside);

    if (total_errors == 0 &&
        outside == baseline_outside)
    {
        printf("CONCURRENT GUARDED ATOMIC SWAP: PASS\n");
    }
    else
    {
        printf("CONCURRENT GUARDED ATOMIC SWAP: FAIL\n");
    }

    return 0;
}
