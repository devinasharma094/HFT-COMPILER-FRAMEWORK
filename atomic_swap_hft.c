#include <stdio.h>
#include <stdbool.h>
#include <stdatomic.h>

typedef double (*model_fn)(double, double, double);

/* Actual generated HFT model functions */
double predict_baseline(double, double, double);
double predict_normal(double, double, double);

/*
 * Same VEC normal-regime guard generated from vec_big.json.
 * Kept local because generated_c/guards.c declares it static inline.
 */
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

/* Active model pointer */
_Atomic(model_fn) active_model = predict_baseline;

/* Atomic runtime swap */
void controlled_swap(model_fn new_model)
{
    atomic_store_explicit(
        &active_model,
        new_model,
        memory_order_release
    );
}

/* Guard-aware execution */
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

    /*
     * Specialized model is allowed only inside
     * its VEC validity envelope.
     */
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

static void run_case(
    const char *name,
    double realized_vol,
    double trade_intensity,
    double avg_trade_size)
{
    int guard = guard_normal(
        realized_vol,
        trade_intensity,
        avg_trade_size
    );

    double before = predict_baseline(
        realized_vol,
        trade_intensity,
        avg_trade_size
    );

    double after = execute_model(
        realized_vol,
        trade_intensity,
        avg_trade_size
    );

    printf("\n%s\n", name);
    printf("Guard: %s\n", guard ? "PASS" : "FAIL");
    printf("Baseline: %.9f\n", before);
    printf("Active result: %.9f\n", after);
    printf("Difference: %.12f\n", after - before);

    if (before == after)
        printf("EQUIVALENCE: PASS\n");
    else
        printf("EQUIVALENCE: FAIL\n");
}

int main(void)
{
    /*
     * Case 1: outside the normal VEC envelope.
     * The active specialized model must fall back to baseline.
     */
    double outside_rv = 0.001;
    double outside_ti = 100.0;
    double outside_size = 50.0;

    /*
     * Case 2: inside the normal VEC envelope.
     */
    double inside_rv = 4.0e-05;
    double inside_ti = 30.0;
    double inside_size = 2500.0;

    printf("=== GUARDED HFT ATOMIC SWAP ===\n");

    printf("\nBefore atomic swap: baseline active\n");

    run_case(
        "Outside VEC envelope",
        outside_rv,
        outside_ti,
        outside_size
    );

    run_case(
        "Inside VEC envelope",
        inside_rv,
        inside_ti,
        inside_size
    );

    /*
     * Atomically activate the specialized normal-regime model.
     */
    controlled_swap(predict_normal);

    printf("\nAfter atomic swap: specialized normal active\n");

    run_case(
        "Outside VEC envelope",
        outside_rv,
        outside_ti,
        outside_size
    );

    run_case(
        "Inside VEC envelope",
        inside_rv,
        inside_ti,
        inside_size
    );

    return 0;
}
